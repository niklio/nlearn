import os
"""
logging_utils.py — Training metrics: timing, memory, validation, and formatting.

Keeps the training loop clean by encapsulating all step-level instrumentation.
"""

import resource
import time

import jax
import wandb

from leaderboard_client import post_entry

# Compute budget (in FLOPs) at which val loss is captured for ranking, so runs
# of different lengths/speeds are compared at equal compute. Overridable via env.
DEFAULT_FLOP_BUDGET = float(os.environ.get("LEADERBOARD_FLOP_BUDGET", 1e16))


def benchmark_peak_tflops(dtype=jax.numpy.float16, n=2048, warmup=3, trials=10):
    """
    Measure achievable peak TFLOPS by timing a large square matmul through the
    SAME path the model uses for its matmuls.

    On IREE-Metal that means the hand-authored simdgroup GEMM kernel (~2.4-2.9
    TFLOPS), NOT IREE's naive jnp.matmul codegen (~0.7) — otherwise MFU is computed
    against the wrong denominator and reads >100%. Uses fp16 (bf16 is unsupported on
    metal-spirv). MFU then means "fraction of our GEMM-kernel throughput the full
    step sustains" — the gap below 100% is the non-GEMM overhead (flash, elementwise,
    cross-entropy, host dispatch gaps). FLOPs for (n,n)@(n,n) = 2*n^3.
    """
    import jax.numpy as jnp
    try:
        from gemm_iree import matmul as _mm   # routes to the Metal GEMM kernel on IREE
    except Exception:
        _mm = jnp.matmul

    a = jnp.ones((n, n), dtype=dtype)
    b = jnp.ones((n, n), dtype=dtype)
    matmul = jax.jit(_mm)

    # Warmup: compile + fill caches
    for _ in range(warmup):
        c = matmul(a, b)
        jax.block_until_ready(c)

    # Timed trials
    best = float('inf')
    for _ in range(trials):
        t0 = time.time()
        c = matmul(a, b)
        jax.block_until_ready(c)
        best = min(best, time.time() - t0)

    flops = 2 * n ** 3
    tflops = flops / best / 1e12
    return tflops


def get_peak_memory_mb():
    """Peak process RSS in MB. On Apple Silicon this includes GPU memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def estimate_flops_per_step(n_params, seq_len, batch_size):
    """
    Estimate FLOPs for one training step (forward + backward).

    Standard approximation from the PaLM and Chinchilla papers:
      Forward pass  ≈ 2 × n_params × seq_len  (one matmul = 2 FLOPs per param per token)
      Backward pass ≈ 2 × forward              (grad w.r.t. activations + grad w.r.t. params)
      Total         ≈ 6 × n_params × tokens_per_step

    This counts dense matmul FLOPs only — the dominant cost. It excludes
    softmax, layer norm, and embedding lookups (< 5% of total).
    """
    tokens_per_step = batch_size * seq_len
    return 6 * n_params * tokens_per_step


class StepTimer:
    """Context-manager style timer for measuring step phases."""

    def __init__(self):
        self.data_time = 0.0
        self.train_time = 0.0
        self.step_time = 0.0
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def mark_data(self):
        now = time.time()
        self.data_time = now - self._t0
        self._t0 = now

    def mark_train(self):
        now = time.time()
        self.train_time = now - self._t0
        self.step_time = self.data_time + self.train_time


class TrainingLogger:
    """Collects per-step metrics, logs to W&B, and prints summaries."""

    def __init__(self, eval_loss_fn, val_batches_fn, val_every=100,
                 n_params=0, seq_len=512, batch_size=32, hw_peak_tflops=0.0,
                 run_name=None, dataset="", flop_budget=None):
        self._eval_loss_fn = eval_loss_fn
        self._val_batches_fn = val_batches_fn
        self._val_every = val_every
        self._step_times = []
        self._total_flops = 0
        self._flops_per_step = estimate_flops_per_step(n_params, seq_len, batch_size)
        self._hw_peak_flops = hw_peak_tflops * 1e12  # convert TFLOPS → FLOPS

        # --- Leaderboard state ---
        self._tokens_per_step = batch_size * seq_len
        self._dataset = dataset
        self._flop_budget = flop_budget if flop_budget is not None else DEFAULT_FLOP_BUDGET
        self._run_name = run_name or f"run-{int(time.time())}"
        self._best_val_loss = float("inf")
        self._loss_at_budget = None  # val loss captured when total_flops crosses budget
        self._last_train_loss = None
        self._last_tflops = 0.0
        self._last_mfu = 0.0
        self._last_step_time = 0.0
        self._peak_mem_mb = 0.0

        # Tell W&B to also plot loss metrics against cumulative TFLOP (a count of
        # 10^12 ops, not the per-second rate) as x-axis.
        wandb.define_metric("loss/total_tflop")
        wandb.define_metric("loss/by_tflop/*", step_metric="loss/total_tflop")

    def _leaderboard_metrics(self, step, n_steps):
        """Snapshot of the run as leaderboard columns (pretraining board)."""
        return {
            "val_loss": None if self._best_val_loss == float("inf") else self._best_val_loss,
            "loss_at_budget": self._loss_at_budget,
            "train_loss": self._last_train_loss,
            "tflops": self._last_tflops,
            "mfu": self._last_mfu,
            "tokens": (step + 1) * self._tokens_per_step,
            "step_time": self._last_step_time,
            "peak_mem_gb": self._peak_mem_mb / 1024.0,
            "total_flops": self._total_flops,
            "step": step,
            "n_steps": n_steps,
            "dataset": self._dataset,
        }

    def _post_leaderboard(self, step, n_steps, status):
        post_entry("pretraining", {
            "id": self._run_name,
            "name": self._run_name,
            "status": status,
            "metrics": self._leaderboard_metrics(step, n_steps),
        })

    def finalize(self, loss, step, n_steps):
        """Post the final 'done' row. Call once after the training loop."""
        self._last_train_loss = float(loss)
        self._post_leaderboard(step, n_steps, status="done")


    def _compute_val_loss(self, params):
        if os.environ.get("NLEARN_NO_VAL") == "1":   # diagnostic: skip validation
            return float('nan')
        # Cap the number of validation batches. Evaluating the *entire* held-out
        # set (~100k tokens => ~100+ forward passes at seq=512) in one shot trips
        # the IREE Metal HAL accumulation hang at seq>=256; a small sample gives a
        # stable estimate and keeps the dispatch/allocation volume bounded.
        max_batches = int(os.environ.get("NLEARN_VAL_BATCHES", "20"))
        val_losses = []
        for vb in self._val_batches_fn():
            val_losses.append(float(self._eval_loss_fn(params, vb)))
            if len(val_losses) >= max_batches:
                break
        return sum(val_losses) / len(val_losses) if val_losses else float('nan')

    def log_step(self, step, loss, timer, params, n_steps):
        """Log metrics for a single training step. Returns metrics dict."""
        self._step_times.append(timer.step_time)
        self._total_flops += self._flops_per_step
        mem_mb = get_peak_memory_mb()

        # Achieved throughput and MFU (fraction of hardware peak used for model math)
        tflops = 0.0
        mfu = 0.0
        if timer.train_time > 0:
            tflops = self._flops_per_step / timer.train_time / 1e12
            if self._hw_peak_flops > 0:
                mfu = self._flops_per_step / (self._hw_peak_flops * timer.train_time)

        # Track latest values for the leaderboard snapshot.
        self._last_train_loss = float(loss)
        self._last_tflops = tflops
        self._last_mfu = mfu
        self._last_step_time = timer.step_time
        self._peak_mem_mb = mem_mb

        metrics = {
            "loss/train": float(loss),
            "hardware/step_time": timer.step_time,
            "hardware/data_time": timer.data_time,
            "hardware/train_time": timer.train_time,
            "hardware/peak_memory_mb": mem_mb,
            "hardware/tflops": tflops,
            "hardware/mfu": mfu,
            "loss/total_tflop": self._total_flops / 1e12,
        }

        # Loss indexed by cumulative TFLOP (live-updating chart)
        metrics["loss/by_tflop/train"] = float(loss)

        # Periodic validation
        do_val = (step % self._val_every == 0)
        if do_val:
            val_loss = self._compute_val_loss(params)
            metrics["loss/val"] = val_loss
            metrics["loss/by_tflop/val"] = val_loss

            # Leaderboard bookkeeping: best val loss + loss at the FLOP budget.
            if val_loss == val_loss:  # skip NaN (e.g. NLEARN_NO_VAL)
                self._best_val_loss = min(self._best_val_loss, val_loss)
                if self._loss_at_budget is None and self._total_flops >= self._flop_budget:
                    self._loss_at_budget = val_loss
            # Live upsert so the board updates mid-run (throttled to val steps).
            self._post_leaderboard(step, n_steps, status="running")

        # Print to console
        if do_val:
            print(f"Step {step:>4}  loss: {loss:.4f}  val_loss: {val_loss:.4f}  "
                  f"tflops: {tflops:.2f}  mfu: {mfu:.1%}  mem: {mem_mb:.0f}MB  "
                  f"step: {timer.step_time:.2f}s "
                  f"(data: {timer.data_time:.3f}s  train: {timer.train_time:.2f}s)")
        elif step % 10 == 0 or step < 5:   # frequent prints → supervisor hang-detection
            print(f"Step {step:>4}  loss: {loss:.4f}  tflops: {tflops:.2f}  mfu: {mfu:.1%}  mem: {mem_mb:.0f}MB  "
                  f"step: {timer.step_time:.2f}s "
                  f"(data: {timer.data_time:.3f}s  train: {timer.train_time:.2f}s)")

        wandb.log(metrics)
        return metrics

    def print_summary(self, loss):
        """Print final training summary."""
        avg = sum(self._step_times[1:]) / max(len(self._step_times) - 1, 1)
        print(f"\nTraining complete. Final loss: {loss:.4f}")
        print(f"Avg step time: {avg:.2f}s (excluding first step JIT)")
        print(f"Total TFLOP: {self._total_flops / 1e12:.2f}\n")
