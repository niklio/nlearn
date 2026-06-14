"""
logging_utils.py — Training metrics: timing, memory, validation, and formatting.

Keeps the training loop clean by encapsulating all step-level instrumentation.
"""

import resource
import time

import jax
import wandb


def benchmark_peak_tflops(dtype=jax.numpy.bfloat16, n=2048, warmup=3, trials=10):
    """
    Measure actual hardware peak TFLOPS by timing a large matmul.

    Runs an (n × n) @ (n × n) matmul and computes throughput.
    FLOPs for a matmul of two (n, n) matrices = 2 * n^3.
    """
    import jax.numpy as jnp

    a = jnp.ones((n, n), dtype=dtype)
    b = jnp.ones((n, n), dtype=dtype)
    matmul = jax.jit(jnp.matmul)

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
                 n_params=0, seq_len=512, batch_size=32, hw_peak_tflops=0.0):
        self._eval_loss_fn = eval_loss_fn
        self._val_batches_fn = val_batches_fn
        self._val_every = val_every
        self._step_times = []
        self._total_flops = 0
        self._flops_per_step = estimate_flops_per_step(n_params, seq_len, batch_size)
        self._hw_peak_flops = hw_peak_tflops * 1e12  # convert TFLOPS → FLOPS

        # Tell W&B to also plot loss metrics against total_flops as x-axis.
        wandb.define_metric("loss/total_flops")
        wandb.define_metric("loss/by_flops/*", step_metric="loss/total_flops")


    def _compute_val_loss(self, params):
        val_losses = []
        for vb in self._val_batches_fn():
            vl = self._eval_loss_fn(params, vb)
            val_losses.append(float(vl))
        return sum(val_losses) / len(val_losses) if val_losses else float('nan')

    def log_step(self, step, loss, timer, params, n_steps):
        """Log metrics for a single training step. Returns metrics dict."""
        self._step_times.append(timer.step_time)
        self._total_flops += self._flops_per_step
        mem_mb = get_peak_memory_mb()

        # MFU: fraction of hardware peak FLOPS used for model math
        mfu = 0.0
        if self._hw_peak_flops > 0 and timer.train_time > 0:
            mfu = self._flops_per_step / (self._hw_peak_flops * timer.train_time)

        metrics = {
            "loss/train": float(loss),
            "hardware/step_time": timer.step_time,
            "hardware/data_time": timer.data_time,
            "hardware/train_time": timer.train_time,
            "hardware/peak_memory_mb": mem_mb,
            "hardware/mfu": mfu,
            "loss/total_flops": self._total_flops,
        }

        # Loss indexed by FLOPs (live-updating chart)
        metrics["loss/by_flops/train"] = float(loss)

        # Periodic validation
        do_val = (step % self._val_every == 0)
        if do_val:
            val_loss = self._compute_val_loss(params)
            metrics["loss/val"] = val_loss
            metrics["loss/by_flops/val"] = val_loss

        # Print to console
        if do_val:
            print(f"Step {step:>4}  loss: {loss:.4f}  val_loss: {val_loss:.4f}  "
                  f"mfu: {mfu:.1%}  mem: {mem_mb:.0f}MB  "
                  f"step: {timer.step_time:.2f}s "
                  f"(data: {timer.data_time:.3f}s  train: {timer.train_time:.2f}s)")
        elif step % 100 == 0 or step < 5:
            print(f"Step {step:>4}  loss: {loss:.4f}  mfu: {mfu:.1%}  mem: {mem_mb:.0f}MB  "
                  f"step: {timer.step_time:.2f}s "
                  f"(data: {timer.data_time:.3f}s  train: {timer.train_time:.2f}s)")

        wandb.log(metrics)
        return metrics

    def print_summary(self, loss):
        """Print final training summary."""
        avg = sum(self._step_times[1:]) / max(len(self._step_times) - 1, 1)
        print(f"\nTraining complete. Final loss: {loss:.4f}")
        print(f"Avg step time: {avg:.2f}s (excluding first step JIT)")
        print(f"Total FLOPs: {self._total_flops:.2e}\n")
