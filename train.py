import sys
import signal
import threading
import queue as queue_mod
from itertools import cycle
import jax
import jax.numpy as jnp
from jax import random
import optax
import wandb
import pickle
import os
import numpy as np
from datasets import load_dataset

import tiktoken

from model import init_model, model_forward, model_forward_features, generate, VOCAB_SIZE, D_MODEL
from attention import print_attention_config

# ---------------------------------------------------------------------------
# CHECKPOINTING
#
# Saving model weights to disk so we can:
#   1. Resume training if it crashes
#   2. Load trained weights later for generation
#   3. Compare checkpoints from different points in training
#
# We use pickle — Python's built-in serialization. It handles nested dicts
# and JAX arrays cleanly. For very large models (>10GB) you'd use orbax
# (JAX's official checkpointing library), but pickle is fine at our scale.
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"

def save_checkpoint(params, step, run_name=None):
    """Save model parameters to disk at a given training step."""
    subdir = os.path.join(CHECKPOINT_DIR, run_name) if run_name else CHECKPOINT_DIR
    os.makedirs(subdir, exist_ok=True)

    path = os.path.join(subdir, f"step_{step:06d}.pkl")
    # Zero-pad the step number so filenames sort correctly (step_000500.pkl, etc.)

    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    # jax.device_get() moves all JAX arrays from GPU memory to CPU numpy arrays.
    # This is necessary before pickling — you can't serialize GPU memory directly.

    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path):
    """Load model parameters from a checkpoint file."""
    with open(path, 'rb') as f:
        params = pickle.load(f)
    # pickle.load restores the nested dict of numpy arrays.
    # JAX will automatically move them back to the GPU when used in computations.
    return params

# ---------------------------------------------------------------------------
# SECTION 1: LOSS FUNCTION
#
# For a language model, the task is: given tokens so far, predict the next one.
# We train on (input, target) pairs created by shifting the same sequence by 1:
#
#   Full sequence:  [h, e, l, l, o, !]
#   Input:          [h, e, l, l, o]     ← what the model sees
#   Target:         [e, l, l, o, !]     ← what the model must predict at each step
#
# The loss measures how wrong the model's predictions are using cross-entropy.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CHUNKED CROSS-ENTROPY
#
# The standard approach computes logits for all VOCAB_SIZE tokens at once:
#   logits = x @ lm_head   →   (seq_len, 50257)   — 103 MB per sequence
# Across a batch of 256 via vmap, that's ~26 GB. Metal can't hold that.
#
# Chunked CE avoids materialising the full logit matrix. Instead we:
#   1. Split lm_head into VOCAB_CHUNK_SIZE columns at a time
#   2. Stream through chunks with lax.scan, accumulating:
#        - running log-sum-exp  (the normaliser for softmax)
#        - the correct token's logit  (the numerator)
#   3. Combine at the end: loss = -(correct_logit - log_sum_exp).mean()
#
# Peak logit memory = batch_size × seq_len × VOCAB_CHUNK_SIZE × 4 bytes
# At bs=256, seq_len=512, chunk=1024: 536 MB vs 26 GB unChunked.
# ---------------------------------------------------------------------------

VOCAB_CHUNK_SIZE = 4096
# 50257 is not divisible by 4096; pad to 53248 = 13 × 4096.
_PAD          = (-VOCAB_SIZE) % VOCAB_CHUNK_SIZE           # 2991
_PADDED_VOCAB = VOCAB_SIZE + _PAD                          # 53248
_N_CHUNKS     = _PADDED_VOCAB // VOCAB_CHUNK_SIZE          # 13

# Boolean mask: True for real vocab tokens, False for padding (only last chunk differs).
# Shape (n_chunks, chunk_size) — passed through lax.scan to suppress padded positions.
_VOCAB_MASK = (jnp.arange(_PADDED_VOCAB) < VOCAB_SIZE).reshape(_N_CHUNKS, VOCAB_CHUNK_SIZE)


def cross_entropy_loss(params, token_ids):
    """
    Chunked cross-entropy: same result as the naive version but never
    materialises the full (seq_len, VOCAB_SIZE) logit matrix.

    params:    full model parameter dict
    token_ids: 1D integer array of shape (seq_len + 1,)
    Returns:   scalar loss
    """
    input_ids  = token_ids[:-1]
    target_ids = token_ids[1:]
    seq_len    = input_ids.shape[0]

    # Hidden states before the output projection: (seq_len, D_MODEL)
    x = model_forward_features(params, input_ids)

    # Pad lm_head with zeros (NOT -inf — that causes NaN in the matmul when
    # x has mixed signs). We suppress padded positions via _VOCAB_MASK after.
    W = params['lm_head']  # (D_MODEL, VOCAB_SIZE)
    if _PAD > 0:
        W = jnp.concatenate([W, jnp.zeros((D_MODEL, _PAD))], axis=1)

    # Reshape into (n_chunks, D_MODEL, chunk_size) for the loop
    W_chunks = W.reshape(D_MODEL, _N_CHUNKS, VOCAB_CHUNK_SIZE).transpose(1, 0, 2)

    # Python for-loop: JIT unrolls this statically at trace time — simpler
    # Metal kernels than lax.scan while still capping per-step memory to
    # (seq_len × VOCAB_CHUNK_SIZE) rather than (seq_len × VOCAB_SIZE).
    running_max   = jnp.full((seq_len,), -jnp.inf)
    running_sum_exp = jnp.zeros((seq_len,))
    correct_logit = jnp.zeros((seq_len,))

    for i in range(_N_CHUNKS):
        chunk_start  = i * VOCAB_CHUNK_SIZE
        chunk_logits = x @ W_chunks[i]                     # (seq_len, VOCAB_CHUNK_SIZE)
        chunk_logits = jnp.where(_VOCAB_MASK[i], chunk_logits, -jnp.inf)

        chunk_max = chunk_logits.max(axis=-1)
        new_max   = jnp.maximum(running_max, chunk_max)

        running_sum_exp = (
            running_sum_exp * jnp.exp(running_max - new_max)
            + jnp.exp(chunk_logits - new_max[:, None]).sum(axis=-1)
        )
        running_max = new_max

        in_chunk      = (target_ids >= chunk_start) & (target_ids < chunk_start + VOCAB_CHUNK_SIZE)
        safe_idx      = jnp.clip(target_ids - chunk_start, 0, VOCAB_CHUNK_SIZE - 1)
        chunk_correct = chunk_logits[jnp.arange(seq_len), safe_idx]
        correct_logit = jnp.where(in_chunk, chunk_correct, correct_logit)

    final_max, final_sum_exp = running_max, running_sum_exp

    log_Z = jnp.log(final_sum_exp) + final_max
    return -(correct_logit - log_Z).mean()


# ---------------------------------------------------------------------------
# SECTION 2: OPTIMIZER (ADAM + LEARNING RATE SCHEDULE)
#
# We now use a learning rate *schedule* instead of a fixed rate:
#
#   Warmup phase (steps 0 → WARMUP_STEPS):
#     LR ramps linearly from 0 → PEAK_LR.
#     Starting at 0 prevents large early updates from destabilizing the
#     randomly initialized weights before the optimizer has built up momentum.
#
#   Cosine decay phase (steps WARMUP_STEPS → N_STEPS):
#     LR follows a cosine curve from PEAK_LR → END_LR.
#     Gradually slowing down lets the model make fine-grained adjustments
#     as it approaches convergence, instead of overshooting the minimum.
#
#   LR curve shape:
#       ^
#  peak |     *
#       |   *   *
#       |  *      *  *  *
#   end | *               * * * * *
#       +-------------------------> steps
#         warmup   cosine decay
# ---------------------------------------------------------------------------

N_STEPS      = 5000    # Total training steps.
WARMUP_STEPS = 200     # Ramp up LR over the first 200 steps (~4% of training).
PEAK_LR      = 1e-3    # Peak learning rate.
END_LR       = 1e-4    # End learning rate — decay to 10% of peak by the final step.


# ---------------------------------------------------------------------------
# SECTION 3: TRAINING STEP
#
# One training step:
#   1. Forward pass → compute loss
#   2. Backward pass → compute gradients of loss w.r.t. every parameter
#   3. Optimizer → convert gradients to parameter updates
#   4. Apply updates → nudge every parameter to reduce the loss
# ---------------------------------------------------------------------------

batched_loss_fn = jax.vmap(cross_entropy_loss, in_axes=(None, 0))
# jax.vmap (vectorized map) transforms cross_entropy_loss so it runs on a whole
# batch at once instead of one sequence at a time.
# in_axes=(None, 0) means:
#   - params (first arg): don't batch over this — share the same params across all sequences
#   - token_ids (second arg): batch over axis 0 — each row is one sequence
# The result is a function that takes (params, batch) where batch is shape
# (BATCH_SIZE, seq_len) and returns BATCH_SIZE loss values simultaneously.
# JAX compiles this into efficient parallel GPU operations automatically.


def batch_loss(params, batch):
    """
    Computes the mean cross-entropy loss over a batch of sequences.

    params: model parameters (shared across all sequences)
    batch:  2D integer array of shape (BATCH_SIZE, seq_len+1)
    Returns: scalar mean loss
    """
    losses = batched_loss_fn(params, batch)
    # losses shape: (BATCH_SIZE,) — one loss value per sequence in the batch.

    return jnp.mean(losses)
    # Average across the batch. This is the single number we differentiate.


loss_and_grad_fn = jax.value_and_grad(batch_loss)
# Same as before, but now differentiating through the batched loss.
# Gradients are automatically averaged across the batch because we used jnp.mean().


def make_train_step(optimizer):
    """Returns a JIT-compiled train step closed over the given optimizer."""
    def train_step(params, opt_state, batch):
        loss, grads = loss_and_grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return jax.jit(train_step)
# jax.jit (Just-In-Time compilation) transforms train_step into a compiled GPU program.
# First call: JAX traces the function and compiles it to optimized GPU code (~5-10s).
# Every subsequent call: runs the compiled program directly — no Python overhead.
# This is the single biggest GPU utilization improvement available.
# The compiled program handles forward pass, backward pass, and Adam update in one shot.


# ---------------------------------------------------------------------------
# SECTION 4: TRAINING DATA
#
# We load TinyShakespeare — ~1.1M characters of Shakespeare plays.
# This is large enough that the model cannot memorize it, so it must learn
# genuine patterns: spelling, punctuation, dramatic dialogue structure, etc.
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
CHECKPOINT_EVERY = 1000

# ---------------------------------------------------------------------------
# DATASET REGISTRY
# ---------------------------------------------------------------------------

DATASETS = {
    "fineweb-edu": {
        "hf_dataset": "HuggingFaceFW/fineweb-edu",
        "hf_config":  "sample-10BT",
        "text_field": "text",
    },
    "c4": {
        "hf_dataset": "allenai/c4",
        "hf_config":  "en",
        "text_field": "text",
    },
    "openwebtext": {
        "hf_dataset": "Skylion007/openwebtext",
        "hf_config":  None,
        "text_field": "text",
    },
}

# ---------------------------------------------------------------------------
# STREAMING DATA LOADER
#
# Streams documents from HuggingFace, tokenizes with tiktoken (Rust, ~100x
# faster than pure-Python BPE), and fills a rolling token buffer in a
# background thread so the GPU never waits for data.
#
# Buffer design:
#   - Background thread: streams docs → tokenizes → puts 500k-token chunks
#     into a queue (up to PREFETCH chunks buffered ahead).
#   - Main thread: randomly samples batches from the buffer.
#     When the buffer drops below half, it tops up from the queue.
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500_000   # tokens per background chunk
PREFETCH   = 4         # chunks to buffer ahead of training

class StreamingLoader:
    def __init__(self, dataset_cfg, seq_len, batch_size):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self._q         = queue_mod.Queue(maxsize=PREFETCH)
        self._buf       = np.array([], dtype=np.int32)

        t = threading.Thread(target=self._producer, args=(dataset_cfg,), daemon=True)
        t.start()

        print("Prefilling token buffer...")
        self._refill()
        print(f"Buffer ready: {len(self._buf):,} tokens")

    def _producer(self, cfg):
        """Background thread: stream → tokenize → enqueue chunks."""
        ds = load_dataset(
            cfg["hf_dataset"],
            name=cfg["hf_config"],
            split="train",
            streaming=True,
        )
        enc = tiktoken.get_encoding("gpt2")
        chunk = []
        for ex in cycle(ds):
            chunk.extend(enc.encode(ex[cfg["text_field"]]))
            chunk.append(enc.eot_token)  # document boundary
            while len(chunk) >= CHUNK_SIZE:
                self._q.put(np.array(chunk[:CHUNK_SIZE], dtype=np.int32))
                chunk = chunk[CHUNK_SIZE:]

    def _refill(self):
        """Block until buffer holds at least CHUNK_SIZE * 2 tokens."""
        while len(self._buf) < CHUNK_SIZE * 2:
            self._buf = np.concatenate([self._buf, self._q.get()])

    def get_batch(self):
        """Return a random batch of shape (batch_size, seq_len+1)."""
        min_tokens = self.batch_size * (self.seq_len + 1) * 4
        if len(self._buf) < min_tokens:
            self._refill()
        n      = len(self._buf)
        starts = np.random.randint(0, n - self.seq_len - 1, size=(self.batch_size,))
        idx    = starts[:, None] + np.arange(self.seq_len + 1)[None, :]
        return jnp.array(self._buf[idx], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# SECTION 5: TRAINING LOOP
# ---------------------------------------------------------------------------

def train(n_steps=N_STEPS, seq_len=512, seed=0, batch_size=BATCH_SIZE, peak_lr=PEAK_LR,
          run_name=None, dataset="fineweb-edu"):
    warmup = min(WARMUP_STEPS, n_steps // 5)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup,
        decay_steps=n_steps,
        end_value=peak_lr / 10,
    )
    optimizer = optax.adam(learning_rate=schedule)
    train_step = make_train_step(optimizer)

    dataset_cfg = DATASETS[dataset]

    wandb.init(
        project="nlearn-transformer",
        name=run_name,
        config={
            "n_steps":      n_steps,
            "seq_len":      seq_len,
            "batch_size":   batch_size,
            "peak_lr":      peak_lr,
            "end_lr":       peak_lr / 10,
            "d_model":      512,
            "n_heads":      8,
            "n_layers":     4,
            "d_ff":         2048,
            "vocab_size":   VOCAB_SIZE,
            "dataset":      dataset,
            "tokenizer":    "tiktoken-gpt2",
            "warmup_steps": warmup,
        }
    )

    key = random.PRNGKey(seed)
    key, model_key = random.split(key)

    print("Initializing model...")
    params = init_model(model_key)
    opt_state = optimizer.init(params)

    # Graceful interrupt: set a flag in the signal handler, check it each step.
    _stop = {"requested": False}
    def _handle_signal(signum, frame):
        print("\nInterrupt received — will save checkpoint and exit after this step.")
        _stop["requested"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    loader = StreamingLoader(dataset_cfg, seq_len, batch_size)
    print(f"Training on {dataset}, batch_size={batch_size}, seq_len={seq_len}")
    print_attention_config()
    print(f"Running for {n_steps} steps...\n")

    import time as _time

    loss = None
    step_times = []
    for step in range(n_steps):
        t_start = _time.time()

        t0 = _time.time()
        batch = loader.get_batch()
        t_data = _time.time() - t0

        t0 = _time.time()
        params, opt_state, loss = train_step(params, opt_state, batch)
        jax.block_until_ready(loss)  # ensure GPU work is done before timing
        t_train = _time.time() - t0

        t_step = _time.time() - t_start

        step_times.append(t_step)
        wandb.log({"loss": float(loss), "step": step,
                   "step_time": t_step, "data_time": t_data, "train_time": t_train})

        if step % 100 == 0 or step < 5:
            print(f"Step {step:>4}  loss: {loss:.4f}  "
                  f"step: {t_step:.2f}s (data: {t_data:.3f}s  train: {t_train:.2f}s)")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step, run_name=run_name)

        if _stop["requested"]:
            print("Saving checkpoint before exit...")
            save_checkpoint(params, step, run_name=run_name)
            wandb.finish()
            sys.exit(0)

    # --- Save final checkpoint ---
    final_path = save_checkpoint(params, n_steps, run_name=run_name)
    avg_step = sum(step_times[1:]) / max(len(step_times) - 1, 1)  # exclude JIT step
    print(f"\nTraining complete. Final loss: {loss:.4f}")
    print(f"Avg step time: {avg_step:.2f}s (excluding first step JIT)\n")

    # --- Generate a sample ---
    enc = tiktoken.get_encoding("gpt2")
    prompt = "The history of artificial intelligence"
    print(f"Generating text from prompt '{prompt}'...")
    key, gen_key = random.split(key)
    prompt_ids = jnp.array(enc.encode(prompt))
    output_ids = generate(params, prompt_ids, n_tokens=200, key=gen_key, temperature=0.8)
    output_text = enc.decode([int(t) for t in output_ids])
    print(f"Output:\n{output_text}\n")

    wandb.log({"generated_text": wandb.Html(f"<pre>{output_text}</pre>")})

    artifact = wandb.Artifact(name="model-checkpoint", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    print("Checkpoint uploaded to W&B artifacts.")

    wandb.finish()
    return params, final_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps",    type=int,   default=N_STEPS)
    p.add_argument("--seq_len",    type=int,   default=512)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    p.add_argument("--peak_lr",    type=float, default=PEAK_LR)
    p.add_argument("--run_name",   type=str,   default=None)
    p.add_argument("--dataset",    type=str,   default="fineweb-edu",
                   choices=list(DATASETS.keys()))
    args = p.parse_args()
    train(**vars(args))
