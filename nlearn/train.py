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

from nlearn.model import init_model, model_forward, model_forward_features, generate, output_projection, VOCAB_SIZE, D_MODEL, COMPUTE_DTYPE
from nlearn.attention import print_attention_config
from nlearn.logging_utils import StepTimer, TrainingLogger, benchmark_peak_tflops
from nlearn.kernels.cross_entropy import cross_entropy as _fused_ce
from nlearn.kernels.cross_entropy import linear_cross_entropy as _fused_linear_ce

# NLEARN_FUSED_LMHEAD=1 fuses lm_head + CE (chunked over vocab) so the (bs·seq, vocab)
# logits are NEVER materialised — the long-context memory enabler (peak ~472MB vs ~1.6GB
# at seq512). It recomputes logits in the backward, ~32% more CE-region compute, so it's
# OFF by default (the materialised lm_head + fused CE keeps MFU higher at short seq).
_FUSED_LMHEAD = os.environ.get("NLEARN_FUSED_LMHEAD") == "1"

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


def _resume_path(run_name):
    subdir = os.path.join(CHECKPOINT_DIR, run_name) if run_name else CHECKPOINT_DIR
    return os.path.join(subdir, "resume.pkl")


def save_resume_state(params, opt_state, step, run_name=None):
    """Full training state (params + optimizer state + step) for auto-resume after a
    GPU/Metal-HAL hang. Written atomically (tmp + rename) so a crash mid-write can't
    corrupt the resume file. The LR schedule position lives in opt_state, so resuming
    continues the schedule correctly; the data stream just continues with fresh data."""
    path = _resume_path(run_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"params": jax.device_get(params),
             "opt_state": jax.device_get(opt_state),
             "step": int(step)}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def load_resume_state(run_name=None):
    """Return (params, opt_state, next_step) if a resume file exists, else None."""
    path = _resume_path(run_name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state["params"], state["opt_state"], state["step"]

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
    W = output_projection(params)  # (D_MODEL, VOCAB_SIZE), tied to the token embedding
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
        raw_logits   = x @ W_chunks[i]                     # (seq_len, VOCAB_CHUNK_SIZE)
        chunk_logits = jnp.where(_VOCAB_MASK[i], raw_logits, -jnp.inf)

        chunk_max = chunk_logits.max(axis=-1)
        new_max   = jnp.maximum(running_max, chunk_max)

        running_sum_exp = (
            running_sum_exp * jnp.exp(running_max - new_max)
            + jnp.exp(chunk_logits - new_max[:, None]).sum(axis=-1)
        )
        running_max = new_max

        in_chunk      = (target_ids >= chunk_start) & (target_ids < chunk_start + VOCAB_CHUNK_SIZE)
        safe_idx      = jnp.clip(target_ids - chunk_start, 0, VOCAB_CHUNK_SIZE - 1)
        # Select the target token's logit via one-hot multiply rather than
        # advanced indexing (chunk_logits[arange, safe_idx]). The gather's
        # backward is a scatter, which IREE's metal-spirv path miscompiles under
        # vmap (malformed tensor.expand_shape). One-hot select has a pure
        # broadcast/reduce backward that lowers cleanly, at the cost of an extra
        # (seq_len, VOCAB_CHUNK_SIZE) tensor we already materialise anyway.
        onehot        = jax.nn.one_hot(safe_idx, VOCAB_CHUNK_SIZE, dtype=raw_logits.dtype)
        chunk_correct = jnp.sum(raw_logits * onehot, axis=-1)  # raw (not -inf-masked) avoids 0*-inf=NaN
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
WARMUP_STEPS = int(os.environ.get("NLEARN_WARMUP", "200"))  # LR-ramp steps (effective updates)
PEAK_LR      = 3e-4    # Peak learning rate. (1e-3 diverges in the fp16/IREE-Metal
                       # setup: a 10k-step B=4 run peaked at val 7.5 by step 200 then
                       # slowly climbed to 8.5 — gradual divergence, not NaN. 3e-4
                       # converges cleanly past warmup.)
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

def _simple_cross_entropy_loss(params, token_ids):
    """Unchunked full-vocab cross-entropy. Used on the IREE-Metal backend, whose
    metal-spirv compiler miscompiles the 13-chunk loop in cross_entropy_loss (a
    fusion/codegen bug — every individual op is correct, but the composed loop
    yields a wrong, eventually-negative loss). Materialises the (seq, VOCAB)
    logit matrix, so it costs more memory than the chunked path. One-hot select
    (not gather) avoids a scatter-under-vmap miscompile on the same backend."""
    input_ids  = token_ids[:-1]
    target_ids = token_ids[1:]
    logits = model_forward(params, input_ids)               # (seq, VOCAB_SIZE)
    log_Z  = jax.scipy.special.logsumexp(logits, axis=-1)
    onehot = jax.nn.one_hot(target_ids, VOCAB_SIZE, dtype=logits.dtype)
    correct_logit = jnp.sum(logits * onehot, axis=-1)
    return -(correct_logit - log_Z).mean()


import nlearn.attention as _attn


def _simple_ce_batched(params, batch):
    """Batched full-vocab cross-entropy via the fused Metal CE kernel (ce_iree):
    one threadgroup per (bs·seq) row streams the vocab once to produce loss + dlogits,
    instead of materialising a (bs·seq, vocab) one-hot + ~6 memory-bound passes. ~8.6×
    faster than the one-hot CE here and the long-context memory enabler (no (M,vocab)
    one-hot). `cross_entropy` flattens the leading dims internally; off IREE-Metal it
    falls back to the jnp one-hot CE."""
    input_ids  = batch[:, :-1]                              # (bs, seq)
    target_ids = batch[:, 1:]                               # (bs, seq)
    if _FUSED_LMHEAD:
        # Fuse lm_head + CE, chunked over vocab — never materialise (bs·seq, vocab).
        x = model_forward_features(params, input_ids)       # (bs, seq, D_MODEL)
        return _fused_linear_ce(x.reshape(-1, D_MODEL), output_projection(params),
                                target_ids.reshape(-1))
    logits = model_forward(params, input_ids)               # (bs, seq, VOCAB_SIZE)
    return _fused_ce(logits, target_ids)


if _attn.USE_IREE_FLASH or os.environ.get("NLEARN_FORCE_BATCHED_CE") == "1":
    # IREE-Metal (or forced, e.g. CPU f32 verification): batched execution (one
    # dispatch per op), full-vocab simple CE. The vmap-per-sequence path below is
    # stale after the batched refactor, so CPU runs force this path too.
    def batch_loss(params, batch):
        return _simple_ce_batched(params, batch)
else:
    # CUDA/CPU: keep the per-sequence chunked CE under vmap (memory-efficient there).
    _batched_loss_fn = jax.vmap(cross_entropy_loss, in_axes=(None, 0))

    def batch_loss(params, batch):
        return jnp.mean(_batched_loss_fn(params, batch))


loss_and_grad_fn = jax.value_and_grad(batch_loss)
# Same as before, but now differentiating through the batched loss.
# Gradients are automatically averaged across the batch because we used jnp.mean().

eval_batch_loss = jax.jit(batch_loss)
# JIT-compiled loss without gradients — used for validation evaluation.


# fp16 loss scaling. EMPIRICAL FINDING: this model trains stably in fp16 WITHOUT
# scaling (no gradient underflow observed — loss falls cleanly to ~7.8), while a
# STATIC scale overflows fp16 in the backward once gradients grow post-warmup and
# produces NaN (caught by a 250-step validation run). Proper dynamic loss scaling
# (back off on overflow) needs lax.cond, which metal-spirv miscompiles. So default
# OFF; set NLEARN_LOSS_SCALE>1 manually only if underflow ever appears.
LOSS_SCALE = float(os.environ.get("NLEARN_LOSS_SCALE", "1.0"))


# Gradient accumulation via optax.MultiSteps: NLEARN_GRAD_ACCUM micro-batches per
# effective optimizer update. Each train_step processes ONE micro-batch (so peak
# memory stays at micro-batch size — bs16 is the single-pass ceiling; bs24+ OOMs),
# and MultiSteps accumulates across calls, applying the real Adam update every Kth.
# This reaches an effective batch (e.g. 32 = 16×2) where lr=1e-3 is stable, without
# the OOM of materialising K micro-graphs at once. K=1 ⇒ plain single-batch training.
GRAD_ACCUM = int(os.environ.get("NLEARN_GRAD_ACCUM", "1"))


def make_train_step(optimizer):
    """Returns a JIT-compiled train step closed over the given optimizer (which may be
    an optax.MultiSteps wrapper that accumulates K micro-batch grads before applying)."""
    def _scaled_loss(p, b):
        return batch_loss(p, b) * LOSS_SCALE

    def train_step(params, opt_state, batch):
        scaled_loss, grads = jax.value_and_grad(_scaled_loss)(params, batch)
        if LOSS_SCALE != 1.0:
            grads = jax.tree_util.tree_map(lambda g: g / LOSS_SCALE, grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, scaled_loss / LOSS_SCALE
    return jax.jit(train_step)
# jax.jit (Just-In-Time compilation) transforms train_step into a compiled GPU program.
# First call: JAX traces the function and compiles it to optimized GPU code (~5-10s).
# Every subsequent call: runs the compiled program directly — no Python overhead.
# This is the single biggest GPU utilization improvement available.
# The compiled program handles forward pass, backward pass, and Adam update in one shot.


# ---------------------------------------------------------------------------
# SECTION 4: TRAINING DATA
#
# Sequential data consumption with validation split.
# Each token is seen exactly once (single-epoch streaming).
# A small held-out validation set is reserved at startup.
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
CHECKPOINT_EVERY = int(os.environ.get("NLEARN_CHECKPOINT_EVERY", "1000"))

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
# faster than pure-Python BPE), and feeds tokens sequentially so the model
# sees fresh data every step (no repetition within an epoch).
#
# Design:
#   - Background thread: streams docs → tokenizes → puts 500k-token chunks
#     into a queue (up to PREFETCH chunks buffered ahead).
#   - Main thread: consumes tokens sequentially from the buffer.
#     Each get_batch() advances a cursor and returns the next contiguous
#     block of batch_size × (seq_len+1) tokens, reshaped into a batch.
#     When the cursor nears the end, old tokens are discarded and fresh
#     chunks are pulled from the queue.
#
# Validation:
#   - A small held-out set is reserved at init (VAL_TOKENS tokens).
#   - val_loss() evaluates on fixed validation batches without gradients.
# ---------------------------------------------------------------------------

CHUNK_SIZE  = 500_000   # tokens per background chunk
PREFETCH    = 4         # chunks to buffer ahead of training
VAL_TOKENS  = 100_000   # tokens reserved for validation (~200 pages of text)
VAL_EVERY   = 100       # evaluate validation loss every N steps

class StreamingLoader:
    def __init__(self, dataset_cfg, seq_len, batch_size):
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self._stride    = batch_size * (seq_len + 1)
        self._cursor    = 0
        self._q         = queue_mod.Queue(maxsize=PREFETCH)
        self._buf       = np.array([], dtype=np.int32)
        self._val_data  = None

        t = threading.Thread(target=self._producer, args=(dataset_cfg,), daemon=True)
        t.start()

        # Reserve validation data from the first tokens in the stream.
        print("Reserving validation data...")
        while len(self._buf) < VAL_TOKENS + CHUNK_SIZE * 2:
            self._buf = np.concatenate([self._buf, self._q.get()])
        self._val_data = self._buf[:VAL_TOKENS].copy()
        self._buf = self._buf[VAL_TOKENS:]  # training data starts after val
        self._cursor = 0
        print(f"Validation: {len(self._val_data):,} tokens")
        print(f"Train buffer: {len(self._buf):,} tokens")

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

    def _ensure_tokens(self, n):
        """Ensure at least n tokens are available ahead of cursor."""
        while len(self._buf) - self._cursor < n:
            try:
                new_chunk = self._q.get(timeout=30)
            except queue_mod.Empty:
                break
            self._buf = np.concatenate([self._buf, new_chunk])
        # Compact: discard consumed tokens to avoid unbounded memory growth.
        if self._cursor > CHUNK_SIZE:
            self._buf = self._buf[self._cursor:]
            self._cursor = 0

    def get_batch(self):
        """Return the next sequential batch of shape (batch_size, seq_len+1)."""
        self._ensure_tokens(self._stride)
        end = self._cursor + self._stride
        flat = self._buf[self._cursor:end]
        self._cursor = end
        return jnp.array(flat.reshape(self.batch_size, self.seq_len + 1), dtype=jnp.int32)

    def val_batches(self):
        """Yield all non-overlapping validation batches."""
        n = len(self._val_data) // (self.seq_len + 1) // self.batch_size
        for i in range(n):
            start = i * self._stride
            flat  = self._val_data[start:start + self._stride]
            yield jnp.array(flat.reshape(self.batch_size, self.seq_len + 1), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# SECTION 5: TRAINING LOOP
# ---------------------------------------------------------------------------

def train(steps=N_STEPS, seq_len=512, seed=0, batch_size=BATCH_SIZE, peak_lr=PEAK_LR,
          run_name=None, dataset="fineweb-edu", resume=False):
    # With gradient accumulation (GRAD_ACCUM micro-steps per update), the LR schedule
    # advances per EFFECTIVE update, so build it over steps//GRAD_ACCUM and run the
    # loop for `steps` micro-steps. (GRAD_ACCUM=1 ⇒ unchanged.)
    eff_updates = max(1, steps // GRAD_ACCUM)
    warmup = min(WARMUP_STEPS, eff_updates // 5)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup,
        decay_steps=eff_updates,
        end_value=peak_lr / 10,
    )
    # AdamW (decoupled weight decay) — the transformer-standard optimizer. WITHOUT
    # weight decay, lr=1e-3 slowly diverged here (val bottomed ~7.53 near peak LR then
    # climbed monotonically as weights grew unbounded) even with warmup 200, while the
    # jax-metal reference sustained lr=1e-3 to val 5.71 — the missing regulariser, not
    # gradient noise, was the difference. wd=0.1 is the usual GPT-ish value. Clipping is
    # off by default: clip_by_global_norm's tree-norm emits vector.create_mask which
    # fails to legalize on metal-spirv (set NLEARN_GRAD_CLIP>0 only off-Metal).
    GRAD_CLIP = float(os.environ.get("NLEARN_GRAD_CLIP", "0"))
    WEIGHT_DECAY = float(os.environ.get("NLEARN_WD", "0.1"))
    base = optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    optimizer = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), base) if GRAD_CLIP > 0 else base
    if GRAD_ACCUM > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=GRAD_ACCUM)
        print(f"Gradient accumulation: {GRAD_ACCUM} micro-batches/update "
              f"(effective batch {batch_size * GRAD_ACCUM}, {eff_updates} updates)")
    train_step = make_train_step(optimizer)

    dataset_cfg = DATASETS[dataset]

    wandb.init(
        project="nlearn-transformer",
        name=run_name,
        id=run_name if run_name else None,   # fixed id so resumed segments share one curve
        resume="allow",
        config={
            "steps":      steps,
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
    # jit the init: eager random.normal emits a Sharding custom_call that the
    # IREE-Metal backend can't legalize; running it inside jit avoids that.
    params = jax.jit(init_model)(model_key)
    opt_state = optimizer.init(params)

    # Auto-resume after a GPU hang: if a resume file exists for this run, restore
    # params + optimizer state (incl. LR-schedule position) and continue from there.
    start_step = 0
    _resumed = load_resume_state(run_name) if resume else None
    if _resumed is not None:
        params, opt_state, start_step = _resumed
        params = jax.device_put(params)
        opt_state = jax.device_put(opt_state)
        print(f"Resumed from step {start_step} (resume.pkl)")

    # Graceful interrupt: set a flag in the signal handler, check it each step.
    _stop = {"requested": False}
    def _handle_signal(signum, frame):
        print("\nInterrupt received — will save checkpoint and exit after this step.")
        _stop["requested"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    loader = StreamingLoader(dataset_cfg, seq_len, batch_size)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Training on {dataset}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Model parameters: {n_params:,}")
    print_attention_config()
    print(f"Running for {steps} steps...\n")

    # Benchmark in the model's compute dtype (bf16 isn't supported on the
    # IREE metal-spirv target). Non-fatal: it's only used for MFU logging.
    try:
        hw_peak_tflops = benchmark_peak_tflops(dtype=COMPUTE_DTYPE)
        print(f"Hardware peak: {hw_peak_tflops:.1f} TFLOPS ({COMPUTE_DTYPE.__name__})")
    except Exception as e:
        hw_peak_tflops = 0.0
        print(f"Hardware peak benchmark skipped ({type(e).__name__}).")
    logger = TrainingLogger(
        eval_batch_loss, loader.val_batches, val_every=VAL_EVERY,
        n_params=n_params, seq_len=seq_len, batch_size=batch_size,
        hw_peak_tflops=hw_peak_tflops,
        run_name=run_name, dataset=dataset,
    )
    timer = StepTimer()

    loss = None
    for step in range(start_step, steps):
        timer.start()
        batch = loader.get_batch()
        timer.mark_data()

        params, opt_state, loss = train_step(params, opt_state, batch)
        jax.block_until_ready(loss)
        timer.mark_train()

        logger.log_step(step, loss, timer, params, steps)

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step, run_name=run_name)
            # Full state for auto-resume across hangs (params + opt_state + next step).
            save_resume_state(params, opt_state, step + 1, run_name=run_name)

        if _stop["requested"]:
            print("Saving checkpoint before exit...")
            save_checkpoint(params, step, run_name=run_name)
            logger.finalize(loss, step, steps)
            wandb.finish()
            sys.exit(0)

    # --- Save final checkpoint ---
    final_path = save_checkpoint(params, steps, run_name=run_name)
    logger.print_summary(loss)
    logger.finalize(loss, steps - 1, steps)

    artifact = wandb.Artifact(name="model-checkpoint", type="model")
    artifact.add_file(final_path)
    wandb.log_artifact(artifact)
    print("Checkpoint uploaded to W&B artifacts.")

    wandb.finish()
    return params, final_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int,   default=N_STEPS)
    p.add_argument("--seq-len",    type=int,   default=512)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--peak-lr",    type=float, default=PEAK_LR)
    p.add_argument("--run-name",   type=str,   default=None)
    p.add_argument("--dataset",    type=str,   default="fineweb-edu",
                   choices=list(DATASETS.keys()))
    p.add_argument("--resume", action="store_true",
                   help="resume from this run's checkpoints/<run>/resume.pkl if present")
    args = p.parse_args()
    train(**vars(args))
