import sys
import signal
import jax                          # Core JAX: autodiff, JIT
import jax.numpy as jnp              # JAX NumPy for array operations
from jax import random               # Explicit random key management
import optax                         # JAX optimizer library (Adam, SGD, etc.)
import wandb                         # Weights & Biases experiment tracking
import pickle                        # Python's built-in serialization — used for checkpoints
import os                            # File path utilities

from model import init_model, model_forward, generate, VOCAB_SIZE
from tokenizer import load_tokenizer, encode as bpe_encode, decode as bpe_decode

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

def save_checkpoint(params, step):
    """Save model parameters to disk at a given training step."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Create the checkpoints/ directory if it doesn't exist yet.

    path = os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pkl")
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

def cross_entropy_loss(params, token_ids):
    """
    Computes the average cross-entropy loss over a sequence of tokens.

    params:    full model parameter dict
    token_ids: 1D integer array of shape (seq_len,)
    Returns:   scalar loss value (lower = better predictions)
    """

    input_ids  = token_ids[:-1]
    # Drop the last token — we don't need a prediction for what comes after it.
    # Shape: (seq_len - 1,)

    target_ids = token_ids[1:]
    # Drop the first token — targets are the "correct answers" shifted left by one.
    # Shape: (seq_len - 1,)
    # target_ids[i] is the token the model should predict after seeing input_ids[i].

    logits = model_forward(params, input_ids)
    # Run the model on the inputs. Shape: (seq_len - 1, VOCAB_SIZE)
    # logits[i] = raw scores for all vocab tokens at position i.

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # log_softmax = log(softmax(x)) — equivalent but numerically more stable than
    # computing softmax then taking log separately.
    # Shape: (seq_len - 1, VOCAB_SIZE)
    # log_probs[i, j] = log probability the model assigns to token j at position i.

    correct_log_probs = log_probs[jnp.arange(len(target_ids)), target_ids]
    # Index into log_probs to get only the log probability of the *correct* next token.
    # jnp.arange(len(target_ids)) = [0, 1, 2, ...] — one index per position.
    # target_ids = [e, l, l, o, !] — the correct token ID at each position.
    # This fancy indexing selects log_probs[0, target_ids[0]], log_probs[1, target_ids[1]], ...
    # Shape: (seq_len - 1,)

    loss = -jnp.mean(correct_log_probs)
    # Cross-entropy loss = negative mean log probability of correct predictions.
    # If the model is certain and right: log_prob ≈ 0  → loss ≈ 0 (good)
    # If the model is uncertain or wrong: log_prob << 0 → loss is large (bad)
    # We negate because we want to *minimize* loss, and log probs are negative.

    return loss


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
CHECKPOINT_EVERY = 500

import numpy as np

# ---------------------------------------------------------------------------
# DATA LOADING
#
# Supports two modes, selected by whether datasets/fineweb.bin exists:
#
#   Binary mode (FineWeb-Edu):
#     numpy.memmap reads the pre-tokenized binary file directly from disk.
#     The OS pages in only the chunks actually accessed — the full file
#     never loads into RAM. This scales to arbitrarily large datasets.
#
#   Text mode (Shakespeare fallback):
#     Loads shakespeare.txt, encodes with BPE tokenizer. Used for quick
#     local experiments when fineweb.bin hasn't been prepared yet.
# ---------------------------------------------------------------------------

FINEWEB_BIN = "datasets/fineweb-edu.bin"

print("Loading BPE tokenizer...")
_vocab, _merges, _char_to_id = load_tokenizer('tokenizer.json')

def encode(text):
    """Encode text to BPE token IDs as a JAX array."""
    return jnp.array(bpe_encode(text, _char_to_id, _merges))

def decode(token_ids):
    """Decode BPE token IDs back to text."""
    return bpe_decode([int(t) for t in token_ids], _vocab)

def load_data():
    """
    Load training data. Uses fineweb.bin if available, otherwise Shakespeare.
    Returns a 1D integer array of token IDs.
    """
    if os.path.exists(FINEWEB_BIN):
        data = np.memmap(FINEWEB_BIN, dtype=np.uint16, mode='r')
        # np.memmap maps the file into virtual memory — reads happen on demand
        # from disk without loading everything into RAM.
        # dtype=uint16 matches what data.py wrote (2 bytes per token).
        data = jnp.array(data, dtype=jnp.int32)
        # Convert to int32 for JAX indexing compatibility.
        print(f"Loaded FineWeb-Edu: {len(data):,} tokens from {FINEWEB_BIN}")
        return data
    else:
        print(f"{FINEWEB_BIN} not found — falling back to Shakespeare. Run: python3 data.py --dataset fineweb-edu --target-tokens 500_000_000")
        with open('datasets/shakespeare.txt', 'r') as f:
            text = f.read()
        data = encode(text)
        print(f"Loaded Shakespeare: {len(data):,} tokens")
        return data


# ---------------------------------------------------------------------------
# SECTION 5: TRAINING LOOP
# ---------------------------------------------------------------------------

def train(n_steps=N_STEPS, seq_len=512, seed=0, batch_size=BATCH_SIZE, peak_lr=PEAK_LR, run_name=None):
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

    wandb.init(
        project="nlearn-transformer",
        name=run_name,
        config={
            "n_steps":      n_steps,
            "seq_len":      seq_len,
            "batch_size":   batch_size,
            "peak_lr":      peak_lr,
            "end_lr":       peak_lr / 10,
            "d_model":      128,
            "n_heads":      4,
            "n_layers":     4,
            "d_ff":         512,
            "vocab_size":   VOCAB_SIZE,
            "dataset":      "fineweb-edu" if os.path.exists(FINEWEB_BIN) else "tinyshakespeare",
            "tokenizer":    "bpe-4000",
            "warmup_steps": WARMUP_STEPS,
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

    data = load_data()
    n_tokens = len(data)
    print(f"Training on {n_tokens} tokens, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Running for {n_steps} steps...\n")

    loss = None
    for step in range(n_steps):
        key, subkey = random.split(key)
        starts = random.randint(subkey, (batch_size,), 0, n_tokens - seq_len - 1)
        offsets = jnp.arange(seq_len + 1)
        batch = data[starts[:, None] + offsets[None, :]]

        params, opt_state, loss = train_step(params, opt_state, batch)

        wandb.log({"loss": float(loss), "step": step})

        if step % 100 == 0:
            print(f"Step {step:>4}  loss: {loss:.4f}")

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step)

        if _stop["requested"]:
            print("Saving checkpoint before exit...")
            save_checkpoint(params, step)
            wandb.finish()
            sys.exit(0)

    # --- Save final checkpoint ---
    final_path = save_checkpoint(params, n_steps)
    print(f"\nTraining complete. Final loss: {loss:.4f}\n")

    # --- Generate text from a Shakespeare-style prompt ---
    print("Generating text from prompt 'ROMEO:'...")
    key, gen_key = random.split(key)
    prompt_ids = encode("ROMEO:")
    output_ids = generate(params, prompt_ids, n_tokens=200, key=gen_key, temperature=0.8)
    output_text = decode(output_ids)
    print(f"Output:\n{output_text}\n")

    wandb.log({"generated_text": wandb.Html(f"<pre>{output_text}</pre>")})

    # --- Save final checkpoint as a versioned W&B Artifact ---
    artifact = wandb.Artifact(name="model-checkpoint", type="model")
    # Artifacts are versioned file storage in W&B — each run produces a new version.
    # type="model" is a convention that tells W&B this is a model file.

    artifact.add_file(final_path)
    # Attach the checkpoint .pkl file to this artifact.

    wandb.log_artifact(artifact)
    # Upload to W&B. Visible in the Artifacts tab of your run.
    # You can download it later, mark it as "best", or link it to a dataset version.
    print(f"Checkpoint uploaded to W&B artifacts.")

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
    args = p.parse_args()
    train(**vars(args))
