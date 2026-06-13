import jax                          # Core JAX: autodiff, JIT
import jax.numpy as jnp              # JAX NumPy for array operations
from jax import random               # Explicit random key management
import optax                         # JAX optimizer library (Adam, SGD, etc.)

from model import init_model, model_forward, generate, VOCAB_SIZE

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
# SECTION 2: OPTIMIZER (ADAM)
#
# Adam (Adaptive Moment Estimation) is the standard optimizer for transformers.
# It's an improvement over plain gradient descent — instead of:
#   param = param - learning_rate * gradient
#
# Adam tracks a running average of past gradients (momentum) and a running
# average of squared gradients (adaptive scaling), giving each parameter
# its own effective learning rate that adapts during training.
#
# learning_rate controls the step size. Too high = unstable. Too low = slow.
# 1e-3 is a common default for Adam.
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-3   # How big each parameter update step is.

optimizer = optax.adam(LEARNING_RATE)
# optax.adam returns an optimizer object with two methods:
#   .init(params)              → creates initial optimizer state (momentum buffers etc.)
#   .update(grads, opt_state)  → computes parameter updates from gradients + state


# ---------------------------------------------------------------------------
# SECTION 3: TRAINING STEP
#
# One training step:
#   1. Forward pass → compute loss
#   2. Backward pass → compute gradients of loss w.r.t. every parameter
#   3. Optimizer → convert gradients to parameter updates
#   4. Apply updates → nudge every parameter to reduce the loss
# ---------------------------------------------------------------------------

loss_and_grad_fn = jax.value_and_grad(cross_entropy_loss)
# jax.value_and_grad wraps any function f(params, ...) and returns a new function
# that computes BOTH f(params, ...) AND df/d(params) in one call.
# This is JAX's autodiff — it automatically differentiates through the entire model.


def train_step(params, opt_state, token_ids):
    """
    Performs one gradient update on a single sequence.

    params:    current model parameters
    opt_state: current optimizer state (Adam momentum buffers)
    token_ids: 1D integer array — one training example

    Returns: (updated_params, updated_opt_state, loss_value)
    """

    loss, grads = loss_and_grad_fn(params, token_ids)
    # loss:  scalar — the cross-entropy loss for this batch
    # grads: a dict with the exact same structure as params, but containing
    #        the gradient of the loss w.r.t. each parameter instead of the parameter itself.
    #        grads['lm_head'][i,j] = "how much does the loss change if lm_head[i,j] increases?"

    updates, opt_state = optimizer.update(grads, opt_state)
    # Adam processes the raw gradients and computes "updates" — the actual amounts
    # to add to each parameter. These are scaled and smoothed versions of the gradients.
    # opt_state is updated in place (momentum buffers, step count, etc.)

    params = optax.apply_updates(params, updates)
    # Adds updates to every parameter: param = param + update
    # (updates are already negated by Adam so this reduces the loss)

    return params, opt_state, loss


# ---------------------------------------------------------------------------
# SECTION 4: TRAINING DATA
#
# We need text to train on. For a learning example we use a short repeated
# string so the model has a realistic chance of memorizing it with few steps.
# In a real system you'd load gigabytes of text from disk in shuffled batches.
# ---------------------------------------------------------------------------

TRAINING_TEXT = (
    "hello world. the cat sat on the mat. "
    "the dog ran fast. a quick brown fox. "
    "hello world. the cat sat on the mat. "
) * 4
# Repeat the text to give the model more exposure to the same patterns.
# Real LLM training uses trillions of tokens — we're using ~hundreds.

def encode(text):
    """Converts a string to a JAX array of ASCII byte values (our token IDs)."""
    return jnp.array([ord(c) for c in text])

def decode(token_ids):
    """Converts an array of token IDs back to a string."""
    return ''.join(
        chr(int(t)) if 32 <= int(t) < 127 else '?'  # Only print printable ASCII.
        for t in token_ids
    )


# ---------------------------------------------------------------------------
# SECTION 5: TRAINING LOOP
# ---------------------------------------------------------------------------

def train(n_steps=200, seq_len=32, seed=0):
    """
    Trains the model for n_steps gradient updates and prints progress.

    n_steps:  total number of parameter updates to perform
    seq_len:  length of each training sequence (chunk of the training text)
    seed:     random seed for reproducibility
    """

    key = random.PRNGKey(seed)         # Initialize the PRNG key.
    key, model_key = random.split(key) # Split off a key for model initialization.

    print("Initializing model...")
    params = init_model(model_key)     # Random initial parameters.

    opt_state = optimizer.init(params)
    # Initialize Adam's state: sets up zero-filled momentum buffers for every parameter.
    # opt_state mirrors the structure of params but holds optimizer internals, not weights.

    data = encode(TRAINING_TEXT)       # Encode all training text to token IDs once.
    n_tokens = len(data)
    print(f"Training on {n_tokens} tokens for {n_steps} steps...\n")

    for step in range(n_steps):

        # --- Pick a random chunk of the training data ---
        key, subkey = random.split(key)  # Fresh key for each step.

        start = int(random.randint(subkey, (), 0, n_tokens - seq_len - 1))
        # Pick a random starting position in the data.
        # random.randint(key, shape, min, max) — shape=() means scalar output.

        chunk = data[start : start + seq_len + 1]
        # Grab seq_len + 1 tokens. The +1 is because cross_entropy_loss will
        # use the first seq_len as input and the last seq_len as targets.

        # --- Gradient update ---
        params, opt_state, loss = train_step(params, opt_state, chunk)

        # --- Log progress ---
        if step % 20 == 0:
            print(f"Step {step:>4}  loss: {loss:.4f}")

    print("\nTraining complete.")
    print(f"Final loss: {loss:.4f}")
    print("(A well-trained model on this tiny dataset should reach loss < 1.0)\n")

    # --- Generate some text to see if the model learned anything ---
    print("Generating text from prompt 'hello'...")
    key, gen_key = random.split(key)
    prompt_ids = encode("hello")
    output_ids = generate(params, prompt_ids, n_tokens=40, key=gen_key, temperature=0.8)
    print(f"Output: \"{decode(output_ids)}\"\n")

    return params


if __name__ == "__main__":
    train()
