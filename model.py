import jax                          # Core JAX library: handles autodiff, JIT compilation, random numbers
import jax.numpy as jnp              # JAX's version of NumPy — same API but runs on GPU/TPU and supports autodiff
from jax import random               # JAX's random number module (different from Python's random — must pass keys explicitly)

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# These are the knobs that define the size and shape of the model.
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 256      # Number of unique tokens the model knows about. 256 = one per ASCII byte, simple for learning.
D_MODEL     = 128      # "Dimension of the model" — every token is represented as a vector of this length.
                       # Bigger = more expressive, but slower. GPT-3 uses 12288.
N_HEADS     = 4        # Number of attention heads. D_MODEL must be divisible by N_HEADS (128 / 4 = 32 per head).
D_FF        = 512      # Hidden size of the feed-forward sublayer inside each block. Typically 4 * D_MODEL.
N_LAYERS    = 4        # How many transformer blocks to stack. More layers = deeper reasoning. GPT-3 has 96.
MAX_SEQ_LEN = 128      # Maximum number of tokens the model can process at once (its "context window").

# ---------------------------------------------------------------------------
# SECTION 1: EMBEDDINGS
#
# Before the transformer sees any text, tokens (integers) must be converted
# into vectors. There are two embedding tables:
#   - Token embeddings: "what is this token?"
#   - Positional embeddings: "where in the sequence is this token?"
# These two vectors are added together to form the input to the transformer.
# ---------------------------------------------------------------------------

def init_embeddings(key):
    """
    Creates the two embedding lookup tables and returns them as a dict.
    'key' is a JAX PRNG key used to generate random initial values.
    JAX requires you to pass random keys explicitly (no global random state).
    """
    key1, key2 = random.split(key)   # Split one key into two independent keys, one per table.
                                     # JAX keys are immutable — you must split to get new randomness.

    token_embed = random.normal(key1, (VOCAB_SIZE, D_MODEL)) * 0.02
    # random.normal draws from a Gaussian (mean=0, std=1).
    # Shape (VOCAB_SIZE, D_MODEL): one D_MODEL-dimensional vector per token.
    # We multiply by 0.02 to keep initial values small — large initial weights
    # can cause unstable training (exploding gradients).

    pos_embed = random.normal(key2, (MAX_SEQ_LEN, D_MODEL)) * 0.02
    # Same idea, but one vector per *position* (0, 1, 2, ... MAX_SEQ_LEN-1).
    # This is how the model learns that position 0 is different from position 5.

    return {'token_embed': token_embed, 'pos_embed': pos_embed}


def embed(params, token_ids):
    """
    Looks up and adds token + position embeddings for a sequence of token IDs.

    token_ids: a 1D integer array of shape (seq_len,), e.g. [72, 101, 108, ...]
    Returns:   a 2D float array of shape (seq_len, D_MODEL)
    """
    seq_len = token_ids.shape[0]   # How many tokens are in this sequence.

    tok = params['token_embed'][token_ids]
    # Index into the token embedding table using the token IDs.
    # This is a lookup: for each integer in token_ids, grab its row from the table.
    # Result shape: (seq_len, D_MODEL)

    pos = params['pos_embed'][:seq_len]
    # Slice the positional embedding table to match the sequence length.
    # [:seq_len] takes rows 0 through seq_len-1.
    # Result shape: (seq_len, D_MODEL)

    return tok + pos
    # Element-wise addition — each token vector gets its position information baked in.
    # Result shape: (seq_len, D_MODEL) — this is the input to the transformer blocks.


# ---------------------------------------------------------------------------
# SECTION 2: CAUSAL MASK + MULTI-HEAD ATTENTION
#
# Attention is the mechanism that lets every token "look at" other tokens
# and decide how much to borrow from each one.
#
# The causal mask enforces the decoder rule: token at position i can only
# attend to positions 0..i. It cannot see the future.
#
# Multi-head attention runs scaled dot-product attention N_HEADS times in
# parallel on smaller slices of the vector, then stitches results together.
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len):
    """
    Builds an upper-triangular matrix of -inf values.
    When added to attention scores, future positions become -inf,
    which softmax turns into 0 — effectively erasing them.

    Returns shape: (seq_len, seq_len)
    """
    mask = jnp.triu(                         # triu = "upper triangular": keeps values on and above the diagonal,
                                             # sets everything below the diagonal to zero.
        jnp.full((seq_len, seq_len), -jnp.inf),  # Start with a matrix entirely filled with -infinity.
        k=1                                  # k=1 means shift the diagonal up by 1, so the diagonal itself
                                             # becomes 0 (a token CAN attend to itself).
    )
    # Result looks like this for seq_len=4:
    #   [[  0, -inf, -inf, -inf],
    #    [  0,    0, -inf, -inf],
    #    [  0,    0,    0, -inf],
    #    [  0,    0,    0,    0]]
    # Row i = "what token i is allowed to see". -inf positions get zeroed out by softmax.
    return mask


def init_attention(key):
    """
    Creates the four weight matrices used in multi-head attention.

    Q, K, V projections map the input to query/key/value spaces.
    W_o projects the concatenated head outputs back to D_MODEL dims.
    """
    key_q, key_k, key_v, key_o = random.split(key, 4)
    # Split into 4 independent keys, one per weight matrix.

    scale = 0.02  # Small init scale to keep values stable at the start of training.

    return {
        'W_q': random.normal(key_q, (D_MODEL, D_MODEL)) * scale,
        # W_q shape: (D_MODEL, D_MODEL) = (128, 128)
        # Multiplying input x @ W_q projects each token vector into "query space".
        # Query = "what am I looking for?"

        'W_k': random.normal(key_k, (D_MODEL, D_MODEL)) * scale,
        # W_k projects input into "key space".
        # Key = "what do I contain / advertise?"

        'W_v': random.normal(key_v, (D_MODEL, D_MODEL)) * scale,
        # W_v projects input into "value space".
        # Value = "what do I actually send if someone attends to me?"

        'W_o': random.normal(key_o, (D_MODEL, D_MODEL)) * scale,
        # W_o is the output projection applied after all heads are concatenated.
        # It mixes information across heads and projects back to D_MODEL dims.
    }


def attention_forward(params, x, mask):
    """
    Runs multi-head self-attention.

    x:    input of shape (seq_len, D_MODEL)
    mask: causal mask of shape (seq_len, seq_len)
    Returns output of shape (seq_len, D_MODEL)
    """
    seq_len, _ = x.shape              # Unpack sequence length; _ = D_MODEL (we already know it).

    d_head = D_MODEL // N_HEADS       # Each head works on a slice of size d_head = 128 // 4 = 32.

    # --- Step 1: Project input into Q, K, V ---

    Q = x @ params['W_q']             # (seq_len, D_MODEL) @ (D_MODEL, D_MODEL) → (seq_len, D_MODEL)
    K = x @ params['W_k']             # Same shape. Every token now has a query, key, and value vector.
    V = x @ params['W_v']             # Same shape.

    # --- Step 2: Split into heads ---
    # Reshape from (seq_len, D_MODEL) to (seq_len, N_HEADS, d_head),
    # then transpose to (N_HEADS, seq_len, d_head) so each head is a separate batch.

    Q = Q.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)
    # .reshape splits the D_MODEL dimension into N_HEADS groups of d_head each.
    # .transpose(1,0,2) reorders axes: (seq_len, N_HEADS, d_head) → (N_HEADS, seq_len, d_head)
    # Now Q[0] is head 0's queries for all tokens, Q[1] is head 1's, etc.

    K = K.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)  # Same reshaping for keys.
    V = V.reshape(seq_len, N_HEADS, d_head).transpose(1, 0, 2)  # Same reshaping for values.

    # --- Step 3: Scaled dot-product attention (the core formula) ---

    scores = jnp.matmul(Q, K.transpose(0, 2, 1))
    # Q shape:             (N_HEADS, seq_len, d_head)
    # K.transpose(0,2,1):  (N_HEADS, d_head, seq_len)  ← swap last two axes to make K^T
    # Result scores shape: (N_HEADS, seq_len, seq_len)
    # scores[h, i, j] = dot product of token i's query with token j's key, for head h.
    # High score = token i wants to attend to token j.

    scores = scores / jnp.sqrt(d_head)
    # Divide by sqrt(d_head) = sqrt(32) ≈ 5.66.
    # Without this, dot products grow large as d_head increases, pushing softmax into
    # regions with near-zero gradients (the "vanishing gradient" problem). This keeps
    # the scores in a well-behaved range. This is the "scaled" in "scaled dot-product attention".

    scores = scores + mask
    # Add the causal mask. Positions where mask = -inf become -inf in scores.
    # mask shape (seq_len, seq_len) broadcasts across the N_HEADS dimension automatically.

    weights = jax.nn.softmax(scores, axis=-1)
    # Softmax over the last axis (the "which token to attend to" axis).
    # Turns raw scores into probabilities that sum to 1 across each row.
    # -inf entries become exactly 0 — future tokens are completely ignored.
    # weights[h, i, j] = "how much does token i attend to token j, in head h?"

    # --- Step 4: Weighted sum of values ---

    out = jnp.matmul(weights, V)
    # weights: (N_HEADS, seq_len, seq_len)
    # V:       (N_HEADS, seq_len, d_head)
    # out:     (N_HEADS, seq_len, d_head)
    # For each token i, this computes a weighted average of all value vectors,
    # where the weights come from the attention distribution we just computed.

    # --- Step 5: Concatenate heads and project ---

    out = out.transpose(1, 0, 2).reshape(seq_len, D_MODEL)
    # .transpose(1,0,2): (N_HEADS, seq_len, d_head) → (seq_len, N_HEADS, d_head)
    # .reshape: merge the last two dims back together → (seq_len, D_MODEL)
    # This is the "concatenate heads" step — we're just undoing the split from Step 2.

    out = out @ params['W_o']
    # Final linear projection: (seq_len, D_MODEL) @ (D_MODEL, D_MODEL) → (seq_len, D_MODEL)
    # This mixes information across heads and allows the model to learn which
    # combinations of head outputs are useful.

    return out  # Shape: (seq_len, D_MODEL) — same shape as the input x.


# ---------------------------------------------------------------------------
# SECTION 3: LAYER NORM + FEED-FORWARD NETWORK
#
# Each transformer block has two sublayers:
#   1. Multi-head attention (Section 2)
#   2. Feed-forward network (this section)
#
# Both sublayers are wrapped with:
#   - Layer normalization (applied BEFORE the sublayer — "pre-norm" style)
#   - A residual connection (input is added back AFTER the sublayer)
#
# Layer norm keeps activations stable during training.
# Residual connections let gradients flow back through many layers without vanishing.
# ---------------------------------------------------------------------------

def init_layer_norm():
    """
    Layer norm has two learnable parameters per feature dimension:
      gamma (scale) — initialized to 1, learned multiplier
      beta  (shift) — initialized to 0, learned offset
    No randomness needed here — these are deterministic starting points.
    """
    return {
        'gamma': jnp.ones(D_MODEL),   # Shape: (D_MODEL,). Ones = no scaling at init.
        'beta':  jnp.zeros(D_MODEL),  # Shape: (D_MODEL,). Zeros = no shift at init.
    }


def layer_norm(params, x):
    """
    Normalizes each token's vector to have mean=0 and std=1,
    then applies a learned scale (gamma) and shift (beta).

    x: shape (seq_len, D_MODEL)
    Returns same shape.
    """
    eps = 1e-5
    # A tiny constant added to the denominator to prevent division by zero
    # in the rare case where variance is exactly 0.

    mean = jnp.mean(x, axis=-1, keepdims=True)
    # Compute the mean across the D_MODEL dimension for each token independently.
    # axis=-1 = last axis = the feature dimension.
    # keepdims=True keeps the shape as (seq_len, 1) instead of (seq_len,)
    # so that broadcasting works correctly in the subtraction below.

    var = jnp.var(x, axis=-1, keepdims=True)
    # Compute the variance across D_MODEL for each token.
    # Shape: (seq_len, 1)

    x_norm = (x - mean) / jnp.sqrt(var + eps)
    # Subtract mean and divide by standard deviation.
    # Now each token vector has mean=0 and std=1 across its D_MODEL features.
    # Shape: (seq_len, D_MODEL)

    return params['gamma'] * x_norm + params['beta']
    # Apply learned scale and shift.
    # gamma and beta are shape (D_MODEL,) — they broadcast across the seq_len dimension.
    # At init this is a no-op (1 * x_norm + 0), but during training
    # the model learns the right scale and shift for each feature.


def init_ffn(key):
    """
    The feed-forward network is two linear layers with a GeLU activation in between.
    It operates on each token independently (no communication between tokens here).

    Architecture: D_MODEL → D_FF → D_MODEL
                    128   →  512  →   128
    The expansion to D_FF (4× wider) gives the model capacity to compute
    complex non-linear transformations on each token's representation.
    """
    key1, key2 = random.split(key)  # Two keys for two weight matrices.

    return {
        'W1': random.normal(key1, (D_MODEL, D_FF)) * 0.02,
        # First linear layer expands from D_MODEL to D_FF.
        # Shape: (128, 512)

        'b1': jnp.zeros(D_FF),
        # Bias for the first layer. Shape: (512,). Initialized to zero.

        'W2': random.normal(key2, (D_FF, D_MODEL)) * 0.02,
        # Second linear layer compresses back from D_FF to D_MODEL.
        # Shape: (512, 128)

        'b2': jnp.zeros(D_MODEL),
        # Bias for the second layer. Shape: (128,). Initialized to zero.
    }


def ffn_forward(params, x):
    """
    Runs the feed-forward network on each token independently.

    x: shape (seq_len, D_MODEL)
    Returns same shape.
    """
    x = x @ params['W1'] + params['b1']
    # Linear projection: (seq_len, 128) @ (128, 512) + (512,) → (seq_len, 512)
    # Each token is now represented in a 512-dimensional space.

    x = jax.nn.gelu(x)
    # GeLU (Gaussian Error Linear Unit) activation function.
    # Similar to ReLU (zeros out negatives) but smooth — it doesn't have a hard cutoff at 0.
    # Used by GPT-2, GPT-3, and most modern LLMs.
    # Without a non-linearity here, two linear layers would collapse into one — no extra power.

    x = x @ params['W2'] + params['b2']
    # Project back down: (seq_len, 512) @ (512, 128) + (128,) → (seq_len, D_MODEL)
    # Each token is back to its original D_MODEL size, but transformed.

    return x  # Shape: (seq_len, D_MODEL)


# ---------------------------------------------------------------------------
# SECTION 4: TRANSFORMER BLOCK + FULL MODEL
#
# A transformer block combines everything from Sections 2 and 3 into one
# repeatable unit. The full model stacks N_LAYERS of these blocks, then
# projects the output to vocabulary logits for next-token prediction.
#
# Data flow through one block:
#
#   x ──► LayerNorm ──► Attention ──► + ──► LayerNorm ──► FFN ──► + ──►
#   │                                 ▲                            ▲
#   └─────────────────────────────────┘                            │
#   └────────────────────────────────────────────────────────────-─┘
#
# The two "+" symbols are residual connections — the original input is
# always added back to the sublayer's output.
# ---------------------------------------------------------------------------

def init_block(key):
    """
    Initializes all parameters for one transformer block:
      - attention weights (W_q, W_k, W_v, W_o)
      - two layer norms (one before attention, one before FFN)
      - feed-forward weights (W1, b1, W2, b2)
    """
    key_attn, key_ffn = random.split(key)  # One key for attention, one for FFN.

    return {
        'ln1':  init_layer_norm(),       # Layer norm applied before attention.
        'attn': init_attention(key_attn),# Multi-head attention weights.
        'ln2':  init_layer_norm(),       # Layer norm applied before FFN.
        'ffn':  init_ffn(key_ffn),       # Feed-forward network weights.
    }


def block_forward(params, x, mask):
    """
    Runs one full transformer block on input x.

    x:    shape (seq_len, D_MODEL)
    mask: causal mask shape (seq_len, seq_len)
    Returns same shape as x.
    """

    # --- Sublayer 1: Multi-head self-attention ---

    x = x + attention_forward(params['attn'], layer_norm(params['ln1'], x), mask)
    # Breaking this down right to left:
    #   1. layer_norm(params['ln1'], x)          — normalize x before attention
    #   2. attention_forward(..., normalized_x)  — run attention on the normalized input
    #   3. x + (result)                          — residual connection: add original x back
    #
    # The residual connection is critical. Without it, in a 4-layer model gradients
    # would have to flow back through 4 chained multiplications and would shrink to
    # near-zero (vanishing gradients). The shortcut gives gradients a direct path back.

    # --- Sublayer 2: Feed-forward network ---

    x = x + ffn_forward(params['ffn'], layer_norm(params['ln2'], x))
    # Same pattern:
    #   1. layer_norm(params['ln2'], x)  — normalize (x now includes the attention update)
    #   2. ffn_forward(..., normalized)  — run FFN on normalized input
    #   3. x + (result)                 — residual connection again

    return x  # Shape: (seq_len, D_MODEL)


# ---------------------------------------------------------------------------
# FULL MODEL
# ---------------------------------------------------------------------------

def init_model(key):
    """
    Initializes every parameter in the full decoder-only transformer.
    Returns a single nested dict containing all parameters.
    """
    # Split into enough keys: 1 for embeddings, N_LAYERS for blocks, 1 for lm_head.
    keys = random.split(key, N_LAYERS + 2)

    return {
        'embeddings': init_embeddings(keys[0]),
        # Token + positional embedding tables. Shape: (VOCAB_SIZE, D_MODEL) and (MAX_SEQ_LEN, D_MODEL).

        'blocks': [init_block(keys[i + 1]) for i in range(N_LAYERS)],
        # A Python list of N_LAYERS block parameter dicts.
        # Each block is independent with its own weights — they don't share parameters.
        # List comprehension: for i in 0,1,2,3 → init_block(keys[1]), init_block(keys[2]), ...

        'ln_final': init_layer_norm(),
        # One last layer norm applied after all blocks, before the output projection.
        # GPT-2 introduced this — it stabilizes the final representations.

        'lm_head': random.normal(keys[-1], (D_MODEL, VOCAB_SIZE)) * 0.02,
        # "Language model head" — projects each token's D_MODEL vector to VOCAB_SIZE logits.
        # Shape: (128, 256). One logit per vocabulary token.
        # The logit with the highest value = the model's best guess for the next token.
    }


def model_forward(params, token_ids):
    """
    Full forward pass: token IDs in, next-token logits out.

    token_ids: 1D integer array of shape (seq_len,)
    Returns:   2D float array of shape (seq_len, VOCAB_SIZE)
               logits[i] = scores for what token comes after position i
    """

    # --- Step 1: Embed tokens ---
    x = embed(params['embeddings'], token_ids)
    # Convert integer token IDs to vectors and add positional information.
    # Shape: (seq_len, D_MODEL)

    # --- Step 2: Build causal mask ---
    seq_len = token_ids.shape[0]
    mask = make_causal_mask(seq_len)
    # Shape: (seq_len, seq_len). Built once and reused by every block.

    # --- Step 3: Pass through all transformer blocks ---
    for block_params in params['blocks']:
        x = block_forward(block_params, x, mask)
    # Each block refines the representation of every token.
    # After N_LAYERS blocks, each token vector contains a rich contextual
    # summary informed by everything that came before it.

    # --- Step 4: Final layer norm ---
    x = layer_norm(params['ln_final'], x)
    # Normalize one last time before projecting to vocabulary.
    # Shape unchanged: (seq_len, D_MODEL)

    # --- Step 5: Project to vocabulary logits ---
    logits = x @ params['lm_head']
    # (seq_len, D_MODEL) @ (D_MODEL, VOCAB_SIZE) → (seq_len, VOCAB_SIZE)
    # For each position in the sequence, we now have one score per vocabulary token.
    # These are raw unnormalized scores (logits) — not probabilities yet.
    # To get probabilities: jax.nn.softmax(logits, axis=-1)

    return logits


# ---------------------------------------------------------------------------
# SECTION 5: TEXT GENERATION
#
# The model outputs logits — raw scores over the vocabulary for each position.
# To generate text, we:
#   1. Take the logits at the last position (the next-token prediction)
#   2. Convert to probabilities via softmax
#   3. Sample a token from that distribution
#   4. Append it to the sequence and repeat
#
# The "temperature" parameter controls randomness:
#   - temperature = 1.0  → sample from the raw distribution (default)
#   - temperature < 1.0  → sharper distribution, more predictable/repetitive
#   - temperature > 1.0  → flatter distribution, more random/creative
# ---------------------------------------------------------------------------

def generate(params, prompt_ids, n_tokens, key, temperature=1.0):
    """
    Autoregressively generates n_tokens new tokens given a prompt.

    params:      full model parameter dict from init_model()
    prompt_ids:  1D integer array of starting token IDs
    n_tokens:    how many new tokens to generate
    key:         JAX PRNG key for sampling
    temperature: float controlling randomness (default 1.0)

    Returns: 1D integer array of shape (len(prompt_ids) + n_tokens,)
    """

    tokens = prompt_ids
    # Start with the prompt. We'll append to this array one token at a time.

    for _ in range(n_tokens):
        # --- Step 1: Run forward pass on current sequence ---
        logits = model_forward(params, tokens)
        # Shape: (seq_len, VOCAB_SIZE)
        # logits[i] = scores for the token that follows position i.

        next_logits = logits[-1]
        # Take only the last row — the prediction for what comes after the final token.
        # Shape: (VOCAB_SIZE,)

        # --- Step 2: Apply temperature ---
        next_logits = next_logits / temperature
        # Dividing by temperature < 1 makes the scores more spread out (sharper peaks).
        # Dividing by temperature > 1 compresses the scores closer together (flatter).
        # At temperature = 1.0 this is a no-op.

        # --- Step 3: Convert logits to probabilities ---
        probs = jax.nn.softmax(next_logits)
        # Softmax: exp(logit_i) / sum(exp(logit_j) for all j)
        # Turns raw scores into a probability distribution that sums to 1.
        # Shape: (VOCAB_SIZE,)

        # --- Step 4: Sample one token from the distribution ---
        key, subkey = random.split(key)
        # Split the key before each use — JAX requires a fresh key per random call.

        next_token = random.categorical(subkey, next_logits)
        # random.categorical draws one sample from a categorical distribution.
        # It takes logits directly (applies softmax internally), returns an integer index.
        # That index is the sampled token ID.
        # Shape: scalar integer.

        # --- Step 5: Append sampled token to sequence ---
        tokens = jnp.append(tokens, next_token)
        # jnp.append concatenates the new token onto the end of the sequence.
        # Next iteration, the model sees one more token and predicts the one after that.

    return tokens
    # Shape: (len(prompt_ids) + n_tokens,) — the full sequence including the original prompt.
