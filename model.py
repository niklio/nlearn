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
