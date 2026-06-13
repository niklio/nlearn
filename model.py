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
