import os
from functools import partial

import jax                          # Core JAX library: handles autodiff, JIT compilation, random numbers
import jax.numpy as jnp              # JAX's version of NumPy — same API but runs on GPU/TPU and supports autodiff
from jax import random               # JAX's random number module (different from Python's random — must pass keys explicitly)

from nlearn.attention import attention, PLATFORM  # Cross-platform attention dispatch (see attention.py)
from nlearn.kernels.gemm import matmul as gemm_matmul  # custom Metal GEMM on IREE-Metal, else jnp.matmul
from nlearn.kernels.gemm import linear                 # batched x[...,K]@W[K,N] as one flattened GEMM

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# These are the knobs that define the size and shape of the model.
# ---------------------------------------------------------------------------

VOCAB_SIZE  = 50257    # GPT-2 vocabulary size (tiktoken "gpt2" encoding).
D_MODEL     = 768      # "Dimension of the model" — every token is a vector of this length.
                       # Scaled 512 → 768 (params ∝ D_MODEL²) toward the ~123M target for loss ~3.2.
N_HEADS     = 12       # Attention heads. D_MODEL must be divisible by N_HEADS (768 / 12 = 64 per head,
                       # matching the flash kernel's d_head). dh stays 64 across the scale-up.
D_FF        = 2048     # SwiGLU hidden size. ~(8/3)·D_MODEL is the param-matched point
                       # vs a 4× GeLU MLP; 2048 ≈ (8/3)·768 for the scaled config.
N_LAYERS    = 12       # Transformer blocks. Scaled 4 → 12: depth is the cheapest real capacity
                       # (the old 4-layer stack was the model's most undersized axis).
MAX_SEQ_LEN = 512      # Maximum context window in tokens. With 3.8x BPE compression, this covers ~2000 characters.

# Depth-scaled init: residual-output projections (attention W_o, FFN W_down) are
# scaled by 1/sqrt(2·N_LAYERS) so the residual stream variance stays ~constant with
# depth — lets deep stacks train stably (GPT-2/Llama-style).
RESIDUAL_SCALE = (2 * N_LAYERS) ** -0.5

# Forward-pass dtype (stored params stay float32). fp16 on Metal because IREE's
# metal-spirv target for Apple GPUs advertises fp16 but NOT bf16 (bf16 matmuls
# fail to legalize); bf16 elsewhere (CUDA/CPU), whose wider exponent range makes
# mixed-precision training more stable.
# NLEARN_COMPUTE_DTYPE overrides this ("float32"/"float16"/"bfloat16") — used to test
# whether the fp16-gradient noise from the custom GEMM caps the loss (run f32 + the
# native matmul fallback via NLEARN_DISABLE_GEMM=1).
_DT = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}
# Default f32 on Metal: activations stay f32 (IREE handles f32 elementwise; bf16
# activations hit a vector.bitcast legalization gap) while the custom GEMM takes the
# matmul inputs to bf16 — wide range everywhere. Pure fp16 CAPS the loss (~7.4) because
# its narrow exponent range overflows as activations grow (verified). bf16 elsewhere.
COMPUTE_DTYPE = _DT.get(
    os.environ.get("NLEARN_COMPUTE_DTYPE", ""),
    jnp.float32 if "metal" in PLATFORM else jnp.bfloat16,
)

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
    token_embed = random.normal(key, (VOCAB_SIZE, D_MODEL)) * 0.02
    # random.normal draws from a Gaussian (mean=0, std=1).
    # Shape (VOCAB_SIZE, D_MODEL): one D_MODEL-dimensional vector per token.
    # We multiply by 0.02 to keep initial values small — large initial weights
    # can cause unstable training (exploding gradients).
    # No positional table: position is handled by RoPE (rotary embeddings) on Q/K.
    return {'token_embed': token_embed}


def embed(params, token_ids):
    """
    Looks up and adds token + position embeddings for a sequence of token IDs.

    token_ids: a 1D integer array of shape (seq_len,), e.g. [72, 101, 108, ...]
    Returns:   a 2D float array of shape (seq_len, D_MODEL)
    """
    # Token embeddings only. Position is injected later by RoPE (rotary embeddings)
    # applied to Q/K inside attention — no learned position table, so the model is no
    # longer capped at MAX_SEQ_LEN and extrapolates to longer contexts.
    return params['token_embed'][token_ids]
    # For (bs, seq) ids this yields (bs, seq, D_MODEL); for (seq,) it yields (seq, D_MODEL).


# ---------------------------------------------------------------------------
# SECTION 2: MULTI-HEAD ATTENTION
#
# Attention lets every token "look at" other tokens and borrow from them.
# Causal masking enforces the decoder rule: token i can only attend to
# positions 0..i (no peeking at future tokens).
#
# The actual attention computation (scores, mask, softmax, weighted sum)
# is delegated to attention.py which dispatches to the best implementation
# for the current platform:
#   - CUDA: cuDNN flash attention (fused kernel, no seq² memory)
#   - Metal/CPU: standard matmul (single fused XLA kernel)
# ---------------------------------------------------------------------------


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

        'W_o': random.normal(key_o, (D_MODEL, D_MODEL)) * (scale * RESIDUAL_SCALE),
        # W_o is the output projection applied after all heads are concatenated.
        # It mixes information across heads and projects back to D_MODEL dims.
        # Residual-output projection → depth-scaled init (×1/sqrt(2·N_LAYERS)).
    }


def _qk_norm(x, eps=1e-6):
    """Parameterless RMSNorm over the head dim (last axis) — used for QK-norm on
    Q and K. x: (bs·heads, seq, d_head). Returns same shape, unit-RMS per head vector."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def apply_rope(x, positions, base=10000.0):
    """Rotary position embedding (RoPE) on Q or K. Rotates the two halves of each
    head vector by position-dependent angles, so Q·K depends on RELATIVE position
    (m−n) — giving position information with no learned table, and extrapolating to
    any sequence length. x: (bs·heads, seq, d_head); positions: (seq,). Applied to
    Q and K before attention; V and the flash kernel are untouched."""
    d = x.shape[-1]
    half = d // 2
    inv_freq = base ** (-jnp.arange(0, half, dtype=jnp.float32) / float(half))  # (half,)
    ang = positions[:, None].astype(jnp.float32) * inv_freq[None, :]            # (seq, half)
    cos = jnp.cos(ang).astype(x.dtype)
    sin = jnp.sin(ang).astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]                                      # (bh, seq, half)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention_forward(params, x):
    """
    Standard multi-head self-attention.
    For seq_len ≤ ~1024 this is fastest on Metal — one fused matmul that XLA
    can compile as a single kernel. Switch to chunked (ATTN_CHUNK_SIZE) if
    you increase seq_len to the point where the seq×seq matrix causes OOM.

    x: input of shape (bs, seq_len, D_MODEL)
    Returns: (bs, seq_len, D_MODEL)

    Batched execution: project with one flattened GEMM each, fold the batch into the
    head axis (bs·heads, seq, d_head), and run attention as a SINGLE flash dispatch
    over bs·heads "heads" — no vmap, no per-sequence dispatches.
    """
    bs, seq_len, _ = x.shape
    d_head = D_MODEL // N_HEADS
    bh = bs * N_HEADS

    def proj(W):
        # (bs, seq, d) -> (bs, seq, heads, dh) -> (bs, heads, seq, dh) -> (bs·heads, seq, dh)
        return (linear(x, W).reshape(bs, seq_len, N_HEADS, d_head)
                .transpose(0, 2, 1, 3).reshape(bh, seq_len, d_head))

    Q, K, V = proj(params['W_q']), proj(params['W_k']), proj(params['W_v'])

    # QK-norm: RMS-normalize Q and K over the head dim (parameterless) before RoPE.
    # Bounds ||q||,||k|| so attention logits can't blow up — stabilizes deep training
    # and tolerates a higher learning rate. (V is untouched.)
    Q = _qk_norm(Q)
    K = _qk_norm(K)

    # RoPE: rotate Q and K by position (no learned table; extrapolates to any seq_len).
    positions = jnp.arange(seq_len)
    Q = apply_rope(Q, positions)
    K = apply_rope(K, positions)

    out = attention(Q, K, V)                                # (bs·heads, seq, d_head)
    out = (out.reshape(bs, N_HEADS, seq_len, d_head)
              .transpose(0, 2, 1, 3).reshape(bs, seq_len, D_MODEL))
    return linear(out, params['W_o'])


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
    RMSNorm has ONE learnable parameter per feature dimension:
      gamma (scale) — initialized to 1, learned multiplier
    (No mean-centering and no beta shift — RMSNorm drops both vs LayerNorm; it's
    cheaper, fewer params, and trains as well or better at depth — Llama/T5 style.)
    """
    return {
        'gamma': jnp.ones(D_MODEL),   # Shape: (D_MODEL,). Ones = no scaling at init.
    }


def layer_norm(params, x):
    """
    RMSNorm: scales each token's vector to unit root-mean-square (no mean-centering),
    then applies a learned per-feature scale (gamma).

        x_norm = x / sqrt(mean(x^2) + eps);   return gamma * x_norm

    x: shape (..., D_MODEL)   Returns same shape.
    (Kept the name `layer_norm` so call sites are unchanged.)
    """
    eps = 1e-5
    ms = jnp.mean(x * x, axis=-1, keepdims=True)         # mean square over features
    x_norm = x * jax.lax.rsqrt(ms + eps)                  # unit-RMS, no centering
    return params['gamma'] * x_norm


def init_ffn(key):
    """
    The feed-forward network is two linear layers with a GeLU activation in between.
    It operates on each token independently (no communication between tokens here).

    Architecture: D_MODEL → D_FF → D_MODEL
                    128   →  512  →   128
    The expansion to D_FF (4× wider) gives the model capacity to compute
    complex non-linear transformations on each token's representation.
    """
    key_g, key_u, key_d = random.split(key, 3)  # gate / up / down

    return {
        # SwiGLU: out = (silu(x·W_gate) ⊙ (x·W_up)) · W_down. Two parallel input
        # projections (gate + up) and a down projection — no biases (SwiGLU drops them).
        'W_gate': random.normal(key_g, (D_MODEL, D_FF)) * 0.02,
        'W_up':   random.normal(key_u, (D_MODEL, D_FF)) * 0.02,
        # W_down is a residual-output projection → depth-scaled init for stable deep training.
        'W_down': random.normal(key_d, (D_FF, D_MODEL)) * (0.02 * RESIDUAL_SCALE),
    }


def ffn_forward(params, x):
    """
    SwiGLU feed-forward, per token: gated GLU with a SiLU (swish) gate.
        h = silu(x · W_gate) ⊙ (x · W_up)        # gate modulates the up-projection
        return h · W_down
    The gating consistently lowers loss vs a plain GeLU MLP (Llama/PaLM).

    x: shape (..., D_MODEL)   Returns same shape.
    """
    gate = jax.nn.silu(linear(x, params['W_gate']))   # silu(x) = x·sigmoid(x)
    up   = linear(x, params['W_up'])
    return linear(gate * up, params['W_down'])


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


def block_forward(params, x):
    """
    Runs one full transformer block on input x.

    x: shape (seq_len, D_MODEL)
    Returns same shape as x.
    """
    x = x + attention_forward(params['attn'], layer_norm(params['ln1'], x))
    x = x + ffn_forward(params['ffn'], layer_norm(params['ln2'], x))
    return x


# ---------------------------------------------------------------------------
# FULL MODEL
# ---------------------------------------------------------------------------

def init_model(key):
    """
    Initializes every parameter in the full decoder-only transformer.
    Returns a single nested dict containing all parameters.
    """
    # Split into enough keys: 1 for embeddings, N_LAYERS for blocks. (No lm_head key —
    # the output projection is TIED to the token embedding, see output_projection().)
    keys = random.split(key, N_LAYERS + 1)

    return {
        'embeddings': init_embeddings(keys[0]),
        # Token embedding table, shape (VOCAB_SIZE, D_MODEL). Doubles as the output
        # projection (tied weights) — no separate lm_head, saving VOCAB_SIZE·D_MODEL params
        # (~25M at D_MODEL=512, the bulk of a small model).

        'blocks': [init_block(keys[i + 1]) for i in range(N_LAYERS)],
        # A Python list of N_LAYERS block parameter dicts.
        # Each block is independent with its own weights — they don't share parameters.

        'ln_final': init_layer_norm(),
        # One last RMSNorm applied after all blocks, before the (tied) output projection.
    }


def output_projection(params):
    """The output (unembedding) matrix, TIED to the token embedding: (D_MODEL, VOCAB_SIZE).
    `linear(x, output_projection(params))` gives the vocab logits."""
    return params['embeddings']['token_embed'].T


def model_forward_features(params, token_ids):
    """
    Forward pass through all transformer blocks, returning hidden states
    before the final (tied) output projection.

    token_ids: 1D integer array of shape (seq_len,)
    Returns:   2D float array of shape (seq_len, D_MODEL)
    """
    # Cast params to COMPUTE_DTYPE (bfloat16) for all matmuls; stored params stay float32.
    # JAX autodiff handles the astype gradient correctly: bf16 grads get upcast to f32
    # when they reach the stored float32 params.
    bf16_params = jax.tree_util.tree_map(
        lambda p: p.astype(COMPUTE_DTYPE) if jnp.issubdtype(p.dtype, jnp.floating) else p,
        params,
    )
    x = embed(bf16_params['embeddings'], token_ids)
    # NLEARN_NO_CHECKPOINT=1 disables remat (diagnostic: does remat-wrapping the
    # flash custom_vjp trigger the seq>=256 hang on metal-spirv?).
    import os as _os
    _block = block_forward if _os.environ.get("NLEARN_NO_CHECKPOINT") == "1" \
        else jax.checkpoint(block_forward, prevent_cse=False)
    for block_params in bf16_params['blocks']:
        x = _block(block_params, x)
    # Cast back to float32 for the final layer norm and loss computation.
    return layer_norm(params['ln_final'], x.astype(jnp.float32))


def model_forward(params, token_ids):
    """
    Full forward pass: token IDs in, next-token logits out.

    token_ids: 1D integer array of shape (seq_len,)
    Returns:   2D float array of shape (seq_len, VOCAB_SIZE)
               logits[i] = scores for what token comes after position i
    """
    x = model_forward_features(params, token_ids)
    # Tied output projection: (bs, seq, D_MODEL) @ (D_MODEL, VOCAB) as one flattened GEMM.
    return linear(x, output_projection(params))


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

@partial(jax.jit, static_argnames=("n_tokens",))
def _generate_jit(params, prompt_ids, n_tokens, key, temperature):
    """JIT core of generate: fixed-size token buffer + lax.scan.

    Keeping the whole loop inside jit (a) avoids the per-step eager random ops
    that emit a Sharding custom_call the IREE-Metal backend can't legalize, and
    (b) is faster everywhere. Each step runs the model on the full buffer; causal
    masking means the logit we read depends only on already-filled positions, so
    the trailing zeros don't affect the result.
    """
    plen = prompt_ids.shape[0]                       # static within the trace
    total = plen + n_tokens
    # Build via concatenate + write via where-mask rather than `.at[].set()`:
    # scatter/dynamic-update miscompiles on the IREE metal-spirv backend.
    buf = jnp.concatenate([prompt_ids, jnp.zeros((n_tokens,), jnp.int32)])
    positions = jnp.arange(total)

    def step(carry, i):
        buf, key = carry
        logits = model_forward(params, buf[None])[0] # add/drop batch dim: (total, VOCAB)
        next_logits = logits[plen + i - 1] / temperature   # gather read (ok on metal)
        key, subkey = random.split(key)
        tok = random.categorical(subkey, next_logits).astype(jnp.int32)
        buf = jnp.where(positions == (plen + i), tok, buf)  # no scatter
        return (buf, key), None

    (buf, _), _ = jax.lax.scan(step, (buf, key), jnp.arange(n_tokens))
    return buf


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
    prompt_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
    # The buffer holds prompt + generated; positions index the (MAX_SEQ_LEN)
    # positional table, so cap total length, trimming the oldest prompt tokens.
    if prompt_ids.shape[0] + n_tokens > MAX_SEQ_LEN:
        prompt_ids = prompt_ids[-(MAX_SEQ_LEN - n_tokens):]
    return _generate_jit(params, prompt_ids, n_tokens, key,
                         jnp.asarray(temperature, jnp.float32))
