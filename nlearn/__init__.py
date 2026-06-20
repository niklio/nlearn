"""nlearn — a from-scratch GPT-style transformer trained on Apple Silicon
through the open-source IREE-Metal stack with hand-authored Metal kernels.

Subpackages:
  nlearn.kernels  — IREE-Metal kernel bindings (GEMM, fused cross-entropy)
  nlearn.data     — corpus streaming + BPE tokenizer

Entry points (run from the repo root):
  python -m nlearn.train      — training loop
  python -m nlearn.generate   — sampling/generation
"""
