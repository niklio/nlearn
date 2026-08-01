"""Run inference from an nlearn checkpoint.

Examples:
    python -m nlearn.generate --hf-run deep_hero --prompt "HAMLET:" --n-tokens 200
    python -m nlearn.generate --hf-run deep_hero --checkpoint 16500 --prompt "Hello" --n-tokens 50
    python -m nlearn.generate --local-checkpoint checkpoints/deep_hero/step_016500.pkl \
        --prompt "Hello" --n-tokens 50
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import tiktoken

from nlearn.checkpoint_store import download_inference_checkpoint
from nlearn.model import generate as model_generate


def load_params(checkpoint_path):
    """Load parameters from either a weights-only or full resume checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    if (
        isinstance(checkpoint, dict)
        and "params" in checkpoint
        and "opt_state" in checkpoint
        and "step" in checkpoint
    ):
        return checkpoint["params"]
    return checkpoint


def run_generation(params, prompt, n_tokens, temperature):
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = jnp.array(enc.encode(prompt))
    print(f'Prompt: "{prompt}" ({len(prompt_ids)} tokens)')
    print(f"Generating {n_tokens} tokens at temperature {temperature}...\n")

    key = jax.random.PRNGKey(0)
    output_ids = model_generate(
        params, prompt_ids, n_tokens=n_tokens, key=key, temperature=temperature
    )
    return enc.decode([int(token) for token in output_ids])


def main():
    parser = argparse.ArgumentParser(description="Generate text from a local or HF checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hf-run",
        help="Checkpoint run name in the HF bucket, for example 'deep_hero'.",
    )
    source.add_argument(
        "--local-checkpoint",
        help="Path to a local weights-only or resume .pkl checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="For --hf-run: latest, resume, or a step number (default: latest).",
    )
    parser.add_argument(
        "--hf-bucket",
        default=None,
        metavar="OWNER/BUCKET",
        help="HF Storage Bucket (defaults to NLEARN_HF_BUCKET).",
    )
    parser.add_argument("--prompt", required=True, help="Input text to generate from.")
    parser.add_argument(
        "--n-tokens", required=True, type=int, help="Number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8). Range: 0.1-2.0.",
    )
    args = parser.parse_args()

    if args.hf_run:
        checkpoint_path = download_inference_checkpoint(
            args.hf_run, selector=args.checkpoint, bucket=args.hf_bucket
        )
    else:
        checkpoint_path = args.local_checkpoint

    params = load_params(checkpoint_path)
    output = run_generation(params, args.prompt, args.n_tokens, args.temperature)
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
