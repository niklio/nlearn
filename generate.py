"""
generate.py — Run inference from a W&B checkpoint.

Usage:
    python3 generate.py --run-id <wandb_run_id> --prompt "Wherefore art thou" --n-tokens 100
    python3 generate.py --run-id <wandb_run_id> --prompt "HAMLET:" --n-tokens 200 --temperature 0.9

Arguments:
    --run-id      W&B run ID (the short string like "7tyjzwiz", visible in the run URL)
    --prompt      Input text to continue from
    --n-tokens    Number of new tokens to generate
    --temperature Sampling temperature (default 0.8). Higher = more random, lower = more predictable.
    --project     W&B project name (default: nlearn-transformer)
    --entity      W&B entity/username (default: read from your W&B login)
"""

import argparse
import os
import pickle
import tempfile

import jax
import jax.numpy as jnp
import wandb

from model import generate as model_generate
from tokenizer import load_tokenizer, encode as bpe_encode, decode as bpe_decode


def download_checkpoint(run_id, project, entity):
    """
    Download the model checkpoint artifact from a W&B run.

    run_id:  short run ID string (e.g. "7tyjzwiz")
    project: W&B project name
    entity:  W&B username or team name
    Returns: local path to the downloaded .pkl file
    """
    api = wandb.Api()
    # wandb.Api() creates a client that talks to the W&B REST API.
    # Credentials come from your ~/.netrc file set up during wandb login.

    run_path = f"{entity}/{project}/{run_id}"
    print(f"Fetching run: {run_path}")

    run = api.run(run_path)
    # Fetch metadata about the run — its config, history, and linked artifacts.

    # Find the model checkpoint artifact logged by this run.
    artifacts = run.logged_artifacts()
    # Returns a list of all artifacts this run produced.

    model_artifact = None
    for artifact in artifacts:
        if artifact.type == "model":
            model_artifact = artifact
            break
    # We logged our artifact with type="model", so we look for that.

    if model_artifact is None:
        raise ValueError(
            f"No model artifact found for run {run_id}.\n"
            f"Make sure the run completed successfully and uploaded a checkpoint.\n"
            f"Runs before the artifact logging was added (avid-cloud-3 and earlier) "
            f"don't have artifacts — use a local checkpoint file instead with --local-checkpoint."
        )

    print(f"Found artifact: {model_artifact.name} (version {model_artifact.version})")

    download_dir = tempfile.mkdtemp()
    # Create a temporary directory to download the artifact into.
    # tempfile.mkdtemp() creates a unique directory in /tmp.

    artifact_dir = model_artifact.download(root=download_dir)
    # Download the artifact files to download_dir.
    # Returns the path to the directory containing the files.

    # Find the .pkl file inside the downloaded directory.
    pkl_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise ValueError(f"No .pkl checkpoint file found in artifact at {artifact_dir}")

    return os.path.join(artifact_dir, pkl_files[0])


def load_params(checkpoint_path):
    """Load model parameters from a .pkl checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)
    # pickle restores the nested dict of numpy arrays.
    # JAX will move them to the GPU automatically when used in computations.


def run_generation(params, prompt, n_tokens, temperature, tokenizer_path):
    """
    Encode the prompt, run generation, decode and return the output text.
    """
    vocab, merges, char_to_id = load_tokenizer(tokenizer_path)

    def encode(text):
        return jnp.array(bpe_encode(text, char_to_id, merges))

    def decode(ids):
        return bpe_decode([int(t) for t in ids], vocab)

    prompt_ids = encode(prompt)
    print(f"Prompt: \"{prompt}\" ({len(prompt_ids)} tokens)")
    print(f"Generating {n_tokens} tokens at temperature {temperature}...\n")

    key = jax.random.PRNGKey(0)
    output_ids = model_generate(params, prompt_ids, n_tokens=n_tokens, key=key, temperature=temperature)
    # model_generate is the generate() function from model.py.
    # Returns the full sequence including the prompt.

    return decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a W&B model checkpoint.")

    parser.add_argument("--run-id",     required=False, default=None,
                        help="W&B run ID (e.g. '7tyjzwiz'). Find it in your run's URL.")
    parser.add_argument("--local-checkpoint", required=False, default=None,
                        help="Path to a local .pkl checkpoint file. Use instead of --run-id.")
    parser.add_argument("--prompt",     required=True,
                        help="Input text to generate from.")
    parser.add_argument("--n-tokens",   required=True, type=int,
                        help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). Range: 0.1–2.0.")
    parser.add_argument("--project",    default="nlearn-transformer",
                        help="W&B project name (default: nlearn-transformer).")
    parser.add_argument("--entity",     default=None,
                        help="W&B entity/username. Defaults to your logged-in W&B user.")
    parser.add_argument("--tokenizer",  default="tokenizer.json",
                        help="Path to tokenizer.json (default: tokenizer.json).")

    args = parser.parse_args()

    # Validate that exactly one of --run-id or --local-checkpoint is provided.
    if args.run_id is None and args.local_checkpoint is None:
        parser.error("Provide either --run-id or --local-checkpoint.")
    if args.run_id and args.local_checkpoint:
        parser.error("Provide only one of --run-id or --local-checkpoint, not both.")

    # Resolve entity from W&B login if not provided.
    if args.run_id and args.entity is None:
        api = wandb.Api()
        args.entity = api.default_entity
        # api.default_entity reads the username from your W&B credentials.
        print(f"Using W&B entity: {args.entity}")

    # Get the checkpoint.
    if args.run_id:
        checkpoint_path = download_checkpoint(args.run_id, args.project, args.entity)
    else:
        checkpoint_path = args.local_checkpoint

    # Load params and generate.
    params = load_params(checkpoint_path)
    output = run_generation(params, args.prompt, args.n_tokens, args.temperature, args.tokenizer)

    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
