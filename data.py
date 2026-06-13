"""
data.py — Prepare FineWeb-Edu training data.

Streams FineWeb-Edu from HuggingFace, (optionally) retrains the BPE tokenizer
on a sample, then tokenizes the full target corpus and writes it to a binary
file that train.py can memory-map directly — no RAM required at training time.

Usage:
    # Full pipeline: retrain tokenizer + write 500M tokens to disk
    python3 data.py --target-tokens 500_000_000

    # Skip tokenizer retraining (use existing tokenizer.json)
    python3 data.py --target-tokens 500_000_000 --no-retrain-tokenizer

    # Quick test with 10M tokens
    python3 data.py --target-tokens 10_000_000

Output:
    datasets/fineweb.bin      — flat uint16 array of token IDs
    tokenizer.json            — updated BPE tokenizer (unless --no-retrain-tokenizer)
"""

import argparse
import os
import numpy as np
from datasets import load_dataset

from tokenizer import train as train_bpe, encode as bpe_encode, load_tokenizer

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASET_NAME   = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
# FineWeb-Edu comes in several sizes on HuggingFace:
#   "sample-10BT"  — 10 billion tokens of educational web text (~50GB raw)
#   "sample-100BT" — 100 billion tokens
#   "default"      — full dataset (~1.3TB)
# We stream from it so none of this downloads upfront — we just read until
# we have as many tokens as we need.

TOKENIZER_SAMPLE_CHARS = 10_000_000
# How many characters to sample from FineWeb-Edu for BPE tokenizer training.
# 10M chars is enough to learn good merge rules for web text vocabulary.
# The tokenizer training itself takes ~2-5 minutes on this size.

BPE_VOCAB_SIZE = 8000
# Larger than our Shakespeare tokenizer (4000) since web text has a much
# richer and more diverse vocabulary — more languages, technical terms, URLs etc.
# 8000 is a good balance between coverage and embedding table size.

OUTPUT_DIR  = "datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fineweb.bin")
# Binary output file. Stored as uint16 (2 bytes per token).
# uint16 supports vocab sizes up to 65,536 — plenty for our 8000-token vocab.
# 500M tokens × 2 bytes = 1GB on disk.


# ---------------------------------------------------------------------------
# STEP 1: STREAM TEXT FROM FINEWEB-EDU
# ---------------------------------------------------------------------------

def stream_text(target_chars, verbose=True):
    """
    Stream raw text from FineWeb-Edu until we have target_chars characters.

    HuggingFace streaming never downloads the full dataset — it fetches
    shards on demand as you iterate. This means we can stop at any point
    and only the data we actually read gets downloaded.

    target_chars: how many characters of text to collect
    Returns: a single string of concatenated document text
    """
    if verbose:
        print(f"Streaming text from {DATASET_NAME} ({DATASET_CONFIG})...")
        print(f"Target: {target_chars:,} characters\n")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,       # Don't download — stream shard by shard.
        trust_remote_code=True,
    )
    # streaming=True returns an IterableDataset — you iterate over it like a
    # generator. Each item is a dict with keys like 'text', 'url', 'score'.
    # FineWeb-Edu's 'score' field is an educational quality rating (1-5).

    chunks = []
    total_chars = 0

    for example in dataset:
        text = example['text']
        # Each example is one web document — anywhere from a few hundred
        # to tens of thousands of characters.

        chunks.append(text)
        chunks.append('\n\n')
        # Separate documents with a blank line so the model learns
        # document boundaries.

        total_chars += len(text)

        if verbose and total_chars % 1_000_000 < len(text):
            print(f"  Collected {total_chars/1e6:.1f}M / {target_chars/1e6:.1f}M chars...")

        if total_chars >= target_chars:
            break

    result = ''.join(chunks)
    if verbose:
        print(f"\nCollected {len(result):,} characters from FineWeb-Edu.")
    return result


# ---------------------------------------------------------------------------
# STEP 2: (OPTIONALLY) RETRAIN THE BPE TOKENIZER ON FINEWEB VOCABULARY
# ---------------------------------------------------------------------------

def retrain_tokenizer(text_sample, vocab_size=BPE_VOCAB_SIZE):
    """
    Train a new BPE tokenizer on a sample of FineWeb-Edu text.

    We can't reuse the Shakespeare tokenizer here — it has tokens like
    'ROMEO' and 'wherefore' that are useless for web text, and it's
    missing common web patterns like 'http', 'the ', 'ing ', etc.

    text_sample: string — representative sample of the training corpus
    vocab_size:  int — target vocabulary size
    """
    print(f"\nRetraining BPE tokenizer on {len(text_sample):,} chars of FineWeb-Edu...")
    print(f"Target vocab size: {vocab_size}\n")

    vocab, merges, char_to_id = train_bpe(text_sample, vocab_size, verbose=True)
    # Same train() function from tokenizer.py — works on any text corpus.

    import json
    with open('tokenizer.json', 'w') as f:
        json.dump({
            'vocab':      {str(k): v for k, v in vocab.items()},
            'merges':     [[[a, b], c] for (a, b), c in merges],
            'char_to_id': char_to_id,
        }, f)

    print(f"\nNew tokenizer saved to tokenizer.json (vocab size: {vocab_size})")
    return vocab, merges, char_to_id


# ---------------------------------------------------------------------------
# STEP 3: TOKENIZE AND WRITE BINARY FILE
# ---------------------------------------------------------------------------

def tokenize_and_write(text, char_to_id, merges, output_path, chunk_size=100_000):
    """
    Encode text to token IDs and write to a flat binary file.

    We process the text in chunks to avoid holding the full token array
    in RAM. Each chunk is encoded and immediately written to disk.

    text:        string — full training corpus
    output_path: where to write the binary file
    chunk_size:  characters per chunk (trades RAM for speed)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nTokenizing {len(text):,} characters and writing to {output_path}...")

    total_tokens = 0
    with open(output_path, 'wb') as f:
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            # Slice out a chunk of raw text.

            token_ids = bpe_encode(chunk, char_to_id, merges)
            # Encode to a list of integer token IDs.

            arr = np.array(token_ids, dtype=np.uint16)
            # Convert to uint16 numpy array — 2 bytes per token.
            # uint16 range: 0-65535, more than enough for vocab_size=8000.

            f.write(arr.tobytes())
            # Write raw bytes directly — no JSON, no overhead.
            # Reading back: np.memmap(path, dtype=np.uint16, mode='r')

            total_tokens += len(token_ids)

            if (i // chunk_size) % 10 == 0:
                print(f"  {i/len(text)*100:.1f}%  {total_tokens:,} tokens written...")

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nDone. {total_tokens:,} tokens written to {output_path} ({size_mb:.1f} MB)")
    print(f"Compression ratio: {len(text)/total_tokens:.2f}x")
    return total_tokens


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu training data.")

    parser.add_argument("--target-tokens",  type=int, default=50_000_000,
                        help="Approx number of tokens to write to disk (default: 50M).")
    parser.add_argument("--no-retrain-tokenizer", action="store_true",
                        help="Skip tokenizer retraining and use existing tokenizer.json.")
    parser.add_argument("--vocab-size",     type=int, default=BPE_VOCAB_SIZE,
                        help=f"BPE vocabulary size if retraining (default: {BPE_VOCAB_SIZE}).")
    parser.add_argument("--output",         default=OUTPUT_FILE,
                        help=f"Output binary file path (default: {OUTPUT_FILE}).")

    args = parser.parse_args()

    # Estimate characters needed — BPE gives ~3-4x compression on English web text,
    # so to get N tokens we need roughly 3.5*N characters.
    target_chars = int(args.target_tokens * 3.5)

    # --- Stream text ---
    full_text = stream_text(target_chars)

    # --- Retrain tokenizer (or load existing) ---
    if args.no_retrain_tokenizer:
        print("\nLoading existing tokenizer.json...")
        vocab, merges, char_to_id = load_tokenizer('tokenizer.json')
    else:
        # Use the first TOKENIZER_SAMPLE_CHARS chars for tokenizer training.
        sample = full_text[:TOKENIZER_SAMPLE_CHARS]
        vocab, merges, char_to_id = retrain_tokenizer(sample, args.vocab_size)

    # --- Tokenize and write ---
    n_tokens = tokenize_and_write(full_text, char_to_id, merges, args.output)

    print(f"\nReady to train. Update train.py to use: datasets/fineweb.bin")
    print(f"Total tokens available: {n_tokens:,}")


if __name__ == "__main__":
    main()
