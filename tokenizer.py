from collections import Counter

# ---------------------------------------------------------------------------
# BPE TOKENIZER — trained from scratch on a text corpus
#
# Three phases:
#   1. train()  — learns merge rules from corpus, runs once
#   2. encode() — applies learned merges to convert text → token IDs
#   3. decode() — converts token IDs → text
#
# Data structures:
#   vocab:  dict mapping int ID → string (e.g. {0: 'a', 1: 'b', ..., 300: 'th'})
#   merges: list of ((id_a, id_b), new_id) in the order they were learned
# ---------------------------------------------------------------------------


def _count_pairs(ids):
    """
    Count every adjacent pair in a list of token IDs.

    ids: list of ints
    Returns: Counter mapping (id_a, id_b) → frequency
    """
    pairs = Counter()
    for i in range(len(ids) - 1):  # Iterate every adjacent pair.
        pairs[(ids[i], ids[i + 1])] += 1
    return pairs


def _merge(ids, pair, new_id):
    """
    Replace every occurrence of (pair[0], pair[1]) in ids with new_id.

    This is the core BPE operation — we scan through the list once,
    and wherever we see the target pair, we replace both elements with
    the single new merged token.

    ids:    list of ints (current token sequence)
    pair:   tuple (id_a, id_b) — the pair to merge
    new_id: int — the new token ID to replace the pair with
    Returns: new list of ints (shorter than ids by the number of merges made)
    """
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)  # Replace the pair with the new merged token.
            i += 2                 # Skip both elements of the pair.
        else:
            result.append(ids[i])  # Not a match — keep the token as-is.
            i += 1
    return result


def train(text, vocab_size, verbose=True):
    """
    Train a BPE tokenizer on text.

    text:       string — the full training corpus
    vocab_size: int — target vocabulary size (how many tokens to learn)
    verbose:    bool — print progress

    Returns: (vocab, merges, char_to_id)
      vocab:      dict {int: str} — token ID → string representation
      merges:     list of ((id_a, id_b), new_id) — learned merge rules in order
      char_to_id: dict {str: int} — maps individual characters to their base IDs
    """

    # --- Step 1: Build the initial character-level vocabulary ---

    unique_chars = sorted(set(text))
    # sorted() gives us a deterministic ordering of all unique characters in the corpus.
    # For Shakespeare this is ~65 chars: letters, punctuation, newlines, spaces.

    char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
    # Map each character to an integer ID.
    # e.g. {'\n': 0, ' ': 1, '!': 2, ..., 'z': 64}

    vocab = {i: ch for i, ch in enumerate(unique_chars)}
    # Reverse mapping: ID → string.
    # e.g. {0: '\n', 1: ' ', 2: '!', ..., 64: 'z'}

    n_base_tokens = len(unique_chars)
    # Number of base (character-level) tokens — the starting vocab size.

    if verbose:
        print(f"Base vocabulary: {n_base_tokens} characters")
        print(f"Target vocabulary: {vocab_size} tokens")
        print(f"Merges to learn: {vocab_size - n_base_tokens}")
        print(f"Corpus size: {len(text):,} characters\n")

    # --- Step 2: Encode the entire corpus as character IDs ---

    ids = [char_to_id[ch] for ch in text]
    # Convert every character in the corpus to its integer ID.
    # This is our working representation — we'll modify it in place as we merge.
    # Initial length: len(text) (one ID per character)

    # --- Step 3: Iteratively find and merge the most frequent pair ---

    merges = []
    # Will accumulate merge rules as we learn them.
    # Format: [((id_a, id_b), new_id), ...]

    next_id = n_base_tokens
    # The first merged token gets ID = n_base_tokens (right after the base chars).
    # e.g. if we have 65 base chars, the first merge produces token ID 65.

    n_merges = vocab_size - n_base_tokens
    # How many merge rounds we need to run.

    for i in range(n_merges):

        # Count all adjacent pairs in the current token sequence.
        pairs = _count_pairs(ids)

        if not pairs:
            # This shouldn't happen on a real corpus, but guard against it.
            print("No pairs left — stopping early.")
            break

        # Find the most frequent pair.
        best_pair = max(pairs, key=pairs.get)
        best_count = pairs[best_pair]

        # Create the new merged token string by concatenating the two token strings.
        new_token_str = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_id] = new_token_str
        # e.g. if best_pair = (id_of_'t', id_of_'h'), new_token_str = 'th'

        # Record this merge rule.
        merges.append((best_pair, next_id))

        # Apply the merge — replace all occurrences of best_pair in ids with next_id.
        ids = _merge(ids, best_pair, next_id)
        # The token sequence gets shorter by (best_count) elements each round.

        if verbose and (i + 1) % 500 == 0:
            print(f"  Merge {i+1:>4}/{n_merges}  '{new_token_str}' (appeared {best_count:,}x)  "
                  f"vocab size: {next_id + 1}  sequence length: {len(ids):,}")

        next_id += 1

    if verbose:
        print(f"\nDone. Final vocab size: {len(vocab)}")
        print(f"Final sequence length: {len(ids):,} (compression ratio: {len(text)/len(ids):.2f}x)\n")

    return vocab, merges, char_to_id


def encode(text, char_to_id, merges):
    """
    Encode a string into a list of token IDs using the learned merge rules.

    text:       string to encode
    char_to_id: dict {str: int} from train()
    merges:     list of merge rules from train()
    Returns:    list of ints
    """

    # Start with character-level encoding.
    ids = [char_to_id.get(ch, char_to_id.get(' ', 1)) for ch in text]
    # .get(ch, fallback) handles any character not seen during training
    # by falling back to a space. In practice this rarely happens.

    # Apply each merge rule in the order it was learned.
    for pair, new_id in merges:
        if len(ids) < 2:
            break  # Nothing left to merge.
        ids = _merge(ids, pair, new_id)
    # Applying merges in order is critical — common pairs (learned first)
    # get merged before rare ones. The order encodes the frequency hierarchy.

    return ids


def decode(ids, vocab):
    """
    Decode a list of token IDs back to a string.

    ids:   list of ints
    vocab: dict {int: str} from train()
    Returns: string
    """
    return ''.join(vocab[i] for i in ids)
    # Each ID maps to a string (char or merged sequence). Concatenate them all.


# ---------------------------------------------------------------------------
# SCRIPT: train on Shakespeare and print some stats
# ---------------------------------------------------------------------------

def load_tokenizer(path):
    """
    Load a previously trained tokenizer from a JSON file.

    path: path to the JSON file saved by train()
    Returns: (vocab, merges, char_to_id) — same as train() returns
    """
    import json

    with open(path, 'r') as f:
        data = json.load(f)

    vocab = {int(k): v for k, v in data['vocab'].items()}
    # JSON requires string keys, so we stored int IDs as strings. Convert back.

    merges = [((a, b), c) for (a, b), c in data['merges']]
    # Restore the list of ((id_a, id_b), new_id) tuples.

    char_to_id = data['char_to_id']
    # Already the right format: {str: int}

    return vocab, merges, char_to_id


if __name__ == "__main__":
    import json

    VOCAB_SIZE = 4000   # Target: 4000 tokens (vs 65 characters baseline)
    DATA_FILE  = "shakespeare.txt"
    OUT_FILE   = "tokenizer.json"

    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'r') as f:
        text = f.read()

    print("Training BPE tokenizer...\n")
    vocab, merges, char_to_id = train(text, VOCAB_SIZE, verbose=True)

    # Save the tokenizer so we can reuse it without retraining
    with open(OUT_FILE, 'w') as f:
        json.dump({
            'vocab':      {str(k): v for k, v in vocab.items()},
            # JSON requires string keys, so we convert int IDs to strings.
            'merges':     [[[a, b], c] for (a, b), c in merges],
            'char_to_id': char_to_id,
        }, f)
    print(f"Saved tokenizer to {OUT_FILE}")

    # Show some example tokens learned
    print("\nSample learned tokens (first 20 merges):")
    for (pair, new_id) in merges[:20]:
        print(f"  {repr(vocab[pair[0]])} + {repr(vocab[pair[1]])} → {repr(vocab[new_id])}")

    # Test encode/decode round-trip
    test = "ROMEO: Wherefore art thou?"
    encoded = encode(test, char_to_id, merges)
    decoded = decode(encoded, vocab)
    print(f"\nEncode/decode test:")
    print(f"  Input:   {repr(test)}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {repr(decoded)}")
    print(f"  Chars: {len(test)}  Tokens: {len(encoded)}  "
          f"Compression: {len(test)/len(encoded):.1f}x")
