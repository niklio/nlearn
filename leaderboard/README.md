# nlearn leaderboard

Live leaderboard at **leaderboard.nikliolios.com** for the optimization problems
the nlearn agent works on. Four boards:

| Board (UI tab) | What it tracks | Ranked by |
|---|---|---|
| **Pretraining** (`pretraining`) | full model training runs | validation loss at a fixed FLOP budget (lower = better) |
| **Flash fwd** (`flashattention`) | the attention forward kernel (`attention.py:attention`) | TFLOP/s (higher = better) |
| **Flash bwd** (`flashattention_bwd`) | the attention backward kernel (VJP of `attention()`: dQ/dK/dV) | TFLOP/s (higher = better) |
| **GEMM** (`gemm`) | the matmul path (`gemm_iree.matmul`) | TFLOP/s (higher = better) |

Metrics are columns. The frontend is mobile-first: each row is a tappable card,
big tap targets, no horizontal scroll, auto-refreshes every 15s.

## Architecture

```
training run ─┐                        ┌─ GET /api/:board ──> browser (live)
              ├─ leaderboard_client ──>│
kernel bench ─┘     (HTTPS POST)       └─ Cloudflare Worker + KV
                                          (src/index.js, one JSON array per board)
                                          serves the static frontend (public/)
```

- **`worker/`** — Cloudflare Worker. `src/index.js` is the API (`/api/*`);
  everything else serves `public/index.html` (the frontend). Data lives in a KV
  namespace, one JSON array per board, entries upserted by `id`.
- **`../leaderboard_client.py`** — tiny non-blocking POST client (repo root, so
  `from leaderboard_client import post_entry` works everywhere).
- **`../bench_kernels.py`** — benchmarks the kernels and posts to the board.
- **`hooks/`** — git pre-commit hook that re-benchmarks kernels on change.

## One-time deploy

```bash
cd leaderboard/worker
npm i -g wrangler          # or use npx wrangler
wrangler login

# 1. Create the KV namespace and paste the ids into wrangler.toml
wrangler kv namespace create LEADERBOARD_KV
wrangler kv namespace create LEADERBOARD_KV --preview

# 2. Set the write token (any strong secret)
wrangler secret put LEADERBOARD_TOKEN

# 3. Deploy — custom_domain in wrangler.toml auto-creates the DNS record
#    for leaderboard.nikliolios.com in the Cloudflare zone.
wrangler deploy
```

## Wiring the posters

Both the training logger and the kernel benchmark post only when these env vars
are set (otherwise they are silent no-ops, so nothing breaks if you forget):

```bash
export LEADERBOARD_URL=https://leaderboard.nikliolios.com
export LEADERBOARD_TOKEN=<the secret you set above>
```

### Pretraining (automatic)

`logging_utils.py:TrainingLogger` already pushes to the leaderboard alongside
W&B — a `running` row updates on each validation step and a final `done` row is
posted when the run ends. Nothing else to do; just train as usual:

```bash
./cluster.py train --run-name my-run     # cluster (vars forwarded automatically)
python train.py --run-name my-run        # local
```

The FLOP budget used for ranking defaults to `1e16`; override with
`LEADERBOARD_FLOP_BUDGET`.

### Kernels (presubmit hook)

```bash
bash leaderboard/hooks/install_hooks.sh
```

Now committing changes to `attention.py`, `gemm_iree.py`, or
`iree_metal/kernels/**` kicks off a background benchmark (it never blocks the
commit) that posts fresh numbers. Run it manually any time:

```bash
python bench_kernels.py --flash          # attention forward (attention.py:attention)
python bench_kernels.py --flash-bwd      # attention backward (VJP of attention())
python bench_kernels.py --gemm           # GEMM kernel (gemm_iree.matmul)
python bench_kernels.py                  # all of the above
```

Each kernel version is a row, keyed by a content hash of the source — re-running
unchanged code updates the same row; a code change posts a new one.

## Local development

```bash
cd leaderboard/worker
echo 'LEADERBOARD_TOKEN=devtoken' > .dev.vars     # gitignored
wrangler dev --local                              # http://localhost:8799
# point the posters at it:  LEADERBOARD_URL=http://localhost:8799 LEADERBOARD_TOKEN=devtoken
```

## API

- `GET  /api/boards` → `{ boards: [...] }`
- `GET  /api/:board` → `{ board, entries: [...], updated_at }`
- `POST /api/:board` → upsert one entry (requires `Authorization: Bearer <token>`)
  - body: `{ "id": "...", "name": "...", "status": "running"|"done", "metrics": { ... } }`
