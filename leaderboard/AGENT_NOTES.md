# Leaderboard — notes for the nlearn agent

A live leaderboard was added to this project at **https://leaderboard.nikliolios.com**
with four boards — Pretraining, FlashAttention forward, FlashAttention backward,
and GEMM. It's a Cloudflare
Worker + KV store; the code lives in `leaderboard/`. You don't need to touch the
Worker — it's deployed and running. You only need to produce data for it.

## What's already wired up

- **`leaderboard_client.py`** (repo root) — a stdlib-only, fire-and-forget POST
  client. Silent no-op if the env vars below are unset, and never raises, so it
  can't break training or commits.
- **`logging_utils.py`** — `TrainingLogger` posts to the **Pretraining** board
  automatically alongside W&B: a `running` row updates on each validation step,
  a `done` row posts when the run ends. `train.py` already passes
  `run_name`/`dataset` and calls `logger.finalize(...)` on both exit paths.
- **`cluster.py`** — `submit_job` reads `LEADERBOARD_URL`/`LEADERBOARD_TOKEN`
  (from `.cluster.conf`, the process env, or `~/.config/nlearn/leaderboard.env`)
  and forwards them into the remote job script, alongside `HF_TOKEN`/`WANDB_API_KEY`.
  So cluster training runs post with no extra flags.
- **`bench_kernels.py`** (repo root) — benchmarks the kernels and posts to the
  **FlashAttention** and **GEMM** boards.
- **`leaderboard/hooks/`** — a git pre-commit hook that benchmarks kernels on
  change.

## Env vars

These enable posting. They're stored on the Mac at
`~/.config/nlearn/leaderboard.env`, auto-sourced by interactive shells:

- `LEADERBOARD_URL=https://leaderboard.nikliolios.com`
- `LEADERBOARD_TOKEN=<write token>`

If you run `train.py`/`bench_kernels.py` directly from a non-interactive shell,
`source ~/.config/nlearn/leaderboard.env` first. (`cluster.py` reads that file
itself, so cluster jobs don't depend on the shell.)

## ⚠️ One action before the next cluster run

Re-sync the repo to the cluster (`./cluster.py sync`, or `./cluster.py setup`).
`logging_utils.py` now imports `leaderboard_client.py`; if that new file isn't on
the cluster, `train.py` will fail to import. No new pip dependencies (stdlib only).

## How to populate each board

- **Pretraining** — train as usual: `./cluster.py train --run-name <name>` (or
  `python train.py --run-name <name>` locally). Rows are keyed by `--run-name`:
  reusing a name overwrites that row, so use distinct names for distinct runs.
  Ranking is validation loss at a fixed compute budget, `LEADERBOARD_FLOP_BUDGET`
  (default `1e16`, overridable via env); runs shorter than the budget show "—"
  there and fall back to best val loss.
  - Note: the supervisor (`supervise_run.sh`) sets `NLEARN_NO_VAL=1`, so
    validation loss is NaN and won't populate. Those rows still show train loss,
    TFLOP/s, MFU, and tokens, but won't rank by val loss. Run with validation
    enabled (don't set `NLEARN_NO_VAL`) if you want the ranking metric.
- **FlashAttention (forward)** — `python bench_kernels.py --flash`. Benchmarks
  `attention.py:attention` (the dispatch function), so it measures whichever
  implementation is active (the IREE-Metal flash kernel when on that backend,
  else standard). Metrics: TFLOP/s, latency, speedup vs naive dense attention,
  peak mem, max abs error, shape, dtype.
- **FlashAttention (backward)** — `python bench_kernels.py --flash-bwd`. Times
  the VJP of `attention()` in isolation (forward residuals captured first, not
  timed), i.e. the dQ/dK/dV work — the `flash_attention_bwd_dq`/`_dkdv` kernels
  when flash is active. Posts to the **flashattention_bwd** board (UI tab "Flash
  bwd"). TFLOP/s uses backward ≈ 2.5× forward FLOPs; max abs error is the max
  over dQ/dK/dV vs an analytic reference.
- **GEMM** — `python bench_kernels.py --gemm`. Benchmarks `gemm_iree.matmul`,
  which dispatches the hand-authored Metal simdgroup GEMM kernel on IREE-Metal
  (f16 in / f32 accumulate) and falls back to `jnp.matmul` elsewhere. Defaults to
  float16 inputs at 4096³; metrics: TFLOP/s, % of peak, latency, shape, dtype,
  max abs error.
- **Automatic on change** — run `bash leaderboard/hooks/install_hooks.sh` once to
  symlink the pre-commit hook. After that, commits touching `attention.py`,
  `gemm_iree.py`, or `iree_metal/kernels/**` kick off a benchmark **in the
  background** — it never blocks or slows the commit. You can also run
  `bench_kernels.py` manually anytime. Kernel rows are keyed by a content hash of
  the source, so re-running unchanged code updates the same row and a code change
  creates a new one.

## Gotchas

- The board can lag a write by up to ~60s (KV read cache). The page
  auto-refreshes every 15s, so a finished run appears within about a minute —
  don't assume it failed if it's not instant.
- If you inspect KV with `wrangler kv key ...`, pass `--remote` — Wrangler v4
  defaults to a *local* store and will show stale/empty data otherwise.
- Adding a metric: include it in the `metrics` dict you post, then add a matching
  column to the board's column spec in `leaderboard/worker/public/index.html`.

See `leaderboard/README.md` for the API shape, deploy steps, and full details.
