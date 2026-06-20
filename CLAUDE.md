# nlearn — agent notes

## Repo layout
Before adding or moving files, read [`STRUCTURE.md`](STRUCTURE.md). The core library
is the `nlearn/` package (`nlearn.model`, `nlearn.kernels.{gemm,cross_entropy}`,
`nlearn.data.{streaming,tokenizer}`, etc.); import from it (`from nlearn.model import …`).
Entry points run via `python -m nlearn.train` / `python -m nlearn.generate`. Ops scripts
live in `scripts/`, benchmarks in `bench/`, utilities in `tools/`; `cluster.py` and the
env bootstraps (`iree_env.sh`/`.runenv.sh`) stay at root.

## Leaderboard
Runs post to the shared multi-tenant leaderboard as project `nlearn`.
Canonical how/when-to-post guide: **https://leaderboard.nikliolios.com/POSTING.md**
(always fetch the latest before changing posting). A `SessionStart` hook
(`.claude/settings.json`) auto-loads it into context each session.
Posting is wired through `nlearn/leaderboard.py` (env: `~/.config/nlearn/leaderboard.env`);
board schema lives in `leaderboard.config.json`.
