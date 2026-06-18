# Leaderboard — notes for the nlearn agent

The leaderboard is a **standalone multi-tenant service** (repo `niklio/leaderboard`,
live at **https://leaderboard.nikliolios.com**). nlearn is the project `nlearn` on it.
The Worker/frontend are NOT in this repo anymore — don't deploy anything from here.
This repo just *posts* to the service.

## Config (env, auto-sourced from ~/.config/nlearn/leaderboard.env)
```
export LEADERBOARD_URL=https://leaderboard.nikliolios.com
export LEADERBOARD_PROJECT=nlearn
export LEADERBOARD_TOKEN=<nlearn project write token>
```
Posting is a silent no-op if these are unset, so it never breaks training.

## How posting works (already wired)
- `leaderboard_client.py` (repo root) — `post_entry(board, entry)` posts to
  `/api/nlearn/<board>`. Used by `logging_utils.py` (pretraining runs) and
  `bench_kernels.py` (kernel boards). No code change needed to use it.
- Boards: `pretraining`, `flashattention`, `gemm`, `crossentropy`.
- Provenance (codebase snapshot link + kernel-version links) rides along as entry
  fields (`commit_url`, `kernels`) — still works; the service stores and renders them.

## Board schema
`leaderboard.config.json` (repo root) is nlearn's schema, already registered on the
service. If you change columns/boards, re-register:
```
python leaderboard_client.py config push leaderboard.config.json
```

## Kernel presubmit
`hooks/` still benchmarks kernels on commit and posts to the flash/gemm boards.
Install with `bash leaderboard/hooks/install_hooks.sh`.

## Onboarding a new project (other repos)
Provisioning uses the admin token (kept by Nik). See `niklio/leaderboard` README.
