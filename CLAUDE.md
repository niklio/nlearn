# nlearn — agent notes

## Repo layout
Before adding or moving files, read [`STRUCTURE.md`](STRUCTURE.md): core library
modules stay flat at root (bare-name imports), ops scripts live in `scripts/`,
benchmarks in `bench/`, utilities in `tools/`. Subdir scripts that import a root
module need the `sys.path` bootstrap; `iree_env.sh`/`.runenv.sh` must stay at root.

## Leaderboard
Runs post to the shared multi-tenant leaderboard as project `nlearn`.
Canonical how/when-to-post guide: **https://leaderboard.nikliolios.com/POSTING.md**
(always fetch the latest before changing posting). A `SessionStart` hook
(`.claude/settings.json`) auto-loads it into context each session.
Posting is wired through `leaderboard_client.py` (env: `~/.config/nlearn/leaderboard.env`);
board schema lives in `leaderboard.config.json`.
