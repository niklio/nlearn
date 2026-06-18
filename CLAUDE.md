# nlearn — agent notes

## Leaderboard
Runs post to the shared multi-tenant leaderboard as project `nlearn`.
Canonical how/when-to-post guide: **https://leaderboard.nikliolios.com/POSTING.md**
(always fetch the latest before changing posting). A `SessionStart` hook
(`.claude/settings.json`) auto-loads it into context each session.
Posting is wired through `leaderboard_client.py` (env: `~/.config/nlearn/leaderboard.env`);
board schema lives in `leaderboard.config.json`.
