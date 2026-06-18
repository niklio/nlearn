# leaderboard (nlearn integration)

The leaderboard is now a standalone multi-tenant service — code and deploy live in
**`niklio/leaderboard`**, hosted at **https://leaderboard.nikliolios.com**. nlearn is
the project `nlearn` on it; this directory only holds nlearn's *integration* with it.

What's here / at the repo root:
- `../leaderboard_client.py` — the shared client. Posts to the service as the project
  in `$LEADERBOARD_PROJECT`. Same `post_entry(board, entry)` API the training logger
  and kernel bench already use.
- `../leaderboard.config.json` — nlearn's board schema (pretraining, flashattention,
  gemm, crossentropy). Re-register after editing:
  `python leaderboard_client.py config push leaderboard.config.json`
- `hooks/` — the kernel presubmit hook (benchmarks kernels on commit and posts).

Config (in `~/.config/nlearn/leaderboard.env`, auto-sourced):
```
export LEADERBOARD_URL=https://leaderboard.nikliolios.com
export LEADERBOARD_PROJECT=nlearn
export LEADERBOARD_TOKEN=<nlearn project write token>
```

See `AGENT_NOTES.md` for the full how-to.
