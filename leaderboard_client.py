"""
leaderboard_client.py — post entries to the nlearn leaderboard.

Designed to be completely non-disruptive: if the leaderboard env vars are
unset it is a silent no-op, and posts run on a background daemon thread with a
short timeout so they can never block (or crash) a training loop or a commit.

Config (env):
  LEADERBOARD_URL    e.g. https://leaderboard.nikliolios.com   (required to enable)
  LEADERBOARD_TOKEN  write token matching the Worker secret     (required to enable)

Usage:
    from leaderboard_client import post_entry
    post_entry("pretraining", {
        "id": "run-2026-06-17a",
        "name": "run-2026-06-17a",
        "status": "running",
        "metrics": {"val_loss": 3.21, "mfu": 0.34, ...},
    })
"""

import json
import os
import threading
import urllib.request

_VALID_BOARDS = ("pretraining", "flashattention", "gemm")


def is_enabled():
    return bool(os.environ.get("LEADERBOARD_URL"))


def _post_sync(board, entry, timeout):
    url = os.environ.get("LEADERBOARD_URL")
    token = os.environ.get("LEADERBOARD_TOKEN", "")
    if not url:
        return False
    try:
        body = json.dumps(entry).encode("utf-8")
        req = urllib.request.Request(
            f"{url.rstrip('/')}/api/{board}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        # Never let leaderboard problems surface into the caller.
        return False


def post_entry(board, entry, blocking=False, timeout=4.0):
    """Upsert one leaderboard entry.

    `entry` must contain a string `id` and a `metrics` dict. By default the
    request is fired on a daemon thread and this returns immediately. Pass
    blocking=True (e.g. in a one-shot CLI) to wait and get the bool result.
    """
    if board not in _VALID_BOARDS:
        raise ValueError(f"board must be one of {_VALID_BOARDS}, got {board!r}")
    if not is_enabled():
        return False
    if "id" not in entry or "metrics" not in entry:
        raise ValueError("entry must contain 'id' and 'metrics'")

    if blocking:
        return _post_sync(board, entry, timeout)

    t = threading.Thread(
        target=_post_sync, args=(board, entry, timeout), daemon=True
    )
    t.start()
    return True


if __name__ == "__main__":
    # Smoke test:  LEADERBOARD_URL=... LEADERBOARD_TOKEN=... python leaderboard_client.py
    import sys
    ok = post_entry(
        "pretraining",
        {
            "id": "smoke-test",
            "name": "smoke-test",
            "status": "done",
            "metrics": {"val_loss": 9.99, "mfu": 0.1, "tokens": 1000, "dataset": "test"},
        },
        blocking=True,
    )
    print("posted:", ok, "(enabled:", is_enabled(), ")")
    sys.exit(0 if ok or not is_enabled() else 1)
