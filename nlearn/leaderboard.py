"""
nlearn.leaderboard — push to a multi-tenant leaderboard.

Non-disruptive by design: posting runs on a background daemon thread with a short
timeout and never raises, so it can't block or crash a training loop. If the env
isn't configured it's a silent no-op.

Config (env):
  LEADERBOARD_URL          e.g. https://leaderboard.nikliolios.com
  LEADERBOARD_PROJECT      your project id, e.g. "nlearn"
  LEADERBOARD_TOKEN        the project's write token
  LEADERBOARD_ADMIN_TOKEN  admin token (only for `projects create`)

Library:
    from nlearn.leaderboard import post_entry, register_config
    register_config({"title": "...", "boards": [...]})        # once, on deploy
    post_entry("pretraining", {"id": "run-1", "metrics": {...}})

CLI:
    python leaderboard_client.py projects create nlearn       # needs admin token
    python leaderboard_client.py config push config.json
    python leaderboard_client.py post <board> '<entry json>'
"""

import json
import os
import threading
import urllib.request

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")  # Cloudflare bot rule rejects urllib's default UA


def _cfg():
    return (os.environ.get("LEADERBOARD_URL", "").rstrip("/"),
            os.environ.get("LEADERBOARD_PROJECT", ""),
            os.environ.get("LEADERBOARD_TOKEN", ""))


def is_enabled():
    url, project, _ = _cfg()
    return bool(url and project)


def _request(method, path, body=None, token=None, timeout=8.0):
    url, _, _ = _cfg()
    if not url:
        return None
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json", "User-Agent": _UA}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _post_sync(board, entry, timeout):
    _, project, token = _cfg()
    try:
        _request("POST", f"/api/{project}/{board}", entry, token=token, timeout=timeout)
        return True
    except Exception:
        return False  # never surface leaderboard problems into the caller


def post_entry(board, entry, blocking=False, timeout=6.0):
    """Upsert one entry to the current project's board. Fire-and-forget by default."""
    if not is_enabled():
        return False
    if "id" not in entry or "metrics" not in entry:
        raise ValueError("entry must contain 'id' and 'metrics'")
    if blocking:
        return _post_sync(board, entry, timeout)
    threading.Thread(target=_post_sync, args=(board, entry, timeout), daemon=True).start()
    return True


def register_config(config, timeout=10.0):
    """Register/update this project's board schema (PUT /config). Blocking."""
    if not is_enabled():
        return False
    _, project, token = _cfg()
    try:
        _request("PUT", f"/api/{project}/config", config, token=token, timeout=timeout)
        return True
    except Exception as e:
        print("register_config failed:", e)
        return False


def create_project(project, admin_token=None, timeout=10.0):
    """Admin: create a project; returns its write token (printed once)."""
    admin_token = admin_token or os.environ.get("LEADERBOARD_ADMIN_TOKEN", "")
    return _request("POST", "/api/admin/projects", {"project": project}, token=admin_token, timeout=timeout)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    cmd = args[0]
    if cmd == "projects" and len(args) >= 3 and args[1] == "create":
        print(json.dumps(create_project(args[2]), indent=2))
    elif cmd == "config" and len(args) >= 3 and args[1] == "push":
        with open(args[2]) as f:
            print("registered:", register_config(json.load(f)))
    elif cmd == "post" and len(args) >= 3:
        print("posted:", post_entry(args[1], json.loads(args[2]), blocking=True))
    else:
        print(__doc__)
        sys.exit(1)
