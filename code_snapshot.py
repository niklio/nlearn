"""
code_snapshot.py — push a browsable git snapshot of the working tree to GitHub.

A training run's codebase snapshot, without requiring the run dir to be a tracked
git repo. We stage the tree's current files into a throwaway index (borrowing a
local clone's object store), write a tree + orphan commit, and push it to
refs/runs/<name>. GitHub then serves github.com/<repo>/tree/<sha> as a browsable,
ZIP-downloadable snapshot — deduped by git's content-addressed object store.

Requires (best-effort; returns None if missing, so callers never break):
  - a local clone of the repo with a GitHub remote (object store + push), found at
    $NLEARN_SNAPSHOT_GIT_DIR or ~/nlearn-git/.git
  - a GitHub token in $GH_TOKEN / $GITHUB_TOKEN / ~/.gh_token

Honors the work tree's .gitignore and excludes build artifacts (dylibs, the IREE
runtime bundle, vmfb, pickles).
"""

import os
import subprocess
import tempfile

REPO = os.environ.get("NLEARN_GH_REPO", "niklio/nlearn")
# Pathspec excludes on top of the work tree's .gitignore.
EXCLUDES = [":!.iree_runtime", ":!*.dylib", ":!*.vmfb", ":!*.pkl", ":!.git"]


def _git(args, env):
    return subprocess.run(["git"] + args, env=env, capture_output=True, text=True)


def _gh_token():
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.gh_token")
    return open(p).read().strip() if os.path.exists(p) else None


def _find_git_dir():
    for cand in (os.environ.get("NLEARN_SNAPSHOT_GIT_DIR"),
                 os.path.expanduser("~/nlearn-git/.git")):
        if cand and os.path.isdir(cand):
            return cand
    return None


def _safe(s):
    out = "".join(c if (c.isalnum() or c in "-_./") else "-" for c in str(s))
    return out.strip("/") or "run"


def snapshot(work_tree, ref_name, git_dir=None, repo=REPO):
    """Snapshot `work_tree` and push it to refs/runs/<ref_name> on `repo`.

    Returns {"commit", "tree", "url"} or None if a snapshot couldn't be made.
    """
    git_dir = git_dir or _find_git_dir()
    token = _gh_token()
    work_tree = os.path.abspath(os.path.expanduser(work_tree))
    if not git_dir or not token or not os.path.isdir(work_tree):
        return None

    env = dict(os.environ)
    env["GIT_DIR"] = git_dir
    env["GIT_WORK_TREE"] = work_tree
    env["GIT_INDEX_FILE"] = tempfile.mktemp(prefix="nlearn-snap-")  # fresh; git creates it
    env.setdefault("GIT_AUTHOR_NAME", "nlearn"); env.setdefault("GIT_AUTHOR_EMAIL", "nlearn@local")
    env.setdefault("GIT_COMMITTER_NAME", "nlearn"); env.setdefault("GIT_COMMITTER_EMAIL", "nlearn@local")

    try:
        if _git(["add", "-A", "--", "."] + EXCLUDES, env).returncode != 0:
            return None
        tree = _git(["write-tree"], env).stdout.strip()
        if not tree:
            return None
        commit = _git(["commit-tree", tree, "-m", f"run snapshot: {ref_name}"], env).stdout.strip()
        if not commit:
            return None
        ref = f"refs/runs/{_safe(ref_name)}"
        push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
        if _git(["push", "--force", push_url, f"{commit}:{ref}"], env).returncode != 0:
            return None
        return {"commit": commit, "tree": tree, "url": f"https://github.com/{repo}/tree/{commit}"}
    except Exception:
        return None
    finally:
        try:
            os.unlink(env["GIT_INDEX_FILE"])
        except OSError:
            pass


if __name__ == "__main__":
    import sys
    wt = sys.argv[1] if len(sys.argv) > 1 else "."
    name = sys.argv[2] if len(sys.argv) > 2 else "_cli_test"
    print(snapshot(wt, name))
