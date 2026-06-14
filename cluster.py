#!/usr/bin/env python3
"""
cluster.py — Submit and manage training jobs on a remote Mac via SSH + pueue.

Configuration is read from .cluster.conf (copy from .cluster.conf.example).
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).parent
CONF_FILE   = REPO_ROOT / ".cluster.conf"
SENTINEL    = REPO_ROOT / ".cluster_ready"
RECENT_JOBS = 10   # how many jobs `jobs` shows by default


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_conf():
    if not CONF_FILE.exists():
        die("Error: .cluster.conf not found. Copy .cluster.conf.example and fill in your details.")
    conf = {}
    with open(CONF_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                conf[k.strip()] = v.split("#")[0].strip()
    for key in ("CLUSTER_HOST", "CLUSTER_USER", "CLUSTER_DIR"):
        if not conf.get(key):
            die(f"Error: {key} not set in .cluster.conf")
    return conf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def ssh_target(conf):
    return f"{conf['CLUSTER_USER']}@{conf['CLUSTER_HOST']}"


def rssh(conf, cmd, check=True, capture=False):
    """Run a single command on the cluster, prepending PATH."""
    full = f"export PATH=/opt/homebrew/bin:$PATH; {cmd}"
    kwargs = {"check": check}
    if capture:
        kwargs.update(capture_output=True, text=True)
    return subprocess.run(["ssh", ssh_target(conf), full], **kwargs)


def rssh_stream(conf, cmd):
    """Run a command on the cluster, streaming combined output to this terminal."""
    full = f"export PATH=/opt/homebrew/bin:$PATH; {cmd}"
    proc = subprocess.Popen(
        ["ssh", ssh_target(conf), full],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        proc.terminate()
    proc.wait()
    return proc.returncode


def sync_code(conf):
    print("Syncing code to cluster...")
    subprocess.run([
        "rsync", "-az",
        "--exclude=.git",
        "--exclude=datasets/",
        "--exclude=checkpoints/",
        "--exclude=wandb/",
        "--exclude=__pycache__/",
        "--exclude=*.pkl",
        "--exclude=.cluster.conf",
        "--exclude=.cluster_ready",
        f"{REPO_ROOT}/",
        f"{ssh_target(conf)}:{conf['CLUSTER_DIR']}/",
    ], check=True)
    print("Sync complete.")


def do_setup(conf):
    print("Bootstrapping cluster...")
    rssh(conf, "brew install pueue && brew services start pueue || true")
    rssh(conf, f"mkdir -p {conf['CLUSTER_DIR']}")
    sync_code(conf)
    rssh(conf, f"pip3 install -r {conf['CLUSTER_DIR']}/requirements.txt")
    SENTINEL.touch()
    print("Cluster ready.")


def ensure_ready(conf):
    if not SENTINEL.exists():
        do_setup(conf)
    else:
        sync_code(conf)
        rssh(conf, f"pip3 install -q -r {conf['CLUSTER_DIR']}/requirements.txt")


def submit_job(conf, cmd, label=""):
    """Write a job script to the cluster and submit via pueue. Returns job ID string."""
    job_script = f"/tmp/cluster_job_{int(time.time())}.sh"
    hf_line    = f"export HF_TOKEN={conf['HF_TOKEN']}" if conf.get("HF_TOKEN") else ""
    script     = (
        f"#!/bin/bash\n"
        f"export PATH=/opt/homebrew/bin:$PATH\n"
        f"{hf_line}\n"
        f"cd {conf['CLUSTER_DIR']}\n"
        f"{cmd}\n"
    )
    subprocess.run(
        ["ssh", ssh_target(conf), f"cat > {job_script} && chmod +x {job_script}"],
        input=script, text=True, check=True,
    )
    label_arg = f"--label {shlex.quote(label)}" if label else ""
    result    = rssh(conf, f"pueue add --print-task-id {label_arg} -- bash {job_script}",
                     capture=True)
    return result.stdout.strip()


def wait_for_wandb_url(conf, job_id, timeout=90):
    """Tail job logs, printing dots, until we see a W&B run URL. Returns URL or None."""
    full    = f"pueue follow {job_id}"
    url_re  = re.compile(r"(https://wandb\.ai/[^\s]+/runs/[^\s]+)")
    proc    = subprocess.Popen(
        ["ssh", ssh_target(conf), f"export PATH=/opt/homebrew/bin:$PATH; {full}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    deadline = time.time() + timeout
    try:
        for line in proc.stdout:
            print(".", end="", flush=True)
            m = url_re.search(line)
            if m:
                proc.terminate()
                return m.group(1)
            if time.time() > deadline:
                break
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()
    return None


def latest_job_id(conf):
    result = rssh(conf, "pueue status --json", capture=True, check=False)
    try:
        tasks = json.loads(result.stdout).get("tasks", {})
    except json.JSONDecodeError:
        die("Couldn't parse pueue output. Is pueue running on the cluster?")
    if not tasks:
        die("No jobs found.")
    return str(max(int(k) for k in tasks))


def status_str(raw):
    """Normalise pueue status which can be a string or e.g. {'Failed': 1}."""
    if isinstance(raw, dict):
        key  = next(iter(raw))
        code = raw[key]
        return f"{key} ({code})"
    return str(raw)


STATUS_COLORS = {
    "Running": "\033[32m",
    "Success": "\033[90m",
    "Queued":  "\033[36m",
    "Killed":  "\033[33m",
    "Paused":  "\033[33m",
}
RESET = "\033[0m"
RED   = "\033[31m"


def format_jobs(tasks):
    lines = [f"  {'ID':>4}  {'Status':<14}  {'Label':<42}  Started"]
    lines.append("  " + "-" * 75)
    for t in tasks:
        raw    = t.get("status", "?")
        st     = status_str(raw)
        color  = STATUS_COLORS.get(st.split()[0], RED)
        start  = (t.get("start") or "")[:16].replace("T", " ") or "-"
        label  = (t.get("label") or "(unlabelled)")[:42]
        lines.append(f"  {t['id']:>4}  {color}{st:<14}{RESET}  {label:<42}  {start}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_generate(args, conf):
    """Run generate.py synchronously — output streams live to your terminal."""
    ensure_ready(conf)
    # shlex.quote each arg so multi-word prompts survive the SSH round-trip
    args_str = " ".join(shlex.quote(a) for a in args)
    hf_env   = f"export HF_TOKEN={conf['HF_TOKEN']}; " if conf.get("HF_TOKEN") else ""
    full_cmd  = (
        f"export PATH=/opt/homebrew/bin:$PATH; "
        f"{hf_env}"
        f"cd {conf['CLUSTER_DIR']} && python3 -u generate.py {args_str}"
    )
    # -t allocates a pseudo-TTY so Ctrl-C propagates to the remote process
    subprocess.run(["ssh", "-t", ssh_target(conf), full_cmd])


def cmd_train(args, conf):
    """Submit a training job and surface the W&B run URL."""
    ensure_ready(conf)
    args_str = " ".join(shlex.quote(a) for a in args)
    cmd      = f"python3 -u train.py {args_str}"
    label    = f"train {args_str}".strip()

    print(f"Submitting: {cmd}")
    job_id = submit_job(conf, cmd, label=label)
    print(f"Job {job_id} submitted. Waiting for W&B URL", end="", flush=True)

    time.sleep(3)  # give pueue a moment to start the process
    url = wait_for_wandb_url(conf, job_id, timeout=90)
    print()        # newline after the dots

    if url:
        print(f"\n  W&B:  {url}")
    else:
        print("\n  W&B URL not detected — the run may still be initialising.")
    print(f"  Logs: ./cluster.py logs {job_id}\n")


def cmd_submit(script, args, conf):
    """Generic queued submission for scripts other than train/generate."""
    ensure_ready(conf)
    if   (REPO_ROOT / f"{script}.py").exists():
        cmd = f"python3 -u {script}.py"
    elif (REPO_ROOT / f"{script}.sh").exists():
        cmd = f"bash {script}.sh"
    elif (REPO_ROOT / script).is_file() and os.access(REPO_ROOT / script, os.X_OK):
        cmd = f"./{script}"
    else:
        die(f"Error: no '{script}', '{script}.py', or '{script}.sh' found in repo root.")

    args_str = " ".join(shlex.quote(a) for a in args)
    full_cmd = f"{cmd} {args_str}".strip()
    label    = f"{script} {args_str}".strip()

    print(f"Submitting: {full_cmd}")
    job_id = submit_job(conf, full_cmd, label=label)
    print(f"Job {job_id} queued.  Logs: ./cluster.py logs {job_id}")


def cmd_jobs(args, conf):
    show_all = "--all" in args
    result   = rssh(conf, "pueue status --json", capture=True, check=False)
    try:
        tasks = list(json.loads(result.stdout).get("tasks", {}).values())
    except (json.JSONDecodeError, AttributeError):
        die("Couldn't parse pueue output. Is pueue running on the cluster?")

    tasks.sort(key=lambda t: t["id"])
    total  = len(tasks)
    subset = tasks if show_all else tasks[-RECENT_JOBS:]

    print(f"\nJobs (showing {len(subset)} of {total}{'' if show_all else f', use --all for full history'}):\n")
    print(format_jobs(subset))
    print()


def cmd_logs(args, conf):
    job_id = args[0] if args else None
    if not job_id:
        job_id = latest_job_id(conf)
        print(f"Tailing latest job ({job_id})...")
    rssh_stream(conf, f"pueue follow {job_id}")


def cmd_cancel(args, conf):
    if not args:
        die("Usage: ./cluster.py cancel <job-id>")
    rssh(conf, f"pueue remove {args[0]}")
    print(f"Job {args[0]} cancelled.")


def cmd_kill(args, conf):
    if not args:
        die("Usage: ./cluster.py kill <job-id>")
    rssh(conf, f"pueue kill {args[0]}")
    print(f"Job {args[0]} killed.")


def cmd_sync(conf):
    sync_code(conf)


def cmd_pull(conf):
    print("Pulling checkpoints from cluster...")
    subprocess.run([
        "rsync", "-az",
        f"{ssh_target(conf)}:{conf['CLUSTER_DIR']}/checkpoints/",
        str(REPO_ROOT / "checkpoints") + "/",
    ], check=True)
    print("Done.")


def cmd_setup(conf):
    do_setup(conf)


def usage():
    print("""\
Usage: ./cluster.py <command> [args]

JOB COMMANDS
  train [options]         Submit training job; prints W&B URL when ready.
  generate [options]      Run inference live in your terminal (not queued).
  <script> [args]         Sync and queue any other script as a pueue job.

  jobs [--all]            Show recent jobs (last 10). --all for full history.
  logs [id]               Tail logs for <id>, or the latest job if omitted.
  cancel <id>             Remove a pending job from the queue.
  kill <id>               Kill a running job immediately.

CLUSTER MANAGEMENT
  sync                    Sync local code to the cluster.
  pull                    Rsync checkpoints/ from cluster back to local.
  setup                   Re-run full cluster bootstrap.

TRAIN OPTIONS
  --n_steps <n>           Training steps (default: 5000)
  --seq_len <n>           Sequence length (default: 512)
  --batch_size <n>        Batch size (default: 32)
  --peak_lr <f>           Peak learning rate (default: 1e-3)
  --dataset <name>        fineweb-edu | c4 | openwebtext
  --run_name <str>        W&B run name

GENERATE OPTIONS
  --prompt <text>         Input text to continue from  (required)
  --n-tokens <n>          Number of new tokens to generate  (required)
  --run-id <id>           W&B run ID to download checkpoint from
  --local-checkpoint <p>  Path to a local .pkl checkpoint file
  --temperature <f>       Sampling temperature (default: 0.8)
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "--help", "-h"):
        usage()
        return

    conf = load_conf()
    cmd  = sys.argv[1]
    rest = sys.argv[2:]

    dispatch = {
        "jobs":   lambda: cmd_jobs(rest, conf),
        "logs":   lambda: cmd_logs(rest, conf),
        "cancel": lambda: cmd_cancel(rest, conf),
        "kill":   lambda: cmd_kill(rest, conf),
        "sync":   lambda: cmd_sync(conf),
        "pull":   lambda: cmd_pull(conf),
        "setup":  lambda: cmd_setup(conf),
    }

    if cmd in dispatch:
        dispatch[cmd]()
    elif cmd == "generate":
        cmd_generate(rest, conf)
    elif cmd == "train":
        cmd_train(rest, conf)
    else:
        cmd_submit(cmd, rest, conf)


if __name__ == "__main__":
    main()
