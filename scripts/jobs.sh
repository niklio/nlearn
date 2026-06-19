#!/usr/bin/env bash
# jobs.sh — at-a-glance health of my nlearn jobs. For each python train/bench
# process: elapsed, total CPU, %CPU, RSS, and the last 3 log lines. A process
# with high ELAPSED but flat TIME / 0.0 %CPU is HUNG (the failure mode that ate
# 13h). Run anytime to poll without relying on completion notifications.
cd "$(dirname "$0")/.."
echo "=== live python jobs ==="
PIDS=$(pgrep -f "venvs/iree/bin/python" || true)
if [ -z "$PIDS" ]; then echo "(none)"; else
  ps -o pid,etime,time,%cpu,rss -p $(echo "$PIDS" | tr '\n' ',' | sed 's/,$//')
fi
echo "=== recent logs ==="
for f in $(ls -t logs/*.log 2>/dev/null | head -4); do
  echo "--- $f (mtime $(stat -f %Sm -t %H:%M:%S "$f")) ---"
  tail -3 "$f"
done
