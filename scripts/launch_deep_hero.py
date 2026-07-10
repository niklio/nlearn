#!/usr/bin/env python3
"""Detached launcher for the DEEP HERO run — target val < 3.0 (2026-07-10).

A 24-layer / 209M model (d=768), trainable on this metal-spirv box ONLY via the depth-ceiling fix:
  - NLEARN_SCAN_LAYERS=1 : lax.scan over layers (forward) + stacked block params (init_model), so the
    compiled graph is constant-size in depth. Cracks the ~12-layer degeneration ceiling.
  - split apply_fn (make_accum_steps): jit ONLY optimizer.update; mean/apply eager — else the fused
    apply graph miscompiles the deep model (loss spikes ~15).
Verified: 24L descends 11.0 -> 9.5 stably. Depth ~2x the old 123M (floor 3.47), so <3.0 is in reach.

AdamW (not Muon): Muon needs stacked-param NS/labeling work; AdamW-deep is proven. Well-formed LR
(warmup 200 << eff 125k updates -> cosine anneals over the marathon). Process-rest survives DCP panics.

Usage: python3 scripts/launch_deep_hero.py
"""
import os, subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "deep_hero"
PIDFILE = os.path.join(REPO, ".deep_hero_supervisor.pid")

CONFIG_ENV = {
    "NLEARN_SCAN_LAYERS": "1",         # depth-ceiling fix: scan + stacked params
    "NLEARN_N_LAYERS": "24",           # GPT-2-medium depth (2x the 123M)
    "NLEARN_UNROLL_GROUP": "3",        # partial unroll: scan 8 groups of 3 unrolled layers.
                                       # Profiled (profile_deep_mfu.py): MFU 28.6%->37.5%, 4.06->3.62s/step
                                       # vs single-layer scan. Free (same math); g3 beat g2/g4/g6.
    "NLEARN_OPTIMIZER": "adamw",       # Muon-deep deferred (needs stacked NS)
    "NLEARN_MANUAL_ACCUM": "1",        # split grad/apply -> under the graph ceiling
    # d=768 default (proven-trainable, faster than d=1024). grad-clip + hard-val on via .runenv.
}
# run steps seq BATCH ACCUM lr : eff batch 32; lr 3e-4 (stable for deep); steps=2e6 -> eff 125k updates
CMD = ["bash", os.path.join(REPO, "scripts", "supervise.sh"),
       RUN, "2000000", "1024", "2", "16", "3e-4"]

# Record CURRENT_RUN so the run-agnostic watchdog supervises + survives reboots.
try:
    with open(os.path.expanduser("~/.claude-rc/CURRENT_RUN"), "w") as f:
        f.write(f"{RUN}\ncd {REPO} && python3 scripts/launch_deep_hero.py\n")
except OSError:
    pass

if os.fork() > 0: os._exit(0)
os.setsid()
if os.fork() > 0: os._exit(0)
env = dict(os.environ); env.update(CONFIG_ENV)
os.chdir(REPO)
logf = open(f"/tmp/superv_{RUN}_launch.log", "a")
p = subprocess.Popen(CMD, env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
with open(PIDFILE, "w") as f: f.write(str(p.pid) + "\n")
p.wait(); os._exit(0)
