#!/usr/bin/env python3
"""Profile deep-model (24L scan) throughput across remat strategies — find the MFU regression.

Runs the real training entrypoint for a short burst under 3 configs, throwaway run-names (so the
deep_hero checkpoint is untouched), and reports steady-state MFU (mean of the last few steps, past
cold-compile). Variants:
  full   : scan + full jax.checkpoint (deep_hero's current config)
  policy : scan + jax.checkpoint(dots_with_no_batch_dims_saveable) — keep matmuls, recompute cheap ops
  none   : scan + NO remat (hold all layer activations)
Self-times-out per variant so a compile hang can't wedge the box.
"""
import os, re, subprocess, sys, statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = "16"
BASE = dict(os.environ)
BASE.update({
    "NLEARN_SCAN_LAYERS": "1", "NLEARN_N_LAYERS": "24",
    "NLEARN_OPTIMIZER": "adamw", "NLEARN_MANUAL_ACCUM": "1",
    "NLEARN_GRAD_ACCUM": "16", "NLEARN_GRAD_CLIP_VALUE": "1.0",
    "NLEARN_REST_EVERY_S": "999999",           # no process-rest during the probe
    "WANDB_MODE": "disabled",
})
VARIANTS = [
    ("g1", {}),                            # baseline: scan over single layers (current deep_hero)
    ("g2", {"NLEARN_UNROLL_GROUP": "2"}),  # scan over 12 groups of 2 unrolled layers
    ("g3", {"NLEARN_UNROLL_GROUP": "3"}),  # 8 groups of 3
    ("g4", {"NLEARN_UNROLL_GROUP": "4"}),  # 6 groups of 4
    ("g6", {"NLEARN_UNROLL_GROUP": "6"}),  # 4 groups of 6 (still < ~12-layer ceiling)
]
STEP_RE = re.compile(r"^Step\s+(\d+)\s+loss:\s+([\d.]+).*?mfu:\s+([\d.]+)%.*?step:\s+([\d.]+)s", re.M)

def run(name, extra):
    env = dict(BASE); env.update(extra)
    cmd = [sys.executable, "-m", "nlearn.train", "--steps", STEPS,
           "--batch-size", "2", "--seq-len", "1024", "--peak-lr", "3e-4",
           "--run-name", f"mfuprobe_{name}"]
    print(f"\n=== variant={name} extra={extra} ===", flush=True)
    try:
        p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        def _d(x): return x.decode() if isinstance(x, (bytes, bytearray)) else (x or "")
        out = _d(e.stdout) + _d(e.stderr)
        print(f"  [TIMEOUT after 300s]", flush=True)
    rows = STEP_RE.findall(out)
    if not rows:
        print("  no step rows parsed; tail:\n" + "\n".join(out.splitlines()[-15:]), flush=True)
        return None
    # steady state = drop first 6 steps (cold compile / recompile), average the rest
    steady = rows[6:] if len(rows) > 8 else rows[1:]
    mfus = [float(r[2]) for r in steady]
    times = [float(r[3]) for r in steady]
    losses = [float(r[1]) for r in steady]
    res = dict(name=name, mfu=statistics.mean(mfus), step=statistics.mean(times),
               loss_last=losses[-1], n=len(steady))
    print(f"  steady MFU={res['mfu']:.1f}%  step={res['step']:.2f}s  "
          f"loss={res['loss_last']:.3f}  (n={res['n']})", flush=True)
    return res

def main():
    results = [run(n, e) for n, e in VARIANTS]
    print("\n=== SUMMARY (deep 24L, bs2/seq1024, accum16) ===", flush=True)
    base = next((r for r in results if r and r["name"] == "g1"), None)
    for r in results:
        if not r: continue
        spd = f"  ({base['step']/r['step']:.2f}x vs g1)" if base and r is not base else ""
        degen = "  <DEGENERATE?>" if abs(r["loss_last"] - 10.8249) < 0.02 else ""
        print(f"  {r['name']:7s}  MFU {r['mfu']:5.1f}%   {r['step']:.2f}s/step   "
              f"loss~{r['loss_last']:.2f}{spd}{degen}", flush=True)

if __name__ == "__main__":
    main()
