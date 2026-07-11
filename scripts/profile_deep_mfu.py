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
# Batch-size sweep at the winning group (g3), holding EFFECTIVE batch = bs*accum = 32 so training
# dynamics/LR are unchanged. g3 used only 3.7GB/16 at bs2 -> a bigger microbatch should fill the GPU
# and raise MFU (small batches underutilize the cores). "bs"/"accum" override the CLI/env per variant.
G3 = {"NLEARN_UNROLL_GROUP": "3"}
VARIANTS = [
    ("bs2_a16",  {**G3, "bs": "2",  "accum": "16"}),   # control (current hero)
    ("bs4_a8",   {**G3, "bs": "4",  "accum": "8"}),
    ("bs8_a4",   {**G3, "bs": "8",  "accum": "4"}),
    ("bs16_a2",  {**G3, "bs": "16", "accum": "2"}),
]
STEP_RE = re.compile(r"^Step\s+(\d+)\s+loss:\s+([\d.]+).*?mfu:\s+([\d.]+)%.*?mem:\s+(\d+)MB.*?step:\s+([\d.]+)s", re.M)

def run(name, extra):
    extra = dict(extra)
    bs = extra.pop("bs", "2")
    accum = extra.pop("accum", None)
    env = dict(BASE); env.update(extra)
    if accum is not None:
        env["NLEARN_GRAD_ACCUM"] = accum   # override effective-batch split
    cmd = [sys.executable, "-m", "nlearn.train", "--steps", STEPS,
           "--batch-size", bs, "--seq-len", "1024", "--peak-lr", "3e-4",
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
    mems = [float(r[3]) for r in steady]
    times = [float(r[4]) for r in steady]
    losses = [float(r[1]) for r in steady]
    res = dict(name=name, mfu=statistics.mean(mfus), step=statistics.mean(times),
               mem=max(mems), loss_last=losses[-1], n=len(steady))
    print(f"  steady MFU={res['mfu']:.1f}%  step={res['step']:.2f}s  "
          f"mem={res['mem']:.0f}MB  loss={res['loss_last']:.3f}  (n={res['n']})", flush=True)
    return res

def main():
    results = [run(n, e) for n, e in VARIANTS]
    print("\n=== SUMMARY (deep 24L, bs2/seq1024, accum16) ===", flush=True)
    base = next((r for r in results if r and r["name"] == "bs2_a16"), None)
    for r in results:
        if not r: continue
        spd = f"  ({r['mfu']/base['mfu']:.2f}x MFU vs bs2)" if base and r is not base else ""
        degen = "  <DEGENERATE?>" if abs(r["loss_last"] - 10.8249) < 0.02 else ""
        print(f"  {r['name']:10s}  MFU {r['mfu']:5.1f}%   {r['step']:.2f}s/step   "
              f"{r['mem']:.0f}MB   loss~{r['loss_last']:.2f}{spd}{degen}", flush=True)

if __name__ == "__main__":
    main()
