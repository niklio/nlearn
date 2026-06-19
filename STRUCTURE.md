# Repository layout & organizing principles

A map of where things live and the rules that keep it navigable. Read this before
adding a file or moving one — a few hard constraints (below) will bite if ignored.

## Layout

```
nlearn/
├── README.md, CLAUDE.md, STRUCTURE.md   # docs: project narrative, agent notes, this file
├── requirements*.txt                    # deps (.txt = cuda / jaxmetal / base)
│
│   ── core library (flat at root on purpose — see Principle 1) ──
├── model.py            # the GPT-style transformer (layers, RoPE, forward, init)
├── attention.py        # cross-platform attention dispatch (Metal flash kernel binding)
├── gemm_iree.py        # custom Metal simdgroup GEMM binding (custom_vjp)
├── ce_iree.py          # fused cross-entropy kernel binding
├── logging_utils.py    # step timing, MFU, validation, leaderboard posting
├── data.py             # corpus streaming + BPE training pipeline
├── tokenizer.py        # BPE tokenizer (train/encode/load)
├── leaderboard_client.py  # multi-tenant leaderboard HTTP client
├── leaderboard.config.json  # board schema for project `nlearn`
│
│   ── entry points (run as `python <file>.py …`) ──
├── train.py            # training loop
├── generate.py         # sampling/generation
├── cluster.py          # remote Mac-mini job orchestration (SSH + pueue)
│
│   ── environment bootstrap (stay at root — see Principle 4) ──
├── iree_env.sh         # IREE-Metal env for the build host (uses ~/src/iree build)
├── .runenv.sh          # IREE-Metal env for THIS mini (prebuilt .iree_runtime bundle)
│
├── scripts/            # operational shell scripts (run/supervise/monitor/cluster ops)
├── bench/              # kernel & perf benchmarks and probes
├── tools/              # one-off dev/data utilities
│
├── iree_metal/         # the Metal kernels + IREE compiler patches + PJRT plugin (self-contained, documented)
├── leaderboard/        # leaderboard agent-notes + git hooks (pre-commit kernel bench)
│
└── (gitignored runtime artifacts) logs/  wandb/  checkpoints/  datasets/  tokenizer.json  __pycache__/
```

## Principles (the rules that keep this clean)

1. **Core library modules live flat at the repo root.** They import each other by
   bare name (`from model import …`, `from gemm_iree import linear`). Python only
   resolves bare imports against the script's own dir / `sys.path`, so burying them
   in a package breaks every import unless you also rewire it. If you add a module
   that `train.py`/`model.py` import, it goes at root.

2. **A subdir script that imports a root module must add the root to `sys.path`.**
   Files in `bench/` (and anywhere below root) that do `import model` / `import
   leaderboard_client` carry this bootstrap as their first executable line:
   ```python
   import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
   Without it, `python bench/foo.py` puts `bench/` (not root) on the path and the
   import fails.

3. **Operational shell scripts go in `scripts/` and resolve the repo root themselves.**
   Never assume the current directory. Use `cd "$(dirname "$0")/.."` (scripts that
   write `logs/…`, read `.cluster.conf`, etc. depend on cwd being root). Supervisors
   that `cd` to an absolute path are fine as-is.

4. **`iree_env.sh` and `.runenv.sh` stay at root.** ~12 kernel test scripts under
   `iree_metal/kernels/test/` and the leaderboard pre-commit hook `source` them by a
   root-relative path. Moving them breaks all of those silently.

5. **Categorize new non-core files by role:** a benchmark/perf probe → `bench/`; a
   one-off data or dev utility → `tools/`; an ops/run script → `scripts/`; Metal
   kernel or compiler work → `iree_metal/`. Keep root to the library + entry points
   + env bootstrap.

6. **Runtime artifacts are gitignored and written at root-relative paths.** `logs/`,
   `wandb/`, `checkpoints/<run>/`, `datasets/`, `tokenizer.json`. Never commit them;
   don't relocate them (code writes/reads these exact paths).

7. **The kernel pre-commit hook greps staged paths.** `leaderboard/hooks/pre-commit`
   triggers a kernel benchmark when `attention.py`, `gemm_iree.py`, `ce_iree.py`, or
   `iree_metal/kernels/*` change — it expects those at root. If you move a kernel
   binding, update the hook's path patterns and its `$ROOT/bench/bench_kernels.py` call.

## "Where does my new file go?" — quick guide

| It is… | Put it in… |
|---|---|
| imported by `train.py`/`model.py` | repo root (Principle 1) |
| a `python X.py` you run directly | repo root (entry point) |
| a benchmark / perf probe | `bench/` (+ Principle 2 if it imports root) |
| a data prep / dev helper | `tools/` (+ Principle 2 if it imports root) |
| a run/supervise/monitor shell script | `scripts/` (+ Principle 3) |
| a Metal kernel, compiler patch, plugin code | `iree_metal/` |
| a produced log/checkpoint/dataset | leave it gitignored; don't commit |

## Known hazard: two working copies

`~/nlearn` is the **agent working copy** (no `.git`, where runs happen and artifacts
pile up). `~/nlearn-git` is the **canonical git clone**. They drift. Notably some core
modules used by `train.py` — `ce_iree.py`, `data.py`, `tokenizer.py` — are currently
**untracked / missing from the git repo**, and this reorg (the `scripts/ bench/ tools/`
split) was done in `~/nlearn`. To make it canonical, reconcile into `~/nlearn-git`:
add the untracked core modules and mirror the directory moves there.
