# Repository layout & organizing principles

A map of where things live and the rules that keep it navigable. Read this before
adding or moving a file.

## Layout

```
nlearn/                          # repo root
├── nlearn/                      # ── the core library (a Python package) ──
│   ├── __init__.py
│   ├── model.py                 # the GPT-style transformer (layers, RoPE, forward, init)
│   ├── attention.py             # cross-platform attention dispatch (Metal flash binding)
│   ├── train.py                 # training loop        →  python -m nlearn.train
│   ├── generate.py              # sampling/generation   →  python -m nlearn.generate
│   ├── logging_utils.py         # step timing, MFU, validation, leaderboard posting
│   ├── leaderboard.py           # multi-tenant leaderboard HTTP client
│   ├── kernels/                 # IREE-Metal kernel bindings (python side of iree_metal/kernels/)
│   │   ├── gemm.py              #   custom simdgroup GEMM (custom_vjp)
│   │   └── cross_entropy.py     #   fused softmax cross-entropy (loss + dlogits)
│   └── data/
│       ├── streaming.py         # streaming corpus loader + BPE training pipeline
│       └── tokenizer.py         # BPE tokenizer (train / encode / load)
│
├── cluster.py                   # remote Mac-mini job orchestration (SSH + pueue) — ops entry point
├── leaderboard.config.json      # board schema for project `nlearn`
├── requirements*.txt            # deps (.txt = base / cuda / jaxmetal)
├── README.md, CLAUDE.md, STRUCTURE.md
│
├── iree_env.sh, .runenv.sh      # IREE-Metal env bootstrap (build host / this mini) — stay at root
│
├── scripts/                     # operational shell scripts (run/supervise/monitor/cluster ops)
├── bench/                       # kernel & perf benchmarks and probes
├── tools/                       # one-off dev/data utilities
│
├── iree_metal/                  # the Metal kernels + IREE compiler patches + PJRT plugin (self-contained)
├── leaderboard/                 # leaderboard agent-notes + git hooks (pre-commit kernel bench)
│
└── (gitignored runtime artifacts) logs/  wandb/  checkpoints/  datasets/  tokenizer.json  .iree_runtime/  __pycache__/
```

## Principles (the rules that keep this clean)

1. **The core library is the `nlearn/` package.** Import from it explicitly:
   `from nlearn.model import …`, `from nlearn.kernels.gemm import linear`,
   `from nlearn.data.tokenizer import …`. New library code goes inside `nlearn/`
   (pick the right subpackage: `kernels/` for Metal-kernel bindings, `data/` for the
   data pipeline, top level for model/training/infra).

2. **Entry points run as modules:** `python -m nlearn.train …`, `python -m nlearn.generate …`.
   `cluster.py` builds these commands; the supervisors in `scripts/` call them. There
   are no `python train.py` scripts at root anymore.

3. **`import nlearn` resolves because the repo root is on `PYTHONPATH`.** `.runenv.sh`
   sets it; `python -m …` run from the repo root adds it automatically. Scripts in
   subdirs that import the package (e.g. `bench/`) carry a one-line bootstrap as their
   first executable line:
   ```python
   import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

4. **Operational shell scripts go in `scripts/` and resolve the repo root themselves.**
   Never assume the current directory — use `cd "$(dirname "$0")/.."`.

5. **`iree_env.sh` and `.runenv.sh` stay at root.** Kernel test scripts under
   `iree_metal/kernels/test/` and the leaderboard pre-commit hook `source` them by a
   root-relative path.

6. **Categorize new non-library files by role:** a benchmark/perf probe → `bench/`; a
   one-off data or dev utility → `tools/`; an ops/run script → `scripts/`; Metal kernel
   or compiler work → `iree_metal/`.

7. **Runtime artifacts are gitignored and written at root-relative paths.** `logs/`,
   `wandb/`, `checkpoints/<run>/`, `datasets/`, `tokenizer.json`, `.iree_runtime/`.
   Never commit them; don't relocate them.

8. **The kernel pre-commit hook greps staged paths.** `leaderboard/hooks/pre-commit`
   triggers a kernel benchmark when `nlearn/attention.py`, `nlearn/kernels/gemm.py`,
   `nlearn/kernels/cross_entropy.py`, or `iree_metal/kernels/*` change. If you move a
   kernel binding, update the hook's path patterns and its `bench/bench_kernels.py` call.

## "Where does my new file go?" — quick guide

| It is… | Put it in… |
|---|---|
| model / training / kernel-binding / data library code | inside `nlearn/` (right subpackage) |
| a new runnable command | `nlearn/<name>.py` with a `main()`, run via `python -m nlearn.<name>` |
| a benchmark / perf probe | `bench/` (+ the `sys.path` bootstrap) |
| a data prep / dev helper | `tools/` (+ bootstrap if it imports `nlearn`) |
| a run/supervise/monitor shell script | `scripts/` (resolve root via `dirname/..`) |
| a Metal kernel, compiler patch, plugin code | `iree_metal/` |
| a produced log/checkpoint/dataset | leave it gitignored; don't commit |

## Single working tree

`~/nlearn` is the one and only working tree **and** git repo (remote
`github.com/niklio/nlearn`, branch `main`). There is no second clone to keep in sync.
