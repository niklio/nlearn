from __future__ import annotations

import json
import os
import platform
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def native_paths() -> dict[str, Path]:
    return {
        "plugin": PACKAGE_ROOT / "_native" / "pjrt_plugin_iree_metal.dylib",
        "compiler": PACKAGE_ROOT / "_native" / "libIREECompiler.dylib",
        "flash_kernel": PACKAGE_ROOT / "kernels" / "flash_attention.metal",
        "gemm_kernel": PACKAGE_ROOT / "kernels" / "gemm.metal",
        "ce_kernel": PACKAGE_ROOT / "kernels" / "cross_entropy.metal",
    }


def build_info() -> dict:
    path = PACKAGE_ROOT / "build_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _iree_lld() -> Path:
    import iree.compiler

    package = Path(iree.compiler.__file__).resolve().parent
    candidates = [
        package / "_mlir_libs" / "iree-lld",
        package / "_mlir_libs" / "iree-lld.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError("iree-lld was not found in the installed iree-base-compiler package")


def configure_environment() -> dict[str, Path]:
    """Point IREE/JAX at wheel-bundled native libraries and custom kernels."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("iree-metal-preview supports macOS arm64 only")
    paths = native_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("IREE-Metal wheel is incomplete; missing: " + ", ".join(missing))

    os.environ.setdefault("IREE_PJRT_COMPILER_LIB_PATH", str(paths["compiler"]))
    # Current compiler patches use the generic IREE_METAL_* names. Keep the
    # original NLEARN_* aliases for wheels built from earlier patch revisions.
    kernel_env = {
        "FLASH": paths["flash_kernel"],
        "GEMM": paths["gemm_kernel"],
        "CE": paths["ce_kernel"],
    }
    for name, path in kernel_env.items():
        os.environ.setdefault(f"IREE_METAL_{name}_KERNEL_PATH", str(path))
        os.environ.setdefault(f"NLEARN_{name}_KERNEL_PATH", str(path))
    os.environ.setdefault(
        "IREE_PJRT_IREE_COMPILER_OPTIONS",
        f"--iree-llvmcpu-embedded-linker-path={_iree_lld()} "
        "--iree-metal-compile-to-metallib=false",
    )
    return paths
