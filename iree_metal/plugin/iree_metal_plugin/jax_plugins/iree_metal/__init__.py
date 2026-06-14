# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import platform
import sys

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)


def probe_iree_compiler_dylib() -> str:
    """Locates the IREE compiler dylib.

    Honors the IREE_PJRT_COMPILER_LIB_PATH env var when set so a locally-built
    (e.g. patched) libIREECompiler can be used instead of the one bundled with
    the installed iree.compiler wheel. Falls back to probing the installed
    package otherwise.
    """
    import os

    override = os.environ.get("IREE_PJRT_COMPILER_LIB_PATH")
    if override:
        return override

    # TODO: Move this out of the ctypes API initialization.
    from iree.compiler.api import ctypes_dl

    return ctypes_dl._probe_iree_compiler_dylib()


def _find_native_library() -> Path:
    """Locates the PJRT plugin shared library.

    CMake emits the shared library with a platform-dependent suffix: ".so" on
    Linux and (depending on the target type) ".dylib" or ".so" on macOS. Probe
    the candidates rather than hard-coding a single suffix so the plugin loads
    regardless of how the dylib was named at build time.
    """
    import iree._pjrt_libs.metal as lib_package

    base = Path(lib_package.__file__).resolve().parent
    stem = "pjrt_plugin_iree_metal"
    # On Darwin prefer .dylib but fall back to .so (and vice-versa elsewhere).
    suffixes = (".dylib", ".so") if platform.system() == "Darwin" else (".so", ".dylib")
    for suffix in suffixes:
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    # Return the conventional path for the platform so the warning below points
    # at the expected location.
    return base / f"{stem}{suffixes[0]}"


def initialize():
    path = _find_native_library()
    if not path.exists():
        logger.warning(
            f"WARNING: Native library {path} does not exist. "
            f"This most likely indicates an issue with how {__package__} "
            f"was built or installed."
        )
    xb.register_plugin(
        "iree_metal",
        priority=500,
        library_path=str(path),
        options={
            "COMPILER_LIB_PATH": str(probe_iree_compiler_dylib()),
        },
    )
