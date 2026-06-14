# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Standalone IREE-Metal PJRT plugin loader for the cluster (no editable build).

JAX auto-discovers this via the `jax_plugins` namespace package when its parent
directory is on PYTHONPATH (JAX's path-based plugin scanning). Unlike the
editable plugin used on the build host, this finds the prebuilt dylibs by path
from the environment, so the Mac mini needs no IREE source tree or CMake build:

  NLEARN_IREE_PLUGIN_DYLIB    -> prebuilt pjrt_plugin_iree_metal.dylib
  IREE_PJRT_COMPILER_LIB_PATH -> prebuilt (patched) libIREECompiler.dylib

cluster.py ships those dylibs and sets these vars in the job environment.
"""

import logging
import os

import jax._src.xla_bridge as xb

logger = logging.getLogger(__name__)


def initialize():
    dylib = os.environ.get("NLEARN_IREE_PLUGIN_DYLIB")
    if not dylib or not os.path.exists(dylib):
        logger.warning(
            "iree_metal plugin: NLEARN_IREE_PLUGIN_DYLIB not set or missing "
            "(%s); skipping registration.", dylib)
        return
    options = {}
    compiler = os.environ.get("IREE_PJRT_COMPILER_LIB_PATH")
    if compiler:
        options["COMPILER_LIB_PATH"] = compiler
    xb.register_plugin(
        "iree_metal", priority=500, library_path=dylib, options=options)
