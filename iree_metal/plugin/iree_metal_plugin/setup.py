#!/usr/bin/python3

# Copyright 2024 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Early splice the _setup_support directory onto the python path.
import os
from pathlib import Path
import sys

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "_setup_support"))

import iree_pjrt_setup
from setuptools import setup, find_namespace_packages

README = r"""
OpenXLA PJRT Plugin for Metal (Apple Silicon)
"""

# Setup and get version information.
CMAKE_BUILD_DIR_ABS = os.path.join(THIS_DIR, "build", "cmake")


class CMakeBuildPy(iree_pjrt_setup.BaseCMakeBuildPy):
    def build_default_configuration(self):
        print("*****************************", file=sys.stderr)
        print("* Building base runtime     *", file=sys.stderr)
        print("*****************************", file=sys.stderr)
        self.build_configuration(
            os.path.join(THIS_DIR, "build", "cmake"),
            extra_cmake_args=(
                "-DIREE_HAL_DRIVER_METAL=ON",
                # The PJRT plugin requires the in-tree compiler bindings
                # (IREELLVMIncludeSetup et al.), so IREE_BUILD_COMPILER must be ON
                # -- runtime-only is unsupported. Limit parallel link jobs so the
                # LLVM/MLIR link steps don't exhaust RAM on a 16GB machine.
                "-DIREE_BUILD_COMPILER=ON",
                "-DLLVM_PARALLEL_LINK_JOBS=1",
            ),
        )
        print("Target populated.", file=sys.stderr)


iree_pjrt_setup.populate_built_package(
    os.path.join(
        CMAKE_BUILD_DIR_ABS,
        "python",
        "iree",
        "_pjrt_libs",
        "metal",
    )
)


setup(
    name=f"iree-pjrt-plugin-metal{iree_pjrt_setup.PACKAGE_SUFFIX}",
    version=f"{iree_pjrt_setup.PACKAGE_VERSION}",
    author="The IREE Team",
    author_email="iree-technical-discussion@lists.lfaidata.foundation",
    license="Apache-2.0",
    description="IREE PJRT Plugin for Metal (Apple Silicon)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/iree-org/iree",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=[
        "jax_plugins.iree_metal",
        "iree._pjrt_libs.metal",
    ],
    package_dir={
        "jax_plugins.iree_metal": "jax_plugins/iree_metal",
        "iree._pjrt_libs.metal": "build/cmake/python/iree/_pjrt_libs/metal",
    },
    package_data={
        "iree._pjrt_libs.metal": ["pjrt_plugin_iree_metal.*"],
    },
    cmdclass={
        "build": iree_pjrt_setup.PjrtPluginBuild,
        "build_py": CMakeBuildPy,
        "bdist_wheel": iree_pjrt_setup.bdist_wheel,
        "install": iree_pjrt_setup.platlib_install,
    },
    zip_safe=False,  # Needs to reference embedded shared libraries.
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            "iree-metal = jax_plugins.iree_metal",
        ],
    },
    install_requires=iree_pjrt_setup.install_requires,
)
