"""Build the self-contained macOS arm64 IREE-Metal developer-preview wheel."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_BUILD = (
    Path.home()
    / "src/iree/integrations/pjrt/python_packages/iree_metal_plugin/build/cmake"
)
NATIVE_BUNDLE_DIR = os.environ.get("IREE_METAL_NATIVE_BUNDLE_DIR")
VERSION = os.environ.get("IREE_METAL_PREVIEW_VERSION", "0.1.0.dev0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state(directory: Path) -> dict:
    revision = _git_revision(directory)
    if revision is None:
        return {"revision": None, "dirty": None, "diff_sha256": None}
    try:
        status = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"], text=True
        )
        diff = subprocess.check_output(
            ["git", "-C", str(directory), "diff", "--binary", "HEAD"]
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": revision, "dirty": None, "diff_sha256": None}
    return {
        "revision": revision,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


class BundleBuildPy(build_py):
    """Copy external native build products into build_lib, never the source tree."""

    def run(self):
        super().run()
        native_manifest = None
        if NATIVE_BUNDLE_DIR:
            native_root = Path(NATIVE_BUNDLE_DIR).expanduser().resolve()
            manifest_path = native_root / "native_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"Native bundle manifest is missing: {manifest_path}"
                )
            native_manifest = json.loads(manifest_path.read_text())
            artifacts = {
                "_native/libIREECompiler.dylib": native_root / "libIREECompiler.dylib",
                "_native/pjrt_plugin_iree_metal.dylib": native_root / "pjrt_plugin_iree_metal.dylib",
                "kernels/flash_attention.metal": REPO_ROOT / "iree_metal/kernels/flash_attention.metal",
                "kernels/gemm.metal": REPO_ROOT / "iree_metal/kernels/gemm.metal",
                "kernels/cross_entropy.metal": REPO_ROOT / "iree_metal/kernels/cross_entropy.metal",
                "licenses/IREE_LICENSE.txt": native_root / "IREE_LICENSE.txt",
            }
            iree_root = None
        else:
            build_root = Path(os.environ.get("IREE_METAL_BUILD_DIR", DEFAULT_BUILD)).expanduser()
            iree_root = build_root
            for _ in range(8):
                if (iree_root / ".git").exists():
                    break
                iree_root = iree_root.parent
            artifacts = {
                "_native/libIREECompiler.dylib": build_root / "iree_core/lib/libIREECompiler.dylib",
                "_native/pjrt_plugin_iree_metal.dylib": (
                    build_root
                    / "python/iree/_pjrt_libs/metal/pjrt_plugin_iree_metal.dylib"
                ),
                "kernels/flash_attention.metal": REPO_ROOT / "iree_metal/kernels/flash_attention.metal",
                "kernels/gemm.metal": REPO_ROOT / "iree_metal/kernels/gemm.metal",
                "kernels/cross_entropy.metal": REPO_ROOT / "iree_metal/kernels/cross_entropy.metal",
                "licenses/IREE_LICENSE.txt": iree_root / "LICENSE",
            }
        missing = [str(path) for path in artifacts.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Cannot build the preview wheel; required artifacts are missing:\n  "
                + "\n  ".join(missing)
                + "\nSet IREE_METAL_BUILD_DIR to the patched IREE build/cmake directory."
            )

        package_root = Path(self.build_lib) / "iree_metal_preview"
        for relative, source in artifacts.items():
            destination = package_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        if native_manifest:
            iree_revision = native_manifest.get("iree_revision")
            iree_source_state = {
                "revision": iree_revision,
                "dirty": native_manifest.get("source_dirty"),
                "diff_sha256": native_manifest.get("source_diff_sha256"),
            }
        else:
            iree_revision = _git_revision(iree_root)
            iree_source_state = _git_state(iree_root)

        manifest = {
            "format_version": 1,
            "package_version": VERSION,
            "nlearn_revision": _git_revision(REPO_ROOT),
            "iree_revision": iree_revision,
            "sources": {
                "nlearn": _git_state(REPO_ROOT),
                "iree": iree_source_state,
            },
            "artifacts": {
                relative: {
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
                for relative, source in artifacts.items()
            },
        }
        if native_manifest:
            manifest["native_bundle"] = native_manifest
        (package_root / "build_info.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


class PreviewWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "macosx_13_0_arm64"


setup(
    name="iree-metal-preview",
    version=VERSION,
    description="Developer preview of the IREE PJRT Metal backend for Apple Silicon",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    license="Apache-2.0 WITH LLVM-exception",
    python_requires=">=3.11",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "jax==0.6.1",
        "jaxlib==0.6.1",
        "iree-base-compiler==3.11.0",
        "iree-base-runtime==3.11.0",
    ],
    entry_points={
        "jax_plugins": ["iree-metal = jax_plugins.iree_metal"],
        "console_scripts": ["iree-metal-doctor = iree_metal_preview.doctor:main"],
    },
    cmdclass={"build_py": BundleBuildPy, "bdist_wheel": PreviewWheel},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: MacOS X",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
    ],
)
