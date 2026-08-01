from __future__ import annotations

import argparse
import hashlib
import json
import platform

from .runtime import build_info, configure_environment, native_paths


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report(probe_jax: bool = False) -> dict:
    paths = native_paths()
    result = {
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "build": build_info(),
        "files": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _fingerprint(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
    }
    if probe_jax:
        configure_environment()
        import jax

        result["jax"] = {
            "version": jax.__version__,
            "devices": [str(device) for device in jax.devices("iree_metal")],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect an iree-metal-preview installation")
    parser.add_argument(
        "--probe-jax", action="store_true", help="compile/load the plugin and enumerate devices"
    )
    args = parser.parse_args()
    result = report(probe_jax=args.probe_jax)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(item["exists"] for item in result["files"].values()):
        raise SystemExit(1)
