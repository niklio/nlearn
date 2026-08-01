#!/bin/bash
set -euo pipefail

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "The IREE Metal native bundle must be built on Apple Silicon." >&2
  exit 2
fi

IREE_SOURCE_DIR="${1:?usage: $0 IREE_SOURCE_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$PWD/native-dist}"
IREE_SOURCE_DIR="$(cd "$IREE_SOURCE_DIR" && pwd)"
IREE_BUILD_DIR="${IREE_METAL_BUILD_DIR:-${IREE_SOURCE_DIR}/integrations/pjrt/python_packages/iree_metal_plugin/build/cmake}"

COMPILER_DYLIB="${IREE_BUILD_DIR}/iree_core/lib/libIREECompiler.dylib"
PLUGIN_DYLIB="${IREE_BUILD_DIR}/python/iree/_pjrt_libs/metal/pjrt_plugin_iree_metal.dylib"
LICENSE_FILE="${IREE_SOURCE_DIR}/LICENSE"

for required in "$COMPILER_DYLIB" "$PLUGIN_DYLIB" "$LICENSE_FILE"; do
  if [ ! -f "$required" ]; then
    echo "Missing required native artifact: $required" >&2
    exit 2
  fi
done

IREE_REVISION="$(git -C "$IREE_SOURCE_DIR" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$IREE_SOURCE_DIR" status --porcelain --untracked-files=normal)"
if [ -n "$SOURCE_STATUS" ] && [ "${IREE_METAL_ALLOW_DIRTY_SOURCE:-0}" != "1" ]; then
  echo "Refusing to package a dirty IREE checkout:" >&2
  printf '%s\n' "$SOURCE_STATUS" >&2
  echo "Set IREE_METAL_ALLOW_DIRTY_SOURCE=1 only for local testing." >&2
  exit 2
fi

for dylib in "$COMPILER_DYLIB" "$PLUGIN_DYLIB"; do
  if ! file "$dylib" | grep -q 'arm64'; then
    echo "Native artifact is not arm64: $dylib" >&2
    file "$dylib" >&2
    exit 2
  fi
  codesign --verify --verbose=2 "$dylib"
done

mkdir -p "$OUTPUT_DIR"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/iree-metal-native.XXXXXX")"
cleanup() {
  case "$STAGING_DIR" in
    "${TMPDIR:-/tmp}"/iree-metal-native.*) rm -rf "$STAGING_DIR" ;;
  esac
}
trap cleanup EXIT

BUNDLE_ROOT="${STAGING_DIR}/iree-metal-native"
mkdir -p "$BUNDLE_ROOT"
cp "$COMPILER_DYLIB" "${BUNDLE_ROOT}/libIREECompiler.dylib"
cp "$PLUGIN_DYLIB" "${BUNDLE_ROOT}/pjrt_plugin_iree_metal.dylib"
cp "$LICENSE_FILE" "${BUNDLE_ROOT}/IREE_LICENSE.txt"

export BUNDLE_ROOT IREE_REVISION IREE_SOURCE_DIR
python3 - <<'PY'
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

root = Path(os.environ["BUNDLE_ROOT"])
source = Path(os.environ["IREE_SOURCE_DIR"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

status = subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"],
    text=True,
)
diff = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
try:
    xcode = subprocess.check_output(["xcodebuild", "-version"], text=True).strip()
except (OSError, subprocess.CalledProcessError):
    xcode = None

artifact_names = [
    "libIREECompiler.dylib",
    "pjrt_plugin_iree_metal.dylib",
    "IREE_LICENSE.txt",
]
manifest = {
    "format_version": 1,
    "iree_repository": "https://github.com/niklio/iree-metal",
    "iree_revision": os.environ["IREE_REVISION"],
    "source_dirty": bool(status),
    "source_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "xcode": xcode,
    "artifacts": {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
        for name in artifact_names
    },
}
(root / "native_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

ARCHIVE="${OUTPUT_DIR}/iree-metal-native-${IREE_REVISION}-macos-arm64.tar.gz"
COPYFILE_DISABLE=1 tar -C "$STAGING_DIR" -czf "$ARCHIVE" iree-metal-native
(cd "$OUTPUT_DIR" && shasum -a 256 "$(basename "$ARCHIVE")") > "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE"
cat "${ARCHIVE}.sha256"
