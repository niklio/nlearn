#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "${BASH_SOURCE[0]%/*}/.." && pwd)"
IREE_REVISION="${IREE_METAL_IREE_REVISION:-$(tr -d '[:space:]' < "${REPO_ROOT}/iree_metal/ci/iree-ref.txt")}"
DESTINATION="${1:-${RUNNER_TEMP:-/tmp}/iree-metal-native-download}"
RELEASE_TAG="iree-metal-native-${IREE_REVISION:0:12}"
ARCHIVE_NAME="iree-metal-native-${IREE_REVISION}-macos-arm64.tar.gz"

mkdir -p "$DESTINATION"
gh release download "$RELEASE_TAG" \
  --repo "${GITHUB_REPOSITORY:-niklio/nlearn}" \
  --pattern "$ARCHIVE_NAME" \
  --pattern "${ARCHIVE_NAME}.sha256" \
  --dir "$DESTINATION"

(cd "$DESTINATION" && shasum -a 256 -c "${ARCHIVE_NAME}.sha256") >&2
tar -C "$DESTINATION" -xzf "${DESTINATION}/${ARCHIVE_NAME}"

MANIFEST="${DESTINATION}/iree-metal-native/native_manifest.json"
python3 - "$MANIFEST" "$IREE_REVISION" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if manifest.get("iree_revision") != sys.argv[2]:
    raise SystemExit(
        f"native bundle revision mismatch: {manifest.get('iree_revision')} != {sys.argv[2]}"
    )
if manifest.get("source_dirty"):
    raise SystemExit("refusing a native bundle built from dirty sources")
PY

printf '%s\n' "${DESTINATION}/iree-metal-native"
