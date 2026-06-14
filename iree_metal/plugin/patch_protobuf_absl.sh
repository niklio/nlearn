#!/bin/sh
# Patches the abseil bundled with protobuf so it builds with Apple clang 17.
#
# abseil's randen copts unconditionally emit universal-binary flags
# (`-Xarch_x86_64 -msse4.1 -Xarch_arm64 ...`) and rely on
# `-Wno-unused-command-line-argument` to ignore the inactive arch. Apple clang
# 17 treats the inactive `-msse4.1` on an arm64-only target as a hard error
# rather than a suppressible warning, breaking the build. Iterate the actual
# target architectures (CMAKE_OSX_ARCHITECTURES) instead.
#
# Run with CWD = protobuf source dir (FetchContent's PATCH_COMMAND default).
# Idempotent: the sed simply finds nothing to replace on a second run.
set -e
f="third_party/abseil-cpp/absl/copts/AbseilConfigureCopts.cmake"
if [ -f "$f" ]; then
  sed -i '' 's/foreach(_arch IN ITEMS "x86_64" "arm64")/foreach(_arch IN LISTS CMAKE_OSX_ARCHITECTURES)/' "$f"
fi
