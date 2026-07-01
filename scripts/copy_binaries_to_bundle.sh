#!/usr/bin/env bash
# Copy prebuilt loom-stream dist/ trees into bundle/_bin/ (CI merge step).
#
#   ./scripts/copy_binaries_to_bundle.sh
#   ./scripts/copy_binaries_to_bundle.sh /path/to/dist
#
# Expected layout (same as welvet C-ABI dist/):
#   dist/linux_amd64/loom-stream
#   dist/macos_arm64/loom-stream
#   dist/windows_amd64/loom-stream.exe

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/dist/loom-stream}"
DST="$ROOT/bundle/_bin"

if [[ ! -d "$SRC" ]]; then
  echo "❌ Source not found: $SRC"
  echo ""
  echo "Build first:"
  echo "  ./scripts/build_loom_stream_all.sh"
  echo "  ./scripts/build_loom_stream_all.sh --out dist/loom-stream"
  exit 1
fi

if ! compgen -G "$SRC"/* >/dev/null 2>&1; then
  echo "❌ No platform folders under $SRC"
  exit 1
fi

echo "Copying loom-stream binaries → bundle"
echo "  from: $SRC"
echo "  to:   $DST"
mkdir -p "$DST"
cp -Rv "$SRC"/* "$DST"/

echo ""
echo "✓ Installed platforms:"
for d in linux_amd64 macos_arm64 macos_amd64 windows_amd64; do
  if [[ -d "$DST/$d" ]]; then
    ls -lh "$DST/$d"/loom-stream* 2>/dev/null || true
  fi
done
