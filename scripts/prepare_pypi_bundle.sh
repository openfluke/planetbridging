#!/usr/bin/env bash
# Bundle python bedrock POC tree + multi-platform loom-stream for the PyPI wheel.
#
#   ./scripts/prepare_pypi_bundle.sh
#   ./scripts/prepare_pypi_bundle.sh --native-only   # current platform only (fast)
#
# Populates:
#   bundle/_data/python/
#   bundle/_bin/linux_amd64|macos_arm64|windows_amd64/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PKG_DIR="$ROOT/bundle"
DATA_DIR="$PKG_DIR/_data"
BIN_DIR="$PKG_DIR/_bin"
NATIVE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --native-only) NATIVE_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

pick_python() {
  for cmd in python python3; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import numpy" 2>/dev/null; then
      echo "$cmd"
      return 0
    fi
  done
  return 1
}

PYTHON="$(pick_python)" || { echo "numpy/python required"; exit 1; }

echo "[prepare_pypi_bundle] ensuring dev-checkout fixtures (optional) …"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/ensure_bedrock_fixtures.py" 2>/dev/null || true

echo "[prepare_pypi_bundle] staging python bedrocks → bundle/_data/python/ (no npz — generated on first use)"
rm -rf "$DATA_DIR/python"
mkdir -p "$DATA_DIR"
rsync -a \
  --exclude='models/' \
  --exclude='reports/' \
  --exclude='fixtures/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "$ROOT/python/" "$DATA_DIR/python/"

if [[ "$NATIVE_ONLY" -eq 1 ]]; then
  "$ROOT/scripts/build_loom_stream_all.sh" --out "$BIN_DIR" --native-only
else
  "$ROOT/scripts/build_loom_stream_all.sh" --out "$BIN_DIR"
fi

SIZE_DATA="$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')"
SIZE_BIN="$(du -sh "$BIN_DIR" 2>/dev/null | awk '{print $1}')"
PLATFORMS="$(find "$BIN_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | tr '\n' ' ')"
echo "[prepare_pypi_bundle] done — data=$SIZE_DATA  binaries=$SIZE_BIN  platforms: $PLATFORMS"
