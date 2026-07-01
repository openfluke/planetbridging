#!/usr/bin/env bash
# Cross-compile loom-stream for PyPI desktop targets (welvet-style layout).
#
#   ./scripts/build_loom_stream_all.sh
#   ./scripts/build_loom_stream_all.sh --out bundle/_bin
#   ./scripts/build_loom_stream_all.sh --native-only
#
# Targets (included in the PyPI wheel):
#   linux_amd64/loom-stream
#   macos_arm64/loom-stream
#   windows_amd64/loom-stream.exe
#
# macOS host cross-build deps (install once):
#   brew tap messense/macos-cross-toolchains
#   brew install x86_64-unknown-linux-gnu mingw-w64
#
# Or build each target natively on CI runners and merge with:
#   ./scripts/copy_binaries_to_bundle.sh /path/to/dist

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/bundle/_bin"
NATIVE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="${2:?--out requires path}"
      shift 2
      ;;
    --native-only)
      NATIVE_ONLY=1
      shift
      ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUT"

build_one() {
  local goos="$1" goarch="$2" tag="$3" binary="$4"
  local cc="${5:-}"
  local cgo=1
  local dest="$OUT/$tag/$binary"

  if [[ -n "$cc" ]] && ! command -v "$cc" >/dev/null 2>&1; then
    echo "[build_loom_stream] SKIP $tag — $cc not found"
    return 1
  fi

  echo "[build_loom_stream] $tag ($goos/$goarch) → $dest"
  mkdir -p "$OUT/$tag"
  env GOOS="$goos" GOARCH="$goarch" CGO_ENABLED="$cgo" CC="${cc}" \
    go build -ldflags="-s -w" -o "$dest" ./cmd/loom-stream/
  chmod +x "$dest" 2>/dev/null || true
  ls -lh "$dest"
}

FAIL=0
OK=0

if [[ "$NATIVE_ONLY" -eq 1 ]]; then
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)  build_one darwin arm64 macos_arm64 loom-stream "" && OK=$((OK+1)) || FAIL=$((FAIL+1)) ;;
    Darwin/x86_64) build_one darwin amd64 macos_amd64 loom-stream "" && OK=$((OK+1)) || FAIL=$((FAIL+1)) ;;
    Linux/x86_64)  build_one linux amd64 linux_amd64 loom-stream "" && OK=$((OK+1)) || FAIL=$((FAIL+1)) ;;
    Linux/aarch64) build_one linux arm64 linux_arm64 loom-stream "" && OK=$((OK+1)) || FAIL=$((FAIL+1)) ;;
    MINGW*|MSYS*|CYGWIN*)
      build_one windows amd64 windows_amd64 loom-stream.exe "" && OK=$((OK+1)) || FAIL=$((FAIL+1)) ;;
    *) echo "[build_loom_stream] unknown native platform"; exit 1 ;;
  esac
else
  build_one darwin arm64 macos_arm64 loom-stream "" && OK=$((OK+1)) || FAIL=$((FAIL+1))
  build_one linux amd64 linux_amd64 loom-stream x86_64-linux-gnu-gcc && OK=$((OK+1)) || FAIL=$((FAIL+1))
  build_one windows amd64 windows_amd64 loom-stream.exe x86_64-w64-mingw32-gcc && OK=$((OK+1)) || FAIL=$((FAIL+1))
fi

echo ""
echo "[build_loom_stream] $OK built, $FAIL skipped/failed → $OUT"
if [[ "$OK" -eq 0 ]]; then
  exit 1
fi
