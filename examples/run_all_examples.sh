#!/usr/bin/env bash
# Run every planetbridging example; save stdout+stderr to examples/outputs/*.txt
#
#   ./examples/run_all_examples.sh
#   ./examples/run_all_examples.sh --quick

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/outputs"
COMBINED="$OUT_DIR/run_all.txt"
QUICK=0

if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
  shift
fi

mkdir -p "$OUT_DIR"

PYTHON=""
for cand in "${PYTHON:-}" python python3; do
  [[ -z "$cand" ]] && continue
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy, planetbridging" 2>/dev/null; then
    PYTHON="$cand"
    break
  fi
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import numpy" 2>/dev/null; then
    PYTHON="$cand"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "[run_all_examples] install first:  pip install planetbridging[pytorch]"
  exit 1
fi

ensure_loom_stream() {
  if "$PYTHON" -c "from planetbridging._binary import find_loom_stream; find_loom_stream()" 2>/dev/null; then
    return 0
  fi
  if [[ -f "$REPO_ROOT/go.mod" ]] && [[ -d "$REPO_ROOT/cmd/loom-stream" ]]; then
    echo "[run_all_examples] building loom-stream (dev checkout) …"
    (cd "$REPO_ROOT" && go build -o bin/loom-stream ./cmd/loom-stream/)
    return 0
  fi
  echo "[run_all_examples] loom-stream not found."
  echo "  pip install --force-reinstall planetbridging"
  echo "  or: go build -o bin/loom-stream ./cmd/loom-stream/"
  exit 1
}
ensure_loom_stream

STAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
: >"$COMBINED"

log_combined() {
  printf '%s\n' "$*" >>"$COMBINED"
}

run_example() {
  local outfile_name="$1"
  local script_slug="$2"
  shift 2
  local script="$SCRIPT_DIR/${script_slug}.py"
  local outfile="$OUT_DIR/${outfile_name}.txt"

  if [[ ! -f "$script" ]]; then
    echo "[run_all_examples] skip missing $script"
    return 1
  fi

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  $outfile_name ($script_slug)"
  echo "════════════════════════════════════════════════════════════════"

  {
    echo "planetbridging example: $outfile_name"
    echo "script:   $script_slug.py"
    echo "started:  $STAMP"
    echo "command:  $PYTHON $script $*"
    echo "cwd:      $REPO_ROOT"
    echo "────────────────────────────────────────────────────────────────"
    echo ""
  } >"$outfile"

  log_combined ""
  log_combined "################################################################"
  log_combined "# $outfile_name ($script_slug)"
  log_combined "################################################################"

  set +e
  (
    cd "$REPO_ROOT"
    if [[ -d "$REPO_ROOT/src/planetbridging" ]]; then
      export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    fi
    "$PYTHON" "$script" "$@" 2>&1
  ) | tee -a "$outfile" | tee -a "$COMBINED"
  local status=${PIPESTATUS[0]}
  set -e

  {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "exit: $status"
  } >>"$outfile"

  log_combined "# exit: $status"
  return "$status"
}

FAILURES=()

_run() {
  local outfile_name="$1"
  local script_slug="$2"
  shift 2
  if ! run_example "$outfile_name" "$script_slug" "$@"; then
    FAILURES+=("$outfile_name")
  fi
}

echo "[run_all_examples] writing to $OUT_DIR/"
log_combined "planetbridging — run all examples"
log_combined "started: $STAMP"
log_combined "cwd:     $REPO_ROOT"
log_combined "python:  $($PYTHON --version 2>&1)"
log_combined "quick:   $QUICK"

_run 01_hello_stream_dense 01_hello_stream
_run 01_hello_stream_mha 01_hello_stream mha
_run 02_all_layer_types 02_all_layer_types
_run 03_cross_engine 03_cross_engine layernorm
_run 04_multi_layer_models 04_multi_layer_models
_run 05_welvet_ladder 05_welvet_ladder cnn1 mha
if [[ "$QUICK" -eq 1 ]]; then
  _run 06_showcase_everything 06_showcase_everything --quick
else
  _run 06_showcase_everything 06_showcase_everything
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  Outputs"
echo "────────────────────────────────────────────────────────────────"
echo "  combined:  $COMBINED"
for f in "$OUT_DIR"/*.txt; do
  [[ "$(basename "$f")" == "run_all.txt" ]] && continue
  [[ -f "$f" ]] || continue
  echo "  $(basename "$f")"
done

if ((${#FAILURES[@]} > 0)); then
  echo ""
  echo "[run_all_examples] FAILED: ${FAILURES[*]}"
  log_combined ""
  log_combined "FAILED: ${FAILURES[*]}"
  exit 1
fi

echo ""
echo "[run_all_examples] all examples passed"
log_combined ""
log_combined "ALL PASSED"
