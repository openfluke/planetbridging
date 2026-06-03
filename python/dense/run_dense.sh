#!/usr/bin/env bash
# Train/infer dense models across Python AI engines and compare 100-sample outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST="${PLANETBRIDGING_HOST:-http://127.0.0.1:9876}"
HOST_PID=""

cleanup() {
  if [[ -n "$HOST_PID" ]] && kill -0 "$HOST_PID" 2>/dev/null; then
    kill "$HOST_PID" 2>/dev/null || true
    wait "$HOST_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_host_if_needed() {
  if curl -fsS "$HOST/health" >/dev/null 2>&1; then
    if curl -fsS "$HOST/api/v1/reports" >/dev/null 2>&1; then
      echo "[run_dense] compare-host already running at $HOST"
      return
    fi
    echo "[run_dense] compare-host unhealthy; restarting"
    pkill -f "planetbridging" 2>/dev/null || pkill -f "go run \." 2>/dev/null || true
    sleep 1
  fi

  echo "[run_dense] starting compare-host at $HOST"
  (
    cd "$REPO_ROOT"
    go run . -addr ":9876"
  ) &
  HOST_PID=$!

  for _ in $(seq 1 30); do
    if curl -fsS "$HOST/health" >/dev/null 2>&1; then
      echo "[run_dense] compare-host ready"
      return
    fi
    sleep 0.5
  done

  echo "[run_dense] compare-host failed to start" >&2
  exit 1
}

# Order matters: pytorch exports ONNX for onnxruntime.
ENGINES=(
  pytorch
  tensorflow
  jax
  sklearn
  paddle
)

start_host_if_needed

FAILURES=0
for engine in "${ENGINES[@]}"; do
  if ! "$SCRIPT_DIR/run_engine.sh" "$engine"; then
    echo "[run_dense] engine failed: $engine (continuing)" >&2
    FAILURES=$((FAILURES + 1))
  fi
done

echo
echo "========== comparison summary =========="
curl -fsS "$HOST/api/v1/compare.txt" || true
echo

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[run_dense] completed with $FAILURES engine failure(s)" >&2
  exit 1
fi

echo "[run_dense] all engines completed"
