#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOST="${PLANETBRIDGING_HOST:-http://127.0.0.1:9876}"

ENGINES=(pytorch tensorflow jax)

if ! curl -fsS "$HOST/health" >/dev/null 2>&1; then
  echo "[run_cnn2] starting compare-host"
  (cd "$REPO_ROOT" && go run .) &
  for _ in $(seq 1 30); do
    curl -fsS "$HOST/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

FAILURES=0
for engine in "${ENGINES[@]}"; do
  if ! "$SCRIPT_DIR/run_engine.sh" "$engine"; then
    FAILURES=$((FAILURES + 1))
  fi
done

echo
curl -fsS "$HOST/api/v1/compare.txt?bedrock=cnn2" || true
echo
exit "$FAILURES"
