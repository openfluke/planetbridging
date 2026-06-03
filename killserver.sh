#!/usr/bin/env bash
# Stop the planetbridging compare-host (default :9876).
set -euo pipefail

PORT="${PLANETBRIDGING_PORT:-9876}"
HOST="http://127.0.0.1:${PORT}"
killed=0

stop_port() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 1
  fi
  local pids
  pids="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
  [[ -z "$pids" ]] && return 1

  echo "[killserver] stopping process(es) on port ${PORT}: ${pids}"
  kill ${pids} 2>/dev/null || true
  sleep 0.3
  pids="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -9 ${pids} 2>/dev/null || true
  fi
  return 0
}

if stop_port; then
  killed=1
fi

if curl -fsS "${HOST}/health" >/dev/null 2>&1; then
  echo "[killserver] host still up; trying pkill"
  pkill -f "planetbridging" 2>/dev/null || true
  pkill -f "go run \\." 2>/dev/null || true
  sleep 0.3
  killed=1
fi

if curl -fsS "${HOST}/health" >/dev/null 2>&1; then
  echo "[killserver] failed to stop compare-host on port ${PORT}" >&2
  exit 1
fi

if [[ "$killed" -eq 1 ]]; then
  echo "[killserver] compare-host stopped"
else
  echo "[killserver] no compare-host found on port ${PORT}"
fi
