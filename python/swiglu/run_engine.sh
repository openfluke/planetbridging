#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:-}"
if [[ -z "$ENGINE" ]]; then
  echo "usage: $0 <engine>" >&2
  echo "engines: pytorch tensorflow jax" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/engines/$ENGINE"
ENV_NAME="pb-swiglu-$ENGINE"
HOST="${PLANETBRIDGING_HOST:-http://127.0.0.1:9876}"

if [[ ! -d "$ENGINE_DIR" ]]; then
  echo "unknown engine: $ENGINE" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_FILE="$ENGINE_DIR/environment.yml"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

conda run --no-capture-output -n "$ENV_NAME" python "$ENGINE_DIR/run.py" \
  --host "$HOST" \
  --models-dir "$SCRIPT_DIR/models" \
  --manifest "$SCRIPT_DIR/manifest.yaml"
