#!/usr/bin/env bash
# Run one dense engine inside its conda env and POST 100-sample outputs to compare-host.
set -euo pipefail

ENGINE="${1:-}"
if [[ -z "$ENGINE" ]]; then
  echo "usage: $0 <engine>" >&2
  echo "engines: pytorch tensorflow jax sklearn paddle onnxruntime" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DENSE_ROOT="$SCRIPT_DIR"
ENGINE_DIR="$DENSE_ROOT/engines/$ENGINE"
ENV_NAME="pb-dense-$ENGINE"
HOST="${PLANETBRIDGING_HOST:-http://127.0.0.1:9876}"

if [[ ! -d "$ENGINE_DIR" ]]; then
  echo "unknown engine: $ENGINE" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniconda and run: conda init zsh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_FILE="$ENGINE_DIR/environment.yml"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[run_engine] creating conda env $ENV_NAME"
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

echo "[run_engine] running $ENGINE in $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" python "$ENGINE_DIR/run.py" \
  --host "$HOST" \
  --models-dir "$DENSE_ROOT/models" \
  --manifest "$DENSE_ROOT/manifest.yaml"
