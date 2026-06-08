#!/usr/bin/env bash
# Create all per-engine conda environments for dense bedrock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniconda and run: conda init zsh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

SKIP_ENGINES=(paddle)

for env_file in "$SCRIPT_DIR"/engines/*/environment.yml; do
  engine="$(basename "$(dirname "$env_file")")"
  for skip in "${SKIP_ENGINES[@]}"; do
    if [[ "$engine" == "$skip" ]]; then
      echo "[setup_conda] skip: $engine (disabled)"
      continue 2
    fi
  done
  env_name="$(grep '^name:' "$env_file" | awk '{print $2}')"
  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "[setup_conda] exists: $env_name"
  else
    echo "[setup_conda] creating: $env_name"
    if ! conda env create -f "$env_file"; then
      echo "[setup_conda] WARNING: failed to create $env_name (continuing)" >&2
    fi
  fi
done

echo "[setup_conda] done"
