"""Shared constants for LayerNorm cross-engine bedrock."""

from pathlib import Path

LAYERNORM_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = LAYERNORM_ROOT / "fixtures"
MODELS_DIR = LAYERNORM_ROOT / "models"
REPORTS_DIR = LAYERNORM_ROOT / "reports"
MANIFEST_PATH = LAYERNORM_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "layernorm"
