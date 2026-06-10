"""Shared constants for Residual cross-engine bedrock."""

from pathlib import Path

RESIDUAL_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = RESIDUAL_ROOT / "fixtures"
MODELS_DIR = RESIDUAL_ROOT / "models"
REPORTS_DIR = RESIDUAL_ROOT / "reports"
MANIFEST_PATH = RESIDUAL_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "residual"
