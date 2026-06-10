"""Shared constants for RMSNorm cross-engine bedrock."""

from pathlib import Path

RMSNORM_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = RMSNORM_ROOT / "fixtures"
MODELS_DIR = RMSNORM_ROOT / "models"
REPORTS_DIR = RMSNORM_ROOT / "reports"
MANIFEST_PATH = RMSNORM_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "rmsnorm"
