"""Shared constants for dense cross-engine bedrock."""

from pathlib import Path

DENSE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = DENSE_ROOT / "fixtures"
MODELS_DIR = DENSE_ROOT / "models"
REPORTS_DIR = DENSE_ROOT / "reports"
MANIFEST_PATH = DENSE_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
