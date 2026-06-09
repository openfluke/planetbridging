"""Shared constants for CNN3 cross-engine bedrock."""

from pathlib import Path

CNN3_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = CNN3_ROOT / "fixtures"
MODELS_DIR = CNN3_ROOT / "models"
REPORTS_DIR = CNN3_ROOT / "reports"
MANIFEST_PATH = CNN3_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "cnn3"
