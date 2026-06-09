"""Shared constants for CNN2 cross-engine bedrock."""

from pathlib import Path

CNN2_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = CNN2_ROOT / "fixtures"
MODELS_DIR = CNN2_ROOT / "models"
REPORTS_DIR = CNN2_ROOT / "reports"
MANIFEST_PATH = CNN2_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "cnn2"
