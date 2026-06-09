"""Shared constants for CNN1 cross-engine bedrock."""

from pathlib import Path

CNN1_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = CNN1_ROOT / "fixtures"
MODELS_DIR = CNN1_ROOT / "models"
REPORTS_DIR = CNN1_ROOT / "reports"
MANIFEST_PATH = CNN1_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "cnn1"
