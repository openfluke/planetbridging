"""Shared constants for LSTM cross-engine bedrock."""

from pathlib import Path

LSTM_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = LSTM_ROOT / "fixtures"
MODELS_DIR = LSTM_ROOT / "models"
REPORTS_DIR = LSTM_ROOT / "reports"
MANIFEST_PATH = LSTM_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "lstm"
