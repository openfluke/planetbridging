"""Shared constants for RNN cross-engine bedrock."""

from pathlib import Path

RNN_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = RNN_ROOT / "fixtures"
MODELS_DIR = RNN_ROOT / "models"
REPORTS_DIR = RNN_ROOT / "reports"
MANIFEST_PATH = RNN_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "rnn"
