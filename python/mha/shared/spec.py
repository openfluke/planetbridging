"""Shared constants for MHA cross-engine bedrock."""

from pathlib import Path

MHA_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = MHA_ROOT / "fixtures"
MODELS_DIR = MHA_ROOT / "models"
REPORTS_DIR = MHA_ROOT / "reports"
MANIFEST_PATH = MHA_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "mha"
ROPE_THETA = 10000.0
