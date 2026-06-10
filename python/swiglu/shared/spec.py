"""Shared constants for SwiGLU cross-engine bedrock."""

from pathlib import Path

SWIGLU_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = SWIGLU_ROOT / "fixtures"
MODELS_DIR = SWIGLU_ROOT / "models"
REPORTS_DIR = SWIGLU_ROOT / "reports"
MANIFEST_PATH = SWIGLU_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "swiglu"
