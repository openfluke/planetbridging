"""Shared constants for mixer cross-engine bedrock."""

from pathlib import Path

MIXER_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = MIXER_ROOT / "fixtures"
MODELS_DIR = MIXER_ROOT / "models"
REPORTS_DIR = MIXER_ROOT / "reports"
MANIFEST_PATH = MIXER_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "mixer"
