"""Shared constants for Embedding cross-engine bedrock."""

from pathlib import Path

EMBEDDING_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = EMBEDDING_ROOT / "fixtures"
MODELS_DIR = EMBEDDING_ROOT / "models"
REPORTS_DIR = EMBEDDING_ROOT / "reports"
MANIFEST_PATH = EMBEDDING_ROOT / "manifest.yaml"

DEFAULT_HOST = "http://127.0.0.1:9876"
BEDROCK = "embedding"
