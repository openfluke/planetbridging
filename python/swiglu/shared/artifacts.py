from __future__ import annotations

import json
from pathlib import Path

from .spec import MODELS_DIR


def model_dir(engine: str, model_id: str) -> Path:
    return MODELS_DIR / engine / model_id


def is_complete(engine: str, model_id: str, required: list[str]) -> bool:
    base = model_dir(engine, model_id)
    if not (base / ".complete").exists():
        return False
    return all((base / name).exists() for name in required)


def write_complete(engine: str, model_id: str, artifacts: list[str], framework_version: str) -> None:
    base = model_dir(engine, model_id)
    base.mkdir(parents=True, exist_ok=True)
    meta = {
        "engine": engine,
        "model_id": model_id,
        "framework_version": framework_version,
        "artifacts": artifacts,
        "bedrock": "swiglu",
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2))
    (base / ".complete").write_text("ok\n")
