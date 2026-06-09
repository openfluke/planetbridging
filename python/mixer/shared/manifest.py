"""Load mixer manifest and model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .mixer_spec import OUTPUT_DIM, VOLUME_C, VOLUME_D, VOLUME_H, VOLUME_W
from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class ModelSpec:
    id: str


@dataclass(frozen=True)
class EngineSpec:
    id: str
    enabled: bool = True


@dataclass(frozen=True)
class Manifest:
    fixture_version: str
    seed: int
    train_samples: int
    test_samples: int
    input_channels: int
    max_spatial: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def model_output_dim(_model: ModelSpec | None = None) -> int:
    return OUTPUT_DIM


def model_input_dim(_model: ModelSpec | None = None) -> int:
    return VOLUME_C * VOLUME_D * VOLUME_H * VOLUME_W


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    engines = tuple(EngineSpec(**e) for e in raw.get("engines", []))
    models = tuple(ModelSpec(id=m["id"]) for m in raw.get("models", []))
    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        input_channels=int(raw.get("input_channels", 1)),
        max_spatial=int(raw.get("max_spatial", 2)),
        output_dim=int(raw.get("output_dim", OUTPUT_DIM)),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=models,
    )
