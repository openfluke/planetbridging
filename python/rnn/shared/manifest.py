"""Load RNN manifest and model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input_size: int
    hidden_size: int
    seq_len: int


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
    max_seq_len: int
    max_input_size: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def model_output_dim(model: ModelSpec) -> int:
    return model.seq_len * model.hidden_size


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    engines = tuple(EngineSpec(**e) for e in raw.get("engines", []))
    models = tuple(
        ModelSpec(
            id=m["id"],
            input_size=int(m["input_size"]),
            hidden_size=int(m["hidden_size"]),
            seq_len=int(m["seq_len"]),
        )
        for m in raw.get("models", [])
    )
    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        max_seq_len=int(raw.get("max_seq_len", 8)),
        max_input_size=int(raw.get("max_input_size", 8)),
        output_dim=int(raw.get("output_dim", 64)),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=models,
    )
