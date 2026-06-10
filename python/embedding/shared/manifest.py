"""Load Embedding manifest and model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class ModelSpec:
    id: str
    vocab_size: int
    seq_len: int
    embed_dim: int


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
    max_vocab: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def model_input_dim(model: ModelSpec) -> int:
    return model.seq_len


def model_output_dim(model: ModelSpec) -> int:
    return model.seq_len * model.embed_dim


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    engines = tuple(EngineSpec(**e) for e in raw.get("engines", []))
    models = tuple(
        ModelSpec(
            id=m["id"],
            vocab_size=int(m["vocab_size"]),
            seq_len=int(m["seq_len"]),
            embed_dim=int(m["embed_dim"]),
        )
        for m in raw.get("models", [])
    )
    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        max_seq_len=int(raw.get("max_seq_len", 8)),
        max_vocab=int(raw.get("max_vocab", 64)),
        output_dim=int(raw.get("output_dim", 64)),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=models,
    )
