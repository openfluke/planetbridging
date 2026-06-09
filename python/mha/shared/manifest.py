"""Load MHA manifest and model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class ModelSpec:
    id: str
    d_model: int
    num_heads: int
    seq_len: int
    num_kv_heads: int = 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def num_kv(self) -> int:
        return self.num_kv_heads or self.num_heads


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
    max_d_model: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def model_output_dim(model: ModelSpec) -> int:
    return model.seq_len * model.d_model


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    engines = tuple(EngineSpec(**e) for e in raw.get("engines", []))
    models = tuple(
        ModelSpec(
            id=m["id"],
            d_model=int(m["d_model"]),
            num_heads=int(m["num_heads"]),
            seq_len=int(m["seq_len"]),
            num_kv_heads=int(m.get("num_kv_heads", 0)),
        )
        for m in raw.get("models", [])
    )
    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        max_seq_len=int(raw.get("max_seq_len", 8)),
        max_d_model=int(raw.get("max_d_model", 16)),
        output_dim=int(raw.get("output_dim", 32)),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=models,
    )
