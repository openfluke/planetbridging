"""Load CNN1 manifest and model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class CNN1LayerSpec:
    kind: str = "cnn1"
    filters: int = 0
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    activation: str = "linear"
    bias: bool = False


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input_channels: int
    seq_len: int
    layers: tuple[CNN1LayerSpec, ...]


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
    max_seq_len: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def cnn1_out_len(seq_len: int, kernel: int, stride: int, padding: int) -> int:
    return (seq_len + 2 * padding - kernel) // stride + 1


def model_output_dim(model: ModelSpec) -> int:
    seq = model.seq_len
    ch = model.input_channels
    for layer in model.layers:
        seq = cnn1_out_len(seq, layer.kernel_size, layer.stride, layer.padding)
        ch = layer.filters
    return ch * seq


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    engines = tuple(EngineSpec(**e) for e in raw.get("engines", []))
    models: list[ModelSpec] = []
    for m in raw.get("models", []):
        layers = tuple(CNN1LayerSpec(**layer) for layer in m["layers"])
        models.append(
            ModelSpec(
                id=m["id"],
                input_channels=int(m.get("input_channels", raw.get("input_channels", 1))),
                seq_len=int(m["seq_len"]),
                layers=layers,
            )
        )
    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        input_channels=int(raw.get("input_channels", 1)),
        max_seq_len=int(raw.get("max_seq_len", 64)),
        output_dim=int(raw.get("output_dim", 8)),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=tuple(models),
    )
