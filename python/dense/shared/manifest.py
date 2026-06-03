"""Load manifest and expose model/engine definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .spec import MANIFEST_PATH


@dataclass(frozen=True)
class LayerSpec:
    units: int
    activation: str = "linear"
    bias: bool | None = None


@dataclass(frozen=True)
class ModelSpec:
    id: str
    input_dim: int
    layers: tuple[LayerSpec, ...]
    bias: bool = True


@dataclass(frozen=True)
class EngineSpec:
    id: str
    enabled: bool = True
    inference_only: bool = False


@dataclass(frozen=True)
class Manifest:
    fixture_version: str
    seed: int
    train_samples: int
    test_samples: int
    max_input_dim: int
    output_dim: int
    epochs: int
    batch_size: int
    learning_rate: float
    engines: tuple[EngineSpec, ...]
    models: tuple[ModelSpec, ...]


def model_output_dim(model: ModelSpec) -> int:
    return model.layers[-1].units


def max_model_output_dim(manifest: Manifest) -> int:
    return max(model_output_dim(m) for m in manifest.models)


def load_manifest(path: Path | None = None) -> Manifest:
    path = path or MANIFEST_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    engines = tuple(
        EngineSpec(
            id=e["id"],
            enabled=bool(e.get("enabled", True)),
            inference_only=bool(e.get("inference_only", False)),
        )
        for e in raw["engines"]
    )

    models: list[ModelSpec] = []
    for m in raw["models"]:
        default_bias = bool(m.get("bias", True))
        layers = tuple(
            LayerSpec(
                units=l["units"],
                activation=l.get("activation", "linear"),
                bias=l.get("bias"),
            )
            for l in m["layers"]
        )
        models.append(
            ModelSpec(
                id=m["id"],
                input_dim=int(m["input_dim"]),
                layers=layers,
                bias=default_bias,
            )
        )

    return Manifest(
        fixture_version=raw["fixture_version"],
        seed=int(raw["seed"]),
        train_samples=int(raw["train_samples"]),
        test_samples=int(raw["test_samples"]),
        max_input_dim=int(raw["max_input_dim"]),
        output_dim=int(raw["output_dim"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        engines=engines,
        models=tuple(models),
    )
