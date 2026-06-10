"""Load/save Loom-format LayerNorm weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layer_stream import layer_stream_from_weights
from .layernorm_forward import init_layernorm_weights
from .manifest import Manifest, ModelSpec


def model_seed(manifest: Manifest, model: ModelSpec) -> int:
    return manifest.seed + sum(ord(c) for c in model.id)


def init_or_load_weights(
    path: Path,
    model: ModelSpec,
    manifest: Manifest,
    *,
    skipped: bool,
) -> dict[str, np.ndarray]:
    if skipped:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    weights = init_layernorm_weights(model.dim, model_seed(manifest, model))
    np.savez(path, **weights)
    return weights


def save_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    np.savez(path, **weights)


def layer_from_weights(model: ModelSpec, weights: dict[str, np.ndarray]):
    return layer_stream_from_weights(model, weights=weights)
