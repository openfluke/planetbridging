"""Load/save Loom-format RNN weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layer_stream import layer_stream_from_weights
from .manifest import Manifest, ModelSpec
from .rnn_forward import init_loom_rnn_weights


def model_seed(manifest: Manifest, model: ModelSpec) -> int:
    return manifest.seed + sum(ord(c) for c in model.id)


def init_or_load_weights(
    path: Path,
    model: ModelSpec,
    manifest: Manifest,
    *,
    skipped: bool,
) -> np.ndarray:
    if skipped:
        data = np.load(path)
        return data["weights"]
    weights = init_loom_rnn_weights(model.input_size, model.hidden_size, model_seed(manifest, model))
    np.savez(path, weights=weights)
    return weights


def save_weights(path: Path, weights: np.ndarray) -> None:
    np.savez(path, weights=weights)


def layer_from_weights(model: ModelSpec, weights: np.ndarray):
    return layer_stream_from_weights(model, weights=weights)
