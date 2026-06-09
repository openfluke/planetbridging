"""Load/save Loom-format LSTM gate weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .lstm_forward import init_loom_lstm_weights
from .layer_stream import layer_stream_from_weights
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
        return {k: data[k] for k in ("i", "f", "g", "o")}
    gates = init_loom_lstm_weights(model.input_size, model.hidden_size, model_seed(manifest, model))
    np.savez(path, **gates)
    return gates


def save_weights(path: Path, gates: dict[str, np.ndarray]) -> None:
    np.savez(path, **gates)


def layer_from_gates(model: ModelSpec, gates: dict[str, np.ndarray]):
    return layer_stream_from_weights(
        model,
        i_weights=gates["i"],
        f_weights=gates["f"],
        g_weights=gates["g"],
        o_weights=gates["o"],
    )
