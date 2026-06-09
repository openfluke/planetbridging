#!/usr/bin/env python3
"""TensorFlow RNN bedrock (Loom weight layout)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.rnn_forward import loom_rnn_forward_batch  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402
from shared.weights import init_or_load_weights, layer_from_weights, save_weights  # noqa: E402

PLANET = "tensorflow"
REQUIRED = ["weights.npz"]


def flatten_out(y: np.ndarray, out_dim: int) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)[:, :out_dim]


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[np.ndarray, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    skipped = is_complete(PLANET, model.id, REQUIRED)
    weights = init_or_load_weights(weights_path, model, manifest, skipped=skipped)
    if not skipped:
        save_weights(weights_path, weights)
        write_complete(PLANET, model.id, REQUIRED, tf.__version__)
    return weights, skipped


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    weights, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)
    native = flatten_out(
        loom_rnn_forward_batch(
            x_test,
            weights=weights,
            input_size=model.input_size,
            hidden_size=model.hidden_size,
        ),
        out_dim,
    )
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="keras",
            outputs=native,
            artifact_paths=[str(out_dir / "weights.npz")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host, planet=PLANET, model=model, manifest=manifest, layer=layer_from_weights(model, weights)
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, tf.__version__, handler))
