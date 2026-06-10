#!/usr/bin/env python3
"""JAX LayerNorm bedrock (numpy reference forward)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.layernorm_forward import loom_layernorm_forward  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402
from shared.weights import init_or_load_weights, layer_from_weights, save_weights  # noqa: E402

PLANET = "jax"
REQUIRED = ["weights.npz"]


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[dict[str, np.ndarray], bool]:
    _ = data
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    skipped = is_complete(PLANET, model.id, REQUIRED)
    weights = init_or_load_weights(weights_path, model, manifest, skipped=skipped)
    if not skipped:
        save_weights(weights_path, weights)
        write_complete(PLANET, model.id, REQUIRED, jax.__version__)
    return weights, skipped


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    _ = models_dir
    weights, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)
    native = loom_layernorm_forward(
        x_test,
        gamma=weights["gamma"],
        beta=weights["beta"],
    )[:, :out_dim]
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="jax",
            outputs=native,
            artifact_paths=[str(out_dir / "weights.npz")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            layer=layer_from_weights(model, weights),
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, jax.__version__, handler))
