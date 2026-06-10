#!/usr/bin/env python3
"""JAX Residual bedrock (numpy reference forward)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.layer_stream import layer_stream_from_model  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.residual_forward import loom_residual_forward  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "jax"
REQUIRED: list[str] = []


def ensure_complete(*, model: ModelSpec) -> bool:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = is_complete(PLANET, model.id, REQUIRED)
    if not skipped:
        write_complete(PLANET, model.id, REQUIRED, jax.__version__)
    return skipped


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    _ = models_dir
    skipped = ensure_complete(model=model)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, _, x_main_test, x_skip_test, _ = slice_model_inputs(data, model, out_dim)
    native = loom_residual_forward(x_main_test, x_skip_test)[:, :out_dim]
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="jax",
            outputs=native,
            artifact_paths=[str(out_dir / "meta.json")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            layer=layer_stream_from_model(model),
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, jax.__version__, handler))
