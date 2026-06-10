"""Stream live SwiGLU layers to Loom .entity."""

from __future__ import annotations

import numpy as np

from .layer_stream import SwiGLULayerStream, post_swiglu_stream
from .manifest import Manifest, ModelSpec, model_output_dim
from .variants import VariantResult


def stream_planet_to_loom(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    manifest: Manifest,
    layer: SwiGLULayerStream,
) -> VariantResult | None:
    out_dim = model_output_dim(model)
    resp = post_swiglu_stream(
        host=host,
        planet=planet,
        model=model,
        fixture_version=manifest.fixture_version,
        layer=layer,
        output_dim=out_dim,
    )
    if resp.get("status") != "ok":
        raise RuntimeError(f"swiglu loom stream: {resp.get('message', resp)}")

    outputs = np.asarray(resp["outputs"], dtype=np.float64)
    entity_path = resp.get("entity_path", "")
    max_diff = resp.get("max_abs_diff")
    if max_diff is not None:
        print(
            f"[{planet}] {model.id} swiglu loom ok entity={entity_path} "
            f"layers={resp.get('layer_count')} native_diff_max={max_diff:.6e}"
        )
    return VariantResult(
        planet=planet,
        stage="loom",
        format="entity",
        outputs=outputs,
        artifact_paths=[entity_path] if entity_path else [],
        train_skipped=True,
    )
