"""Stream live CNN2 layers to Loom .entity."""

from __future__ import annotations

from typing import Any

import numpy as np

from .extractors import extract_jax_conv2d, extract_keras_conv2d, extract_pytorch_conv2d
from .layer_stream import post_cnn2_stream
from .manifest import Manifest, ModelSpec, model_output_dim
from .variants import VariantResult


def stream_planet_to_loom(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    manifest: Manifest,
    net: Any,
    extractor: str,
    params: Any | None = None,
) -> VariantResult | None:
    layers = _extract(extractor, net, model, params)
    out_dim = model_output_dim(model)
    resp = post_cnn2_stream(
        host=host,
        planet=planet,
        model=model,
        fixture_version=manifest.fixture_version,
        layers=layers,
        output_dim=out_dim,
    )
    if resp.get("status") != "ok":
        raise RuntimeError(f"cnn2 loom stream: {resp.get('message', resp)}")

    outputs = np.asarray(resp["outputs"], dtype=np.float64)
    entity_path = resp.get("entity_path", "")
    max_diff = resp.get("max_abs_diff")
    if max_diff is not None:
        print(
            f"[{planet}] {model.id} cnn2 loom ok entity={entity_path} "
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


def _extract(extractor: str, net: Any, model: ModelSpec, params: Any | None):
    if extractor == "pytorch":
        return extract_pytorch_conv2d(net, model)
    if extractor == "tensorflow":
        return extract_keras_conv2d(net, model)
    if extractor == "jax":
        if params is None:
            raise ValueError("jax extractor requires params")
        return extract_jax_conv2d(params, model)
    raise ValueError(f"unknown extractor: {extractor}")
