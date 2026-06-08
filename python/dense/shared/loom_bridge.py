"""Bridge live planet dense layers → Loom .entity via compare-host stream API."""

from __future__ import annotations

from typing import Any

import numpy as np

from .extractors import (
    extract_jax_mlp,
    extract_keras_dense,
    extract_paddle_mlp,
    extract_pytorch_sequential,
    extract_sklearn_mlp,
)
from .layer_stream import DenseLayerStream, post_layer_stream
from .manifest import Manifest, ModelSpec
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
    """Loop dense layers from the live model, stream to Go, return loom VariantResult."""
    layers = _extract(extractor, net, model, params)
    resp = post_layer_stream(
        host=host,
        planet=planet,
        model=model,
        fixture_version=manifest.fixture_version,
        layers=layers,
    )
    if resp.get("status") != "ok":
        raise RuntimeError(f"loom stream: {resp.get('message', resp)}")

    outputs = np.asarray(resp["outputs"], dtype=np.float64)
    entity_path = resp.get("entity_path", "")
    max_diff = resp.get("max_abs_diff")
    if max_diff is not None:
        print(
            f"[{planet}] {model.id} loom stream ok entity={entity_path} "
            f"layers={resp.get('layer_count')} native_diff_max={max_diff:.6e} "
            f"exact={resp.get('exact_match')}"
        )
    else:
        print(f"[{planet}] {model.id} loom stream ok entity={entity_path}")

    return VariantResult(
        planet=planet,
        stage="loom",
        format="entity",
        outputs=outputs,
        artifact_paths=[entity_path] if entity_path else [],
        train_skipped=True,
    )


def _extract(
    extractor: str,
    net: Any,
    model: ModelSpec,
    params: Any | None,
) -> list[DenseLayerStream]:
    match extractor:
        case "pytorch":
            return extract_pytorch_sequential(net, model)
        case "keras":
            return extract_keras_dense(net, model)
        case "jax":
            if params is None:
                raise ValueError("jax extractor requires params")
            return extract_jax_mlp(params, model)
        case "sklearn":
            return extract_sklearn_mlp(net, model)
        case "paddle":
            return extract_paddle_mlp(net, model)
        case _:
            raise ValueError(f"unknown extractor {extractor!r}")
