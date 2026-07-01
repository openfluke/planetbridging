"""High-level absorb API — stream live models into Loom .entity for any bedrock."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import engines
from .engines import EngineStreamResult
from .layers.dense import LayerSpec, layers_from_jax, layers_from_keras, layers_from_pytorch, layers_from_sklearn
from .stream import StreamResult, stream_dense


def stream(
    bedrock: str,
    planet: str,
    *,
    model_id: str | None = None,
    **kwargs: Any,
) -> EngineStreamResult:
    """Stream any bedrock layer type from a live AI engine (pytorch/tensorflow/jax/sklearn)."""
    return engines.stream(bedrock, planet, model_id=model_id, **kwargs)


def stream_all_bedrocks(planet: str, **kwargs: Any) -> list[EngineStreamResult]:
    return engines.stream_all_bedrocks(planet, **kwargs)


def stream_all_planets(bedrock: str, **kwargs: Any) -> list[EngineStreamResult]:
    return engines.stream_all_planets(bedrock, **kwargs)


def _default_specs(units: Sequence[int], *, activation: str = "relu") -> tuple[LayerSpec, ...]:
    return tuple(LayerSpec(u, activation) for u in units)


def pytorch(
    net: Any,
    *,
    planet: str = "pytorch",
    model_id: str = "model",
    input_dim: int,
    layer_specs: Sequence[LayerSpec] | None = None,
    layer_units: Sequence[int] | None = None,
    activation: str = "relu",
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixture_version: str = "",
    root: str | Path | None = None,
    **kwargs: Any,
) -> StreamResult:
    """Stream a PyTorch nn.Module (sequential Dense stack) into Loom .entity."""
    specs = _resolve_specs(layer_specs, layer_units, activation)
    layers = layers_from_pytorch(net, specs, input_dim=input_dim)
    return stream_dense(
        planet=planet,
        model_id=model_id,
        layers=layers,
        input_dim=input_dim,
        inputs=inputs,
        native_outputs=native_outputs,
        output_path=output_path,
        models_dir=models_dir,
        fixture_version=fixture_version,
        root=root,
        **kwargs,
    )


def keras(
    net: Any,
    *,
    planet: str = "tensorflow",
    model_id: str = "model",
    input_dim: int,
    layer_specs: Sequence[LayerSpec] | None = None,
    layer_units: Sequence[int] | None = None,
    activation: str = "relu",
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixture_version: str = "",
    root: str | Path | None = None,
    **kwargs: Any,
) -> StreamResult:
    """Stream a Keras / TensorFlow model into Loom .entity."""
    specs = _resolve_specs(layer_specs, layer_units, activation)
    layers = layers_from_keras(net, specs, input_dim=input_dim)
    return stream_dense(
        planet=planet,
        model_id=model_id,
        layers=layers,
        input_dim=input_dim,
        inputs=inputs,
        native_outputs=native_outputs,
        output_path=output_path,
        models_dir=models_dir,
        fixture_version=fixture_version,
        root=root,
        **kwargs,
    )


def jax(
    params: Any,
    *,
    planet: str = "jax",
    model_id: str = "model",
    input_dim: int,
    layer_specs: Sequence[LayerSpec] | None = None,
    layer_units: Sequence[int] | None = None,
    activation: str = "relu",
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixture_version: str = "",
    root: str | Path | None = None,
    **kwargs: Any,
) -> StreamResult:
    """Stream Flax/JAX params dict into Loom .entity."""
    specs = _resolve_specs(layer_specs, layer_units, activation)
    layers = layers_from_jax(params, specs, input_dim=input_dim)
    return stream_dense(
        planet=planet,
        model_id=model_id,
        layers=layers,
        input_dim=input_dim,
        inputs=inputs,
        native_outputs=native_outputs,
        output_path=output_path,
        models_dir=models_dir,
        fixture_version=fixture_version,
        root=root,
        **kwargs,
    )


def sklearn(
    reg: Any,
    *,
    planet: str = "sklearn",
    model_id: str = "model",
    input_dim: int,
    layer_specs: Sequence[LayerSpec] | None = None,
    layer_units: Sequence[int] | None = None,
    activation: str = "linear",
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixture_version: str = "",
    root: str | Path | None = None,
    **kwargs: Any,
) -> StreamResult:
    """Stream sklearn MLPRegressor into Loom .entity."""
    specs = _resolve_specs(layer_specs, layer_units, activation)
    layers = layers_from_sklearn(reg, specs, input_dim=input_dim)
    return stream_dense(
        planet=planet,
        model_id=model_id,
        layers=layers,
        input_dim=input_dim,
        inputs=inputs,
        native_outputs=native_outputs,
        output_path=output_path,
        models_dir=models_dir,
        fixture_version=fixture_version,
        root=root,
        **kwargs,
    )


def _resolve_specs(
    layer_specs: Sequence[LayerSpec] | None,
    layer_units: Sequence[int] | None,
    activation: str,
) -> tuple[LayerSpec, ...]:
    if layer_specs is not None:
        return tuple(layer_specs)
    if layer_units is None:
        raise ValueError("layer_specs or layer_units required")
    return _default_specs(layer_units, activation=activation)
