"""Extract dense weights by looping live planet models — no checkpoint file parsing."""

from __future__ import annotations

from typing import Any

import numpy as np

from .layer_stream import (
    DenseLayerStream,
    keras_kernel_to_loom,
    layer_streams_from_specs,
    pytorch_weight_to_loom,
)
from .manifest import LayerSpec, ModelSpec, model_output_dim


def _layer_bias(layer: LayerSpec, model: ModelSpec) -> bool:
    if layer.bias is not None:
        return layer.bias
    return model.bias


# --- PyTorch ---


def extract_pytorch_sequential(net: Any, model: ModelSpec) -> list[DenseLayerStream]:
    import torch.nn as nn

    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if len(linears) != len(model.layers):
        raise ValueError(f"pytorch: {len(linears)} Linear modules, want {len(model.layers)}")

    kernels: list[np.ndarray] = []
    biases: list[np.ndarray | None] = []
    for lin, spec in zip(linears, model.layers):
        w = lin.weight.detach().cpu().numpy()
        kernels.append(w)
        if _layer_bias(spec, model) and lin.bias is not None:
            biases.append(lin.bias.detach().cpu().numpy())
        else:
            biases.append(None)
    return layer_streams_from_specs(model, kernels, biases)


# --- TensorFlow / Keras ---


def extract_keras_dense(net: Any, model: ModelSpec) -> list[DenseLayerStream]:
    import tensorflow as tf

    dense_layers = [l for l in net.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) != len(model.layers):
        raise ValueError(f"tensorflow: {len(dense_layers)} Dense layers, want {len(model.layers)}")

    streams: list[DenseLayerStream] = []
    in_dim = model.input_dim
    for i, (spec, layer) in enumerate(zip(model.layers, dense_layers)):
        weights = layer.get_weights()
        if not weights:
            raise ValueError(f"tensorflow: layer {i} has no weights")
        kernel = weights[0]
        flat = keras_kernel_to_loom(kernel)
        bias_arr: np.ndarray | None = None
        if _layer_bias(spec, model):
            if len(weights) < 2:
                raise ValueError(f"tensorflow: layer {i} missing bias")
            bias_arr = np.asarray(weights[1], dtype=np.float32).reshape(-1)

        streams.append(
            DenseLayerStream(
                index=i,
                input_dim=in_dim,
                output_dim=spec.units,
                activation=spec.activation.lower(),
                weights=flat,
                bias=bias_arr,
            )
        )
        in_dim = spec.units
    return streams


# --- JAX / Flax ---


def extract_jax_mlp(params: Any, model: ModelSpec) -> list[DenseLayerStream]:
    """Walk Flax params dict for Dense_N/kernel and Dense_N/bias."""
    kernels: list[np.ndarray] = []
    biases: list[np.ndarray | None] = []

    for i, spec in enumerate(model.layers):
        key = f"Dense_{i}"
        block = params.get(key) if isinstance(params, dict) else None
        if block is None:
            raise ValueError(f"jax: missing params block {key}")
        kernel = np.asarray(block["kernel"], dtype=np.float32)
        kernels.append(kernel)
        if _layer_bias(spec, model) and "bias" in block:
            biases.append(np.asarray(block["bias"], dtype=np.float32))
        else:
            biases.append(None)

    return layer_streams_from_specs(model, kernels, biases)


# --- sklearn ---


def extract_sklearn_mlp(reg: Any, model: ModelSpec) -> list[DenseLayerStream]:
    """MLPRegressor coefs_: list of [out, in] weight matrices."""
    if len(reg.coefs_) != len(model.layers):
        raise ValueError(f"sklearn: {len(reg.coefs_)} coef layers, want {len(model.layers)}")

    kernels = [np.asarray(c, dtype=np.float32) for c in reg.coefs_]
    biases: list[np.ndarray | None] = []
    for i, spec in enumerate(model.layers):
        if _layer_bias(spec, model) and reg.intercepts_ is not None:
            biases.append(np.asarray(reg.intercepts_[i], dtype=np.float32))
        else:
            biases.append(None)
    return layer_streams_from_specs(model, kernels, biases)


# --- Paddle ---


def extract_paddle_mlp(net: Any, model: ModelSpec) -> list[DenseLayerStream]:
    import paddle.nn as nn

    linears = [l for l in net.layers if isinstance(l, nn.Linear)]
    if len(linears) != len(model.layers):
        raise ValueError(f"paddle: {len(linears)} Linear layers, want {len(model.layers)}")

    kernels: list[np.ndarray] = []
    biases: list[np.ndarray | None] = []
    for lin, spec in zip(linears, model.layers):
        w = lin.weight.numpy()
        kernels.append(w)
        if _layer_bias(spec, model) and lin.bias is not None:
            biases.append(lin.bias.numpy())
        else:
            biases.append(None)
    return layer_streams_from_specs(model, kernels, biases)
