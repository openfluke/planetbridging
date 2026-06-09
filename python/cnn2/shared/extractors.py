"""Extract Conv2d weights from live planet models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .layer_stream import (
    CNN2LayerStream,
    keras_conv2d_to_loom,
    layer_streams_from_specs,
    pytorch_conv2d_to_loom,
)
from .manifest import ModelSpec


def extract_pytorch_conv2d(net: Any, model: ModelSpec) -> list[CNN2LayerStream]:
    import torch.nn as nn

    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    if len(convs) != len(model.layers):
        raise ValueError(f"pytorch: {len(convs)} Conv2d modules, want {len(model.layers)}")
    kernels = [pytorch_conv2d_to_loom(c.weight.detach().cpu().numpy()) for c in convs]
    return layer_streams_from_specs(model, kernels)


def extract_keras_conv2d(net: Any, model: ModelSpec) -> list[CNN2LayerStream]:
    import tensorflow as tf

    conv_layers = [l for l in net.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if len(conv_layers) != len(model.layers):
        raise ValueError(f"tensorflow: {len(conv_layers)} Conv2D layers, want {len(model.layers)}")
    kernels = []
    for layer in conv_layers:
        w = layer.get_weights()
        if not w:
            raise ValueError("tensorflow: conv layer missing weights")
        kernels.append(keras_conv2d_to_loom(w[0]))
    return layer_streams_from_specs(model, kernels)


def extract_jax_conv2d(params: Any, model: ModelSpec) -> list[CNN2LayerStream]:
    kernels: list[np.ndarray] = []
    for i, spec in enumerate(model.layers):
        key = f"Conv_{i}"
        block = params.get(key) if isinstance(params, dict) else None
        if block is None:
            raise ValueError(f"jax: missing params block {key}")
        k = np.asarray(block["kernel"], dtype=np.float32)
        kernels.append(keras_conv2d_to_loom(k))
    return layer_streams_from_specs(model, kernels)
