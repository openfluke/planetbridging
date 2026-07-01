"""Dense layer stream types — live weights from planet runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LayerSpec:
    units: int
    activation: str = "relu"
    bias: bool = True


@dataclass(frozen=True)
class DenseLayer:
    """One dense layer extracted from a live planet model."""

    index: int
    input_dim: int
    output_dim: int
    activation: str
    weights: np.ndarray  # float32 row-major [out × in]
    bias: np.ndarray | None = None

    def to_json_dict(self) -> dict:
        d: dict = {
            "index": self.index,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "weights": self.weights.astype(np.float64).tolist(),
        }
        if self.bias is not None and len(self.bias) > 0:
            d["bias"] = self.bias.astype(np.float64).tolist()
        return d


def keras_kernel_to_loom(kernel: np.ndarray) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(k.T).reshape(-1)


def pytorch_weight_to_loom(weight: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32)).reshape(-1)


def layers_from_specs(
    *,
    input_dim: int,
    specs: Sequence[LayerSpec],
    kernels: Sequence[np.ndarray],
    biases: Sequence[np.ndarray | None],
) -> list[DenseLayer]:
    if len(kernels) != len(specs) or len(biases) != len(specs):
        raise ValueError("kernels/biases length must match layer specs")

    streams: list[DenseLayer] = []
    in_dim = input_dim
    for i, spec in enumerate(specs):
        w = np.asarray(kernels[i], dtype=np.float32)
        if w.ndim == 2:
            if w.shape == (spec.units, in_dim):
                flat = pytorch_weight_to_loom(w)
            elif w.shape == (in_dim, spec.units):
                flat = keras_kernel_to_loom(w)
            else:
                raise ValueError(
                    f"layer {i}: kernel shape {w.shape} != ({spec.units},{in_dim}) or ({in_dim},{spec.units})"
                )
        else:
            flat = w.reshape(-1)
        if flat.size != in_dim * spec.units:
            raise ValueError(f"layer {i}: weight count {flat.size} != {in_dim}×{spec.units}")

        b = biases[i]
        if b is not None:
            b = np.asarray(b, dtype=np.float32).reshape(-1)
            if b.size != spec.units:
                raise ValueError(f"layer {i}: bias len {b.size} != {spec.units}")

        streams.append(
            DenseLayer(
                index=i,
                input_dim=in_dim,
                output_dim=spec.units,
                activation=spec.activation.lower(),
                weights=flat,
                bias=b if spec.bias else None,
            )
        )
        in_dim = spec.units
    return streams


def layers_from_pytorch(net: Any, specs: Sequence[LayerSpec], *, input_dim: int) -> list[DenseLayer]:
    import torch.nn as nn

    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if len(linears) != len(specs):
        raise ValueError(f"pytorch: {len(linears)} Linear modules, want {len(specs)}")

    kernels = [lin.weight.detach().cpu().numpy() for lin in linears]
    biases: list[np.ndarray | None] = []
    for lin, spec in zip(linears, specs):
        if spec.bias and lin.bias is not None:
            biases.append(lin.bias.detach().cpu().numpy())
        else:
            biases.append(None)
    return layers_from_specs(input_dim=input_dim, specs=specs, kernels=kernels, biases=biases)


def layers_from_keras(net: Any, specs: Sequence[LayerSpec], *, input_dim: int) -> list[DenseLayer]:
    import tensorflow as tf

    dense_layers = [l for l in net.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) != len(specs):
        raise ValueError(f"keras: {len(dense_layers)} Dense layers, want {len(specs)}")

    streams: list[DenseLayer] = []
    in_dim = input_dim
    for i, (spec, layer) in enumerate(zip(specs, dense_layers)):
        weights = layer.get_weights()
        if not weights:
            raise ValueError(f"keras: layer {i} has no weights")
        flat = keras_kernel_to_loom(weights[0])
        bias_arr: np.ndarray | None = None
        if spec.bias:
            if len(weights) < 2:
                raise ValueError(f"keras: layer {i} missing bias")
            bias_arr = np.asarray(weights[1], dtype=np.float32).reshape(-1)
        streams.append(
            DenseLayer(
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


def layers_from_jax(params: Any, specs: Sequence[LayerSpec], *, input_dim: int) -> list[DenseLayer]:
    kernels: list[np.ndarray] = []
    biases: list[np.ndarray | None] = []
    for i, spec in enumerate(specs):
        key = f"Dense_{i}"
        block = params.get(key) if isinstance(params, dict) else None
        if block is None:
            raise ValueError(f"jax: missing params block {key}")
        kernels.append(np.asarray(block["kernel"], dtype=np.float32))
        if spec.bias and "bias" in block:
            biases.append(np.asarray(block["bias"], dtype=np.float32))
        else:
            biases.append(None)
    return layers_from_specs(input_dim=input_dim, specs=specs, kernels=kernels, biases=biases)


def layers_from_sklearn(reg: Any, specs: Sequence[LayerSpec], *, input_dim: int) -> list[DenseLayer]:
    if len(reg.coefs_) != len(specs):
        raise ValueError(f"sklearn: {len(reg.coefs_)} coef layers, want {len(specs)}")
    kernels = [np.asarray(c, dtype=np.float32) for c in reg.coefs_]
    biases: list[np.ndarray | None] = []
    for i, spec in enumerate(specs):
        if spec.bias and reg.intercepts_ is not None:
            biases.append(np.asarray(reg.intercepts_[i], dtype=np.float32))
        else:
            biases.append(None)
    return layers_from_specs(input_dim=input_dim, specs=specs, kernels=kernels, biases=biases)
