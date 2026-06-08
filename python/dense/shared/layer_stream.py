"""Stream dense layer weights from live planet models into Loom .entity."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .manifest import LayerSpec, ModelSpec
from .spec import DEFAULT_HOST


@dataclass(frozen=True)
class DenseLayerStream:
    """One dense layer extracted from a planet runtime (not a checkpoint file)."""

    index: int
    input_dim: int
    output_dim: int
    activation: str
    weights: np.ndarray  # float32, row-major [out × in] for Loom
    bias: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "weights": self.weights.astype(np.float64).tolist(),
        }
        if self.bias is not None and len(self.bias) > 0:
            d["bias"] = self.bias.astype(np.float64).tolist()
        return d


def loom_row_major_weights(kernel: np.ndarray) -> np.ndarray:
    """Convert planet kernel to Loom Dense layout: row-major [out × in].

    Accepts:
      - PyTorch / ONNX style [out, in]
      - Keras / sklearn style [in, out] (transposed when in_dim != out_dim on axis 0)
    """
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim != 2:
        raise ValueError(f"expected rank-2 kernel, got shape {k.shape}")
    # Caller passes kernels already oriented; helper for Keras [in,out] → [out,in]
    return np.ascontiguousarray(k)


def keras_kernel_to_loom(kernel: np.ndarray) -> np.ndarray:
    """Keras Dense kernel is [in_features, out_features] → Loom [out, in] row-major."""
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(k.T).reshape(-1)


def pytorch_weight_to_loom(weight: np.ndarray) -> np.ndarray:
    """nn.Linear weight is [out, in] — already Loom layout."""
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32)).reshape(-1)


def stream_payload(
    *,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layers: list[DenseLayerStream],
) -> dict[str, Any]:
    return {
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "input_dim": model.input_dim,
        "layers": [layer.to_json_dict() for layer in layers],
    }


def post_layer_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layers: list[DenseLayerStream],
) -> dict[str, Any]:
    """POST live layer weights to Go host → builds VolumetricNetwork → .entity → infer."""
    host = host.rstrip("/")
    payload = stream_payload(
        planet=planet,
        model=model,
        fixture_version=fixture_version,
        layers=layers,
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"loom stream failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"compare-host unreachable at {host}; start with 'go run .' from planetbridging"
        ) from exc


def layer_streams_from_specs(
    model: ModelSpec,
    kernels: list[np.ndarray],
    biases: list[np.ndarray | None],
) -> list[DenseLayerStream]:
    if len(kernels) != len(model.layers):
        raise ValueError(f"expected {len(model.layers)} kernels, got {len(kernels)}")
    if len(biases) != len(model.layers):
        raise ValueError(f"expected {len(model.layers)} bias slots, got {len(biases)}")

    streams: list[DenseLayerStream] = []
    in_dim = model.input_dim
    for i, spec in enumerate(model.layers):
        w = np.asarray(kernels[i], dtype=np.float32)
        if w.ndim == 2:
            if w.shape == (spec.units, in_dim):
                flat = pytorch_weight_to_loom(w)
            elif w.shape == (in_dim, spec.units):
                flat = keras_kernel_to_loom(w)
            else:
                raise ValueError(f"layer {i}: kernel shape {w.shape} != ({spec.units},{in_dim}) or ({in_dim},{spec.units})")
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
            DenseLayerStream(
                index=i,
                input_dim=in_dim,
                output_dim=spec.units,
                activation=spec.activation.lower(),
                weights=flat,
                bias=b,
            )
        )
        in_dim = spec.units
    return streams


class SupportsDenseExtract(Protocol):
    """Planet models that can yield dense layer weights in memory."""

    def extract_dense_layers(self, model: ModelSpec) -> list[DenseLayerStream]: ...
