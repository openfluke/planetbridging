"""Stream CNN2 layer weights to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import ModelSpec, cnn2_out_spatial
from .spec import BEDROCK, DEFAULT_HOST


@dataclass(frozen=True)
class CNN2LayerStream:
    index: int
    in_channels: int
    filters: int
    input_height: int
    input_width: int
    kernel_size: int
    stride: int
    padding: int
    activation: str
    weights: np.ndarray  # float32 flat [filters × in_ch × kH × kW]
    bias: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "cnn2",
            "index": self.index,
            "in_channels": self.in_channels,
            "filters": self.filters,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "activation": self.activation,
            "weights": self.weights.astype(np.float64).tolist(),
        }
        if self.bias is not None and len(self.bias) > 0:
            d["bias"] = self.bias.astype(np.float64).tolist()
        return d


def pytorch_conv2d_to_loom(weight: np.ndarray) -> np.ndarray:
    """nn.Conv2d weight [out, in, kH, kW] — already Loom layout."""
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32)).reshape(-1)


def keras_conv2d_to_loom(kernel: np.ndarray) -> np.ndarray:
    """Keras Conv2D kernel [kH, kW, in, out] → Loom [out, in, kH, kW]."""
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(np.transpose(k, (3, 2, 0, 1))).reshape(-1)


def layer_streams_from_specs(
    model: ModelSpec,
    kernels: list[np.ndarray],
) -> list[CNN2LayerStream]:
    if len(kernels) != len(model.layers):
        raise ValueError(f"expected {len(model.layers)} kernels, got {len(kernels)}")

    streams: list[CNN2LayerStream] = []
    in_ch = model.input_channels
    height, width = model.height, model.width
    for i, (spec, kernel) in enumerate(zip(model.layers, kernels)):
        want = spec.filters * in_ch * spec.kernel_size * spec.kernel_size
        flat = np.asarray(kernel, dtype=np.float32).reshape(-1)
        if flat.size != want:
            raise ValueError(f"layer {i}: weight count {flat.size} != {want}")

        streams.append(
            CNN2LayerStream(
                index=i,
                in_channels=in_ch,
                filters=spec.filters,
                input_height=height,
                input_width=width,
                kernel_size=spec.kernel_size,
                stride=spec.stride,
                padding=spec.padding,
                activation=spec.activation.lower(),
                weights=flat,
                bias=None,
            )
        )
        height = cnn2_out_spatial(height, spec.kernel_size, spec.stride, spec.padding)
        width = cnn2_out_spatial(width, spec.kernel_size, spec.stride, spec.padding)
        in_ch = spec.filters
    return streams


def post_cnn2_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layers: list[CNN2LayerStream],
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "input_channels": model.input_channels,
        "height": model.height,
        "width": model.width,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict() for layer in layers],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/cnn2",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"cnn2 loom stream failed ({exc.code}): {detail}") from exc
