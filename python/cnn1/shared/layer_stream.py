"""Stream CNN1 layer weights to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import CNN1LayerSpec, ModelSpec, cnn1_out_len
from .spec import BEDROCK, DEFAULT_HOST


@dataclass(frozen=True)
class CNN1LayerStream:
    index: int
    in_channels: int
    filters: int
    input_length: int
    kernel_size: int
    stride: int
    padding: int
    activation: str
    weights: np.ndarray  # float32 flat [filters × in_ch × kernel]
    bias: np.ndarray | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "cnn1",
            "index": self.index,
            "in_channels": self.in_channels,
            "filters": self.filters,
            "input_length": self.input_length,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "activation": self.activation,
            "weights": self.weights.astype(np.float64).tolist(),
        }
        if self.bias is not None and len(self.bias) > 0:
            d["bias"] = self.bias.astype(np.float64).tolist()
        return d


def pytorch_conv1d_to_loom(weight: np.ndarray) -> np.ndarray:
    """nn.Conv1d weight [out, in, k] — already Loom layout."""
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32)).reshape(-1)


def keras_conv1d_to_loom(kernel: np.ndarray) -> np.ndarray:
    """Keras Conv1D kernel [k, in, out] → Loom [out, in, k]."""
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(np.transpose(k, (2, 1, 0))).reshape(-1)


def layer_streams_from_specs(
    model: ModelSpec,
    kernels: list[np.ndarray],
) -> list[CNN1LayerStream]:
    if len(kernels) != len(model.layers):
        raise ValueError(f"expected {len(model.layers)} kernels, got {len(kernels)}")

    streams: list[CNN1LayerStream] = []
    in_ch = model.input_channels
    seq = model.seq_len
    for i, (spec, kernel) in enumerate(zip(model.layers, kernels)):
        want = spec.filters * in_ch * spec.kernel_size
        flat = np.asarray(kernel, dtype=np.float32).reshape(-1)
        if flat.size != want:
            raise ValueError(f"layer {i}: weight count {flat.size} != {want}")

        streams.append(
            CNN1LayerStream(
                index=i,
                in_channels=in_ch,
                filters=spec.filters,
                input_length=seq,
                kernel_size=spec.kernel_size,
                stride=spec.stride,
                padding=spec.padding,
                activation=spec.activation.lower(),
                weights=flat,
                bias=None,
            )
        )
        seq = cnn1_out_len(seq, spec.kernel_size, spec.stride, spec.padding)
        in_ch = spec.filters
    return streams


def post_cnn1_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layers: list[CNN1LayerStream],
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "input_channels": model.input_channels,
        "seq_len": model.seq_len,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict() for layer in layers],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/cnn1",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"cnn1 loom stream failed ({exc.code}): {detail}") from exc
