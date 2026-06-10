"""Stream SwiGLU layer weights to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import ModelSpec
from .spec import BEDROCK, DEFAULT_HOST
from .swiglu_forward import pack_swiglu_weights


@dataclass(frozen=True)
class SwiGLULayerStream:
    index: int
    input_dim: int
    intermediate_dim: int
    weights: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": "swiglu",
            "index": self.index,
            "input_dim": self.input_dim,
            "intermediate_dim": self.intermediate_dim,
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
        }


def layer_stream_from_weights(
    model: ModelSpec,
    *,
    weights: dict[str, np.ndarray],
) -> SwiGLULayerStream:
    packed = pack_swiglu_weights(weights)
    return SwiGLULayerStream(
        index=0,
        input_dim=model.input_dim,
        intermediate_dim=model.intermediate_dim,
        weights=packed,
    )


def post_swiglu_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layer: SwiGLULayerStream,
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "input_dim": model.input_dim,
        "intermediate_dim": model.intermediate_dim,
        "seq_len": model.seq_len,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict()],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/swiglu",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"swiglu loom stream failed ({exc.code}): {detail}") from exc
