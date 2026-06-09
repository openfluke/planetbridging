"""Stream LSTM layer weights to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifest import ModelSpec
from .spec import BEDROCK, DEFAULT_HOST


@dataclass(frozen=True)
class LSTMLayerStream:
    index: int
    input_size: int
    hidden_size: int
    seq_len: int
    i_weights: np.ndarray
    f_weights: np.ndarray
    g_weights: np.ndarray
    o_weights: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        def arr(a: np.ndarray) -> list[float]:
            return np.asarray(a, dtype=np.float64).tolist()

        return {
            "kind": "lstm",
            "index": self.index,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len,
            "i_weights": arr(self.i_weights),
            "f_weights": arr(self.f_weights),
            "g_weights": arr(self.g_weights),
            "o_weights": arr(self.o_weights),
        }


def layer_stream_from_weights(
    model: ModelSpec,
    *,
    i_weights: np.ndarray,
    f_weights: np.ndarray,
    g_weights: np.ndarray,
    o_weights: np.ndarray,
) -> LSTMLayerStream:
    return LSTMLayerStream(
        index=0,
        input_size=model.input_size,
        hidden_size=model.hidden_size,
        seq_len=model.seq_len,
        i_weights=np.asarray(i_weights, dtype=np.float32).reshape(-1),
        f_weights=np.asarray(f_weights, dtype=np.float32).reshape(-1),
        g_weights=np.asarray(g_weights, dtype=np.float32).reshape(-1),
        o_weights=np.asarray(o_weights, dtype=np.float32).reshape(-1),
    )


def post_lstm_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layer: LSTMLayerStream,
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "input_size": model.input_size,
        "hidden_size": model.hidden_size,
        "seq_len": model.seq_len,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict()],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/lstm",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"lstm loom stream failed ({exc.code}): {detail}") from exc
