"""Stream MHA layer weights to Go host."""

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
class MHALayerStream:
    index: int
    d_model: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    q_weights: np.ndarray
    q_bias: np.ndarray | None
    k_weights: np.ndarray
    k_bias: np.ndarray | None
    v_weights: np.ndarray
    v_bias: np.ndarray | None
    o_weights: np.ndarray
    o_bias: np.ndarray | None

    def to_json_dict(self) -> dict[str, Any]:
        def arr(a: np.ndarray | None) -> list[float] | None:
            if a is None:
                return None
            return np.asarray(a, dtype=np.float64).tolist()

        return {
            "kind": "mha",
            "index": self.index,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "seq_len": self.seq_len,
            "q_weights": arr(self.q_weights),
            "q_bias": arr(self.q_bias),
            "k_weights": arr(self.k_weights),
            "k_bias": arr(self.k_bias),
            "v_weights": arr(self.v_weights),
            "v_bias": arr(self.v_bias),
            "o_weights": arr(self.o_weights),
            "o_bias": arr(self.o_bias),
        }


def pytorch_linear_to_loom(weight: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(weight, dtype=np.float32)).reshape(-1)


def keras_dense_to_loom(kernel: np.ndarray) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(k.T).reshape(-1)


def layer_stream_from_weights(
    model: ModelSpec,
    *,
    q_w: np.ndarray,
    q_b: np.ndarray | None,
    k_w: np.ndarray,
    k_b: np.ndarray | None,
    v_w: np.ndarray,
    v_b: np.ndarray | None,
    o_w: np.ndarray,
    o_b: np.ndarray | None,
) -> MHALayerStream:
    return MHALayerStream(
        index=0,
        d_model=model.d_model,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv,
        head_dim=model.head_dim,
        seq_len=model.seq_len,
        q_weights=np.asarray(q_w, dtype=np.float32).reshape(-1),
        q_bias=None if q_b is None else np.asarray(q_b, dtype=np.float32),
        k_weights=np.asarray(k_w, dtype=np.float32).reshape(-1),
        k_bias=None if k_b is None else np.asarray(k_b, dtype=np.float32),
        v_weights=np.asarray(v_w, dtype=np.float32).reshape(-1),
        v_bias=None if v_b is None else np.asarray(v_b, dtype=np.float32),
        o_weights=np.asarray(o_w, dtype=np.float32).reshape(-1),
        o_bias=None if o_b is None else np.asarray(o_b, dtype=np.float32),
    )


def post_mha_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layer: MHALayerStream,
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "d_model": model.d_model,
        "seq_len": model.seq_len,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict()],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/mha",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"mha loom stream failed ({exc.code}): {detail}") from exc
