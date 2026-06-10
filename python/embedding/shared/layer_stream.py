"""Stream Embedding layer weights to Go host."""

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
class EmbeddingLayerStream:
    index: int
    vocab_size: int
    embedding_dim: int
    weights: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": "embedding",
            "index": self.index,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
        }


def layer_stream_from_weights(
    model: ModelSpec,
    *,
    weights: dict[str, np.ndarray],
) -> EmbeddingLayerStream:
    table = np.asarray(weights["table"], dtype=np.float32)
    flat = np.ascontiguousarray(table).reshape(-1)
    return EmbeddingLayerStream(
        index=0,
        vocab_size=model.vocab_size,
        embedding_dim=model.embed_dim,
        weights=flat,
    )


def post_embedding_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layer: EmbeddingLayerStream,
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "vocab_size": model.vocab_size,
        "seq_len": model.seq_len,
        "embed_dim": model.embed_dim,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict()],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/embedding",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding loom stream failed ({exc.code}): {detail}") from exc

