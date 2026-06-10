"""Stream Residual layer metadata to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .manifest import ModelSpec
from .spec import BEDROCK, DEFAULT_HOST


@dataclass(frozen=True)
class ResidualLayerStream:
    index: int
    dim: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": "residual",
            "index": self.index,
            "dim": self.dim,
        }


def layer_stream_from_model(model: ModelSpec) -> ResidualLayerStream:
    return ResidualLayerStream(index=0, dim=model.dim)


def post_residual_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layer: ResidualLayerStream,
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "dim": model.dim,
        "seq_len": model.seq_len,
        "output_dim": output_dim,
        "layers": [layer.to_json_dict()],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/residual",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"residual loom stream failed ({exc.code}): {detail}") from exc
