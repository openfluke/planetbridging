"""Stream live weights into Loom .entity via loom-stream CLI (no HTTP).

Supports all 13 POC bedrock layer types: dense, cnn1–3, mha, lstm, rnn,
layernorm, embedding, rmsnorm, swiglu, residual, mixer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._binary import repo_root, run_loom_stream
from .compare import compare_outputs, diff_label
from .layers.dense import DenseLayer


@dataclass(frozen=True)
class StreamResult:
    entity_path: str
    outputs: np.ndarray
    layer_count: int
    weight_bytes: int
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    exact_match: bool | None = None
    compare_label: str | None = None
    native_reference: np.ndarray | None = None

    @property
    def output_dim(self) -> int:
        if self.outputs.ndim == 1:
            return 1
        return int(self.outputs.shape[-1])


def _finalize_entity_path(entity_path: str, output_path: str | Path | None) -> str:
    if output_path is not None and entity_path and Path(entity_path) != Path(output_path):
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(entity_path).read_bytes())
        return str(dest)
    return entity_path


def _result_from_response(
    resp: dict,
    *,
    layer_count: int,
    native_outputs: np.ndarray | None,
    entity_path: str,
) -> StreamResult:
    outputs = np.asarray(resp.get("outputs", []), dtype=np.float64)
    max_diff = mean_diff = None
    exact = label = None
    if native_outputs is not None and outputs.size > 0:
        max_diff, mean_diff, exact = compare_outputs(native_outputs, outputs)
        label = diff_label(max_diff, exact=exact)
    return StreamResult(
        entity_path=entity_path,
        outputs=outputs,
        layer_count=int(resp.get("layer_count", layer_count)),
        weight_bytes=int(resp.get("weight_bytes", 0)),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        exact_match=exact,
        compare_label=label,
        native_reference=native_outputs,
    )


def stream_bedrock(
    bedrock: str,
    payload: dict[str, Any],
    *,
    fixture_version: str = "",
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    root: str | Path | None = None,
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    binary: str | Path | None = None,
    skip_infer: bool = False,
) -> StreamResult:
    """Generic stream for any bedrock — same JSON contract as compare-host HTTP.

    ``payload`` must match the Go bridge stream request for that bedrock
    (see python/<bedrock>/shared/layer_stream.py post_*_stream functions).
    """
    bedrock = bedrock.lower().strip()
    if not payload.get("layers"):
        raise ValueError("payload.layers required")

    root_path = Path(root) if root else repo_root()
    fv = fixture_version or str(payload.get("fixture_version", ""))
    envelope: dict[str, Any] = {
        "bedrock": bedrock,
        "root": str(root_path),
    }
    if fv:
        envelope["fixture_version"] = fv
    if models_dir is not None:
        envelope["models_dir"] = str(models_dir)
    if fixtures_dir is not None:
        envelope["fixtures_dir"] = str(fixtures_dir)
    if output_path is not None:
        envelope["output_path"] = str(output_path)
    if skip_infer:
        envelope["skip_infer"] = True
    if inputs is not None:
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        envelope["inputs"] = x.tolist()

    body = {**payload}
    if "bedrock" not in body:
        body["bedrock"] = bedrock
    if fv and "fixture_version" not in body:
        body["fixture_version"] = fv

    resp = run_loom_stream(binary=Path(binary) if binary else None, envelope=envelope, payload=body)
    if resp.get("status") != "ok":
        raise RuntimeError(resp.get("message", resp))

    entity_path = _finalize_entity_path(resp.get("entity_path", ""), output_path)
    return _result_from_response(
        resp,
        layer_count=len(payload["layers"]),
        native_outputs=native_outputs,
        entity_path=entity_path,
    )


def stream_dense(
    *,
    planet: str,
    model_id: str,
    layers: Sequence[DenseLayer],
    input_dim: int,
    fixture_version: str = "",
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    root: str | Path | None = None,
    inputs: np.ndarray | None = None,
    native_outputs: np.ndarray | None = None,
    binary: str | Path | None = None,
    skip_infer: bool = False,
) -> StreamResult:
    """Stream dense layers → .entity + optional Loom infer."""
    if not layers:
        raise ValueError("layers required")
    payload = {
        "planet": planet,
        "model_id": model_id,
        "fixture_version": fixture_version,
        "input_dim": input_dim,
        "layers": [layer.to_json_dict() for layer in layers],
    }
    return stream_bedrock(
        "dense",
        payload,
        fixture_version=fixture_version,
        output_path=output_path,
        models_dir=models_dir,
        fixtures_dir=fixtures_dir,
        root=root,
        inputs=inputs,
        native_outputs=native_outputs,
        binary=binary,
        skip_infer=skip_infer,
    )


def stream_mixer(
    *,
    planet: str,
    model_id: str,
    layers: list[dict],
    output_dim: int,
    fixture_version: str = "mixer_bedrock_v1",
    output_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    fixtures_dir: str | Path | None = None,
    root: str | Path | None = None,
    native_outputs: np.ndarray | None = None,
    binary: str | Path | None = None,
) -> StreamResult:
    """Stream mixer v1/v2 layer chain (10 or 16 layers) → .entity."""
    payload = {
        "bedrock": "mixer",
        "planet": planet,
        "model_id": model_id,
        "fixture_version": fixture_version,
        "output_dim": output_dim,
        "layers": layers,
    }
    return stream_bedrock(
        "mixer",
        payload,
        fixture_version=fixture_version,
        output_path=output_path,
        models_dir=models_dir,
        fixtures_dir=fixtures_dir,
        root=root,
        native_outputs=native_outputs,
        binary=binary,
    )


# Convenience wrappers — pass full payload dict (see python/<bedrock>/shared/layer_stream.py)


def _stream_typed(bedrock: str, payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return stream_bedrock(bedrock, payload, **kwargs)


def stream_cnn1(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("cnn1", payload, **kwargs)


def stream_cnn2(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("cnn2", payload, **kwargs)


def stream_cnn3(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("cnn3", payload, **kwargs)


def stream_mha(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("mha", payload, **kwargs)


def stream_lstm(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("lstm", payload, **kwargs)


def stream_rnn(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("rnn", payload, **kwargs)


def stream_layernorm(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("layernorm", payload, **kwargs)


def stream_embedding(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("embedding", payload, **kwargs)


def stream_rmsnorm(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("rmsnorm", payload, **kwargs)


def stream_swiglu(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("swiglu", payload, **kwargs)


def stream_residual(payload: dict[str, Any], **kwargs: Any) -> StreamResult:
    return _stream_typed("residual", payload, **kwargs)
