"""Stream mixer layer weights to Go host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from . import mixer_spec as ms
from .manifest import ModelSpec
from .spec import BEDROCK, DEFAULT_HOST


def _arr(a: np.ndarray | None) -> list[float] | None:
    if a is None:
        return None
    return np.asarray(a, dtype=np.float64).tolist()


def layers_json_from_weights(weights: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "cnn3",
            "index": 0,
            "in_channels": ms.VOLUME_C,
            "filters": ms.CNN3_FILTERS,
            "input_depth": ms.VOLUME_D,
            "input_height": ms.VOLUME_H,
            "input_width": ms.VOLUME_W,
            "kernel_size": ms.CNN3_KERNEL,
            "stride": 1,
            "padding": 0,
            "activation": "linear",
            "weights": _arr(weights["cnn3"]),
        },
        {
            "kind": "dense",
            "index": 1,
            "input_dim": 2,
            "output_dim": ms.DENSE_BRIDGE1,
            "activation": "linear",
            "weights": _arr(weights["dense1_w"]),
            "bias": _arr(weights.get("dense1_b")),
        },
        {
            "kind": "cnn2",
            "index": 2,
            "in_channels": 1,
            "filters": ms.CNN2_FILTERS,
            "input_height": ms.CNN2_H,
            "input_width": ms.CNN2_W,
            "kernel_size": ms.CNN2_KERNEL,
            "stride": 1,
            "padding": 0,
            "activation": "linear",
            "weights": _arr(weights["cnn2"]),
        },
        {
            "kind": "dense",
            "index": 3,
            "input_dim": 6,
            "output_dim": ms.DENSE_BRIDGE2,
            "activation": "relu",
            "weights": _arr(weights["dense2_w"]),
            "bias": _arr(weights.get("dense2_b")),
        },
        {
            "kind": "cnn1",
            "index": 4,
            "in_channels": 1,
            "filters": ms.CNN1_FILTERS,
            "input_length": ms.CNN1_LEN,
            "kernel_size": ms.CNN1_KERNEL,
            "stride": 1,
            "padding": 0,
            "activation": "linear",
            "weights": _arr(weights["cnn1"]),
        },
        {
            "kind": "dense",
            "index": 5,
            "input_dim": 4,
            "output_dim": ms.DENSE_BRIDGE3,
            "activation": "linear",
            "weights": _arr(weights["dense3_w"]),
            "bias": _arr(weights.get("dense3_b")),
        },
        {
            "kind": "mha",
            "index": 6,
            "d_model": ms.MHA_D_MODEL,
            "num_heads": ms.MHA_HEADS,
            "num_kv_heads": ms.MHA_HEADS,
            "head_dim": ms.MHA_HEAD_DIM,
            "seq_len": ms.MHA_SEQ,
            "q_weights": _arr(weights["mha_q_w"]),
            "q_bias": _arr(weights.get("mha_q_b")),
            "k_weights": _arr(weights["mha_k_w"]),
            "k_bias": _arr(weights.get("mha_k_b")),
            "v_weights": _arr(weights["mha_v_w"]),
            "v_bias": _arr(weights.get("mha_v_b")),
            "o_weights": _arr(weights["mha_o_w"]),
            "o_bias": _arr(weights.get("mha_o_b")),
        },
        {
            "kind": "rnn",
            "index": 7,
            "input_size": ms.RECURRENT_IN,
            "hidden_size": ms.RECURRENT_HID,
            "seq_len": ms.RECURRENT_SEQ,
            "weights": _arr(weights["rnn"]),
        },
        {
            "kind": "lstm",
            "index": 8,
            "input_size": ms.RECURRENT_IN,
            "hidden_size": ms.RECURRENT_HID,
            "seq_len": ms.RECURRENT_SEQ,
            "i_weights": _arr(weights["lstm_i"]),
            "f_weights": _arr(weights["lstm_f"]),
            "g_weights": _arr(weights["lstm_g"]),
            "o_weights": _arr(weights["lstm_o"]),
        },
        {
            "kind": "dense",
            "index": 9,
            "input_dim": 8,
            "output_dim": ms.OUTPUT_DIM,
            "activation": "linear",
            "weights": _arr(weights["dense4_w"]),
            "bias": _arr(weights.get("dense4_b")),
        },
    ]


def post_mixer_stream(
    *,
    host: str,
    planet: str,
    model: ModelSpec,
    fixture_version: str,
    layers: list[dict[str, Any]],
    output_dim: int,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "bedrock": BEDROCK,
        "planet": planet,
        "model_id": model.id,
        "fixture_version": fixture_version,
        "output_dim": output_dim,
        "layers": layers,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/loom/stream/mixer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"mixer loom stream failed ({exc.code}): {detail}") from exc
