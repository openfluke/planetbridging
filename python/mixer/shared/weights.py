"""Init, save, and stream mixer_all_v1 layer weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import mixer_spec as ms  # noqa: F401 — used by init_or_load_weights
from .layer_stream import layers_json_from_weights
from .manifest import Manifest, ModelSpec
from .mixer_forward import _load_sibling_forward


def model_seed(manifest: Manifest, model: ModelSpec) -> int:
    return manifest.seed + sum(ord(c) for c in model.id)


def _init_dense(rng: np.random.Generator, in_dim: int, out_dim: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    w = rng.standard_normal((out_dim, in_dim), dtype=np.float32) * scale
    b = np.zeros(out_dim, dtype=np.float32)
    return w.reshape(-1), b


def _init_conv3(rng: np.random.Generator, scale: float) -> np.ndarray:
    shape = (ms.CNN3_FILTERS, ms.VOLUME_C, ms.CNN3_KERNEL, ms.CNN3_KERNEL, ms.CNN3_KERNEL)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).reshape(-1)


def _init_conv2(rng: np.random.Generator, scale: float) -> np.ndarray:
    shape = (ms.CNN2_FILTERS, 1, ms.CNN2_KERNEL, ms.CNN2_KERNEL)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).reshape(-1)


def _init_conv1(rng: np.random.Generator, scale: float) -> np.ndarray:
    shape = (ms.CNN1_FILTERS, 1, ms.CNN1_KERNEL)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).reshape(-1)


def _init_mha(rng: np.random.Generator, scale: float) -> dict[str, np.ndarray]:
    d = ms.MHA_D_MODEL
    q_dim = ms.MHA_HEADS * ms.MHA_HEAD_DIM
    return {
        "mha_q_w": (rng.standard_normal((q_dim, d), dtype=np.float32) * scale).reshape(-1),
        "mha_q_b": np.zeros(q_dim, dtype=np.float32),
        "mha_k_w": (rng.standard_normal((q_dim, d), dtype=np.float32) * scale).reshape(-1),
        "mha_k_b": np.zeros(q_dim, dtype=np.float32),
        "mha_v_w": (rng.standard_normal((q_dim, d), dtype=np.float32) * scale).reshape(-1),
        "mha_v_b": np.zeros(q_dim, dtype=np.float32),
        "mha_o_w": (rng.standard_normal((d, q_dim), dtype=np.float32) * scale).reshape(-1),
        "mha_o_b": np.zeros(d, dtype=np.float32),
    }


def init_mixer_v2_weights(seed: int, scale: float = 0.02) -> dict[str, np.ndarray]:
    weights = init_mixer_weights(seed, scale)
    rng = np.random.default_rng(seed + 99)
    emb_fwd = _load_sibling_forward("embedding", "embedding_forward")
    ln_fwd = _load_sibling_forward("layernorm", "layernorm_forward")
    rms_fwd = _load_sibling_forward("rmsnorm", "rmsnorm_forward")
    swiglu_fwd = _load_sibling_forward("swiglu", "swiglu_forward")

    emb = emb_fwd.init_embedding_weights(ms.EMBED_VOCAB, ms.EMBED_DIM, seed + 10)
    ln = ln_fwd.init_layernorm_weights(ms.EMBED_DIM, seed + 11)
    rms = rms_fwd.init_rmsnorm_weights(ms.EMBED_DIM, seed + 12)
    sw = swiglu_fwd.init_swiglu_weights(ms.EMBED_DIM, ms.SWIGLU_INTER, seed + 13)

    weights["embed_table"] = emb["table"]
    weights["layernorm_gamma"] = ln["gamma"]
    weights["layernorm_beta"] = ln["beta"]
    weights["rmsnorm_gamma"] = rms["gamma"]
    weights["swiglu_gate_w"] = sw["gate_w"]
    weights["swiglu_up_w"] = sw["up_w"]
    weights["swiglu_down_w"] = sw["down_w"]
    weights["swiglu_gate_b"] = sw["gate_b"]
    weights["swiglu_up_b"] = sw["up_b"]
    weights["swiglu_down_b"] = sw["down_b"]
    return weights


def init_mixer_weights(seed: int, scale: float = 0.02) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rnn_fwd = _load_sibling_forward("rnn", "rnn_forward")
    lstm_fwd = _load_sibling_forward("lstm", "lstm_forward")

    dense1_w, dense1_b = _init_dense(rng, 2, ms.DENSE_BRIDGE1, scale)
    dense2_w, dense2_b = _init_dense(rng, 6, ms.DENSE_BRIDGE2, scale)
    dense3_w, dense3_b = _init_dense(rng, 4, ms.DENSE_BRIDGE3, scale)
    dense4_w, dense4_b = _init_dense(rng, 8, ms.OUTPUT_DIM, scale)

    weights: dict[str, np.ndarray] = {
        "cnn3": _init_conv3(rng, scale),
        "dense1_w": dense1_w,
        "dense1_b": dense1_b,
        "cnn2": _init_conv2(rng, scale),
        "dense2_w": dense2_w,
        "dense2_b": dense2_b,
        "cnn1": _init_conv1(rng, scale),
        "dense3_w": dense3_w,
        "dense3_b": dense3_b,
        "rnn": rnn_fwd.init_loom_rnn_weights(ms.RECURRENT_IN, ms.RECURRENT_HID, seed + 1, scale),
        **(_init_mha(rng, scale)),
    }
    gates = lstm_fwd.init_loom_lstm_weights(ms.RECURRENT_IN, ms.RECURRENT_HID, seed + 2, scale)
    weights["lstm_i"] = gates["i"]
    weights["lstm_f"] = gates["f"]
    weights["lstm_g"] = gates["g"]
    weights["lstm_o"] = gates["o"]
    weights["dense4_w"] = dense4_w
    weights["dense4_b"] = dense4_b
    return weights


def init_or_load_weights(
    path: Path,
    model: ModelSpec,
    manifest: Manifest,
    *,
    skipped: bool,
) -> dict[str, np.ndarray]:
    if skipped:
        data = np.load(path)
        return {k: data[k] for k in data.files}
    seed = model_seed(manifest, model)
    if model.id == ms.MODEL_ID_V2:
        weights = init_mixer_v2_weights(seed)
    else:
        weights = init_mixer_weights(seed)
    np.savez(path, **weights)
    return weights


def save_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    np.savez(path, **weights)


def layer_streams_from_weights(weights: dict[str, np.ndarray], model: ModelSpec) -> list[dict[str, Any]]:
    return layers_json_from_weights(weights, model.id)
