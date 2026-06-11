"""Numpy reference forward for mixer_all_v1 (matches bridge/mixer_infer.go)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import mixer_spec as ms


def _ensure_shared_pkg(bedrock: str, shared_dir: Path) -> str:
    """Register bedrock/shared as a package so sibling forwards can use relative imports."""
    pkg_name = f"_mixer_{bedrock}.shared"
    if pkg_name in sys.modules:
        return pkg_name
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
    )
    pkg.__path__ = [str(shared_dir)]
    sys.modules[pkg_name] = pkg
    return pkg_name


def _load_shared_module(pkg_name: str, shared_dir: Path, stem: str) -> Any:
    mod_name = f"{pkg_name}.{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = shared_dir / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        mod_name, path, submodule_search_locations=[str(shared_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_sibling_forward(bedrock: str, module: str) -> Any:
    root = Path(__file__).resolve().parents[2]
    shared_dir = root / bedrock / "shared"
    pkg_name = _ensure_shared_pkg(bedrock, shared_dir)
    if (shared_dir / "spec.py").exists():
        _load_shared_module(pkg_name, shared_dir, "spec")
    return _load_shared_module(pkg_name, shared_dir, module)


_lstm_fwd = _load_sibling_forward("lstm", "lstm_forward")
_rnn_fwd = _load_sibling_forward("rnn", "rnn_forward")
_mha_fwd = _load_sibling_forward("mha", "mha_forward")
_ln_fwd = _load_sibling_forward("layernorm", "layernorm_forward")
_emb_fwd = _load_sibling_forward("embedding", "embedding_forward")
_rms_fwd = _load_sibling_forward("rmsnorm", "rmsnorm_forward")
_swiglu_fwd = _load_sibling_forward("swiglu", "swiglu_forward")


def load_mha_torch_forward() -> Any:
    """Load mha/shared/mha_forward_torch without colliding with mixer.shared."""
    root = Path(__file__).resolve().parents[2]
    shared_dir = root / "mha" / "shared"
    pkg_name = _ensure_shared_pkg("mha", shared_dir)
    _load_shared_module(pkg_name, shared_dir, "spec")
    mod = _load_shared_module(pkg_name, shared_dir, "mha_forward_torch")
    return mod.loom_mha_forward_torch


def _out_spatial(spatial: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    return (spatial + 2 * padding - kernel) // stride + 1


def _activate(x: np.ndarray, activation: str) -> np.ndarray:
    act = activation.lower()
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "tanh":
        return np.tanh(x)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    return x


def _dense_linear(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
    activation: str,
) -> np.ndarray:
    """x [N, in], weights row-major [out, in]."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n, in_dim = x.shape
    out_dim = w.size // in_dim
    w_mat = w.reshape(out_dim, in_dim)
    out = x @ w_mat.T
    if bias is not None and len(bias) > 0:
        out = out + np.asarray(bias, dtype=np.float64)
    return _activate(out, activation)


def _flatten_leading(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return x.reshape(n, -1)


def _cnn3_forward(
    x: np.ndarray,
    *,
    weights: np.ndarray,
    in_c: int,
    filters: int,
    k: int,
    stride: int = 1,
    padding: int = 0,
    activation: str = "linear",
) -> np.ndarray:
    """x [N, C, D, H, W] → [N, filters, outD, outH, outW]."""
    n, _, in_d, in_h, in_w = x.shape
    out_d = _out_spatial(in_d, k, stride, padding)
    out_h = _out_spatial(in_h, k, stride, padding)
    out_w = _out_spatial(in_w, k, stride, padding)
    w = np.asarray(weights, dtype=np.float64).reshape(filters, in_c, k, k, k)
    out = np.zeros((n, filters, out_d, out_h, out_w), dtype=np.float64)
    for b in range(n):
        for f in range(filters):
            for od in range(out_d):
                for oh in range(out_h):
                    for ow in range(out_w):
                        s = 0.0
                        for ic in range(in_c):
                            for kd in range(k):
                                id_ = od * stride + kd - padding
                                if id_ < 0 or id_ >= in_d:
                                    continue
                                for kh in range(k):
                                    ih = oh * stride + kh - padding
                                    if ih < 0 or ih >= in_h:
                                        continue
                                    for kw in range(k):
                                        iw = ow * stride + kw - padding
                                        if 0 <= iw < in_w:
                                            s += float(x[b, ic, id_, ih, iw]) * float(w[f, ic, kd, kh, kw])
                        out[b, f, od, oh, ow] = s
    return _activate(out, activation)


def _cnn2_forward(
    x: np.ndarray,
    *,
    weights: np.ndarray,
    in_c: int,
    filters: int,
    k: int,
    stride: int = 1,
    padding: int = 0,
    activation: str = "linear",
) -> np.ndarray:
    """x [N, C, H, W] → [N, filters, outH, outW]."""
    n, _, in_h, in_w = x.shape
    out_h = _out_spatial(in_h, k, stride, padding)
    out_w = _out_spatial(in_w, k, stride, padding)
    w = np.asarray(weights, dtype=np.float64).reshape(filters, in_c, k, k)
    out = np.zeros((n, filters, out_h, out_w), dtype=np.float64)
    for b in range(n):
        for f in range(filters):
            for oh in range(out_h):
                for ow in range(out_w):
                    s = 0.0
                    for ic in range(in_c):
                        for kh in range(k):
                            for kw in range(k):
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                if 0 <= ih < in_h and 0 <= iw < in_w:
                                    s += float(x[b, ic, ih, iw]) * float(w[f, ic, kh, kw])
                    out[b, f, oh, ow] = s
    return _activate(out, activation)


def _cnn1_forward(
    x: np.ndarray,
    *,
    weights: np.ndarray,
    in_c: int,
    filters: int,
    k: int,
    stride: int = 1,
    padding: int = 0,
    activation: str = "linear",
) -> np.ndarray:
    """x [N, C, L] → [N, filters, outLen]."""
    n, _, seq_len = x.shape
    out_len = _out_spatial(seq_len, k, stride, padding)
    w = np.asarray(weights, dtype=np.float64).reshape(filters, in_c, k)
    out = np.zeros((n, filters, out_len), dtype=np.float64)
    for b in range(n):
        for f in range(filters):
            for o in range(out_len):
                s = 0.0
                for ic in range(in_c):
                    for ki in range(k):
                        pos = o * stride + ki - padding
                        if 0 <= pos < seq_len:
                            s += float(x[b, ic, pos]) * float(w[f, ic, ki])
                out[b, f, o] = s
    return _activate(out, activation)


def loom_mixer_forward(x: np.ndarray, weights: dict[str, np.ndarray], output_dim: int | None = None) -> np.ndarray:
    """Single batch row or batch [N, 1, 2, 2, 2] → [N, output_dim]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:
        x = x[np.newaxis, ...]
    out_dim = output_dim or ms.OUTPUT_DIM

    t = _cnn3_forward(
        x,
        weights=weights["cnn3"],
        in_c=ms.VOLUME_C,
        filters=ms.CNN3_FILTERS,
        k=ms.CNN3_KERNEL,
    )
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense1_w"], weights.get("dense1_b"), "linear")
    t = t.reshape(t.shape[0], 1, ms.CNN2_H, ms.CNN2_W)

    t = _cnn2_forward(
        t,
        weights=weights["cnn2"],
        in_c=1,
        filters=ms.CNN2_FILTERS,
        k=ms.CNN2_KERNEL,
    )
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense2_w"], weights.get("dense2_b"), "relu")
    t = t.reshape(t.shape[0], 1, ms.CNN1_LEN)

    t = _cnn1_forward(
        t,
        weights=weights["cnn1"],
        in_c=1,
        filters=ms.CNN1_FILTERS,
        k=ms.CNN1_KERNEL,
    )
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense3_w"], weights.get("dense3_b"), "linear")
    t = t.reshape(t.shape[0], ms.MHA_SEQ, ms.MHA_D_MODEL)

    q_dim = ms.MHA_HEADS * ms.MHA_HEAD_DIM
    d_model = ms.MHA_D_MODEL
    t = _mha_fwd.loom_mha_forward_batch(
        t,
        q_w=np.asarray(weights["mha_q_w"], dtype=np.float64).reshape(q_dim, d_model),
        q_b=weights.get("mha_q_b"),
        k_w=np.asarray(weights["mha_k_w"], dtype=np.float64).reshape(q_dim, d_model),
        k_b=weights.get("mha_k_b"),
        v_w=np.asarray(weights["mha_v_w"], dtype=np.float64).reshape(q_dim, d_model),
        v_b=weights.get("mha_v_b"),
        o_w=np.asarray(weights["mha_o_w"], dtype=np.float64).reshape(d_model, q_dim),
        o_b=weights.get("mha_o_b"),
        num_heads=ms.MHA_HEADS,
        head_dim=ms.MHA_HEAD_DIM,
    )

    t = _rnn_fwd.loom_rnn_forward_batch(
        t,
        weights=weights["rnn"],
        input_size=ms.RECURRENT_IN,
        hidden_size=ms.RECURRENT_HID,
    )

    t = _lstm_fwd.loom_lstm_forward_batch(
        t,
        i_weights=weights["lstm_i"],
        f_weights=weights["lstm_f"],
        g_weights=weights["lstm_g"],
        o_weights=weights["lstm_o"],
        input_size=ms.RECURRENT_IN,
        hidden_size=ms.RECURRENT_HID,
    )

    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense4_w"], weights.get("dense4_b"), "linear")
    return t[:, :out_dim]


def loom_mixer_forward_batch(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    output_dim: int | None = None,
) -> np.ndarray:
    """[N, 1, 2, 2, 2] → [N, output_dim]."""
    return loom_mixer_forward(x, weights, output_dim=output_dim)


def _v1_prefix(x: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """CNN3→Dense→CNN2→Dense→CNN1→Dense through spatial spine; returns [N, seq, d_model]."""
    t = _cnn3_forward(
        x,
        weights=weights["cnn3"],
        in_c=ms.VOLUME_C,
        filters=ms.CNN3_FILTERS,
        k=ms.CNN3_KERNEL,
    )
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense1_w"], weights.get("dense1_b"), "linear")
    t = t.reshape(t.shape[0], 1, ms.CNN2_H, ms.CNN2_W)
    t = _cnn2_forward(t, weights=weights["cnn2"], in_c=1, filters=ms.CNN2_FILTERS, k=ms.CNN2_KERNEL)
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense2_w"], weights.get("dense2_b"), "relu")
    t = t.reshape(t.shape[0], 1, ms.CNN1_LEN)
    t = _cnn1_forward(t, weights=weights["cnn1"], in_c=1, filters=ms.CNN1_FILTERS, k=ms.CNN1_KERNEL)
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense3_w"], weights.get("dense3_b"), "linear")
    return t.reshape(t.shape[0], ms.MHA_SEQ, ms.MHA_D_MODEL)


def loom_mixer_v2_forward(
    x: np.ndarray,
    token_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    output_dim: int | None = None,
) -> np.ndarray:
    """Full 16-layer stack: all 12 Loom types."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:
        x = x[np.newaxis, ...]
    out_dim = output_dim or ms.OUTPUT_DIM
    _ = _v1_prefix(x, weights)  # dense3 spine (embedding uses token ids)

    n = x.shape[0]
    seq, dim = ms.EMBED_SEQ, ms.EMBED_DIM
    t = _emb_fwd.loom_embedding_forward(token_ids, table=weights["embed_table"])
    t = t.reshape(n, seq, dim)

    t = _ln_fwd.loom_layernorm_forward(
        t,
        gamma=weights["layernorm_gamma"],
        beta=weights["layernorm_beta"],
    ).reshape(n, seq, dim)
    skip_attn = t.copy()

    q_dim = ms.MHA_HEADS * ms.MHA_HEAD_DIM
    d_model = ms.MHA_D_MODEL
    t = _mha_fwd.loom_mha_forward_batch(
        t,
        q_w=np.asarray(weights["mha_q_w"], dtype=np.float64).reshape(q_dim, d_model),
        q_b=weights.get("mha_q_b"),
        k_w=np.asarray(weights["mha_k_w"], dtype=np.float64).reshape(q_dim, d_model),
        k_b=weights.get("mha_k_b"),
        v_w=np.asarray(weights["mha_v_w"], dtype=np.float64).reshape(q_dim, d_model),
        v_b=weights.get("mha_v_b"),
        o_w=np.asarray(weights["mha_o_w"], dtype=np.float64).reshape(d_model, q_dim),
        o_b=weights.get("mha_o_b"),
        num_heads=ms.MHA_HEADS,
        head_dim=ms.MHA_HEAD_DIM,
    )
    t = t + skip_attn

    t = _rms_fwd.loom_rmsnorm_forward(t, gamma=weights["rmsnorm_gamma"]).reshape(n, seq, dim)
    skip_mlp = t.copy()

    t = _swiglu_fwd.loom_swiglu_forward(
        t,
        gate_w=weights["swiglu_gate_w"],
        up_w=weights["swiglu_up_w"],
        down_w=weights["swiglu_down_w"],
        gate_b=weights["swiglu_gate_b"],
        up_b=weights["swiglu_up_b"],
        down_b=weights["swiglu_down_b"],
    ).reshape(n, seq, dim)

    t = t + skip_mlp

    t = _rnn_fwd.loom_rnn_forward_batch(
        t,
        weights=weights["rnn"],
        input_size=ms.RECURRENT_IN,
        hidden_size=ms.RECURRENT_HID,
    )
    t = _lstm_fwd.loom_lstm_forward_batch(
        t,
        i_weights=weights["lstm_i"],
        f_weights=weights["lstm_f"],
        g_weights=weights["lstm_g"],
        o_weights=weights["lstm_o"],
        input_size=ms.RECURRENT_IN,
        hidden_size=ms.RECURRENT_HID,
    )
    t = _flatten_leading(t)
    t = _dense_linear(t, weights["dense4_w"], weights.get("dense4_b"), "linear")
    return t[:, :out_dim]


def loom_mixer_v2_forward_batch(
    x: np.ndarray,
    token_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    output_dim: int | None = None,
) -> np.ndarray:
    return loom_mixer_v2_forward(x, token_ids, weights, output_dim=output_dim)
