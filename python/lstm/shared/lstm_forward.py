"""Loom-compatible LSTM forward (i,f,g,o gates, zero initial state)."""

from __future__ import annotations

import math

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _gate_size(input_size: int, hidden_size: int) -> int:
    return hidden_size * input_size + hidden_size * hidden_size + hidden_size


def pack_gate(w_ih: np.ndarray, w_hh: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Pack one gate: ih [hidden,input] row-major, hh [hidden,hidden], bias [hidden]."""
    return np.concatenate(
        [
            np.asarray(w_ih, dtype=np.float32).reshape(-1),
            np.asarray(w_hh, dtype=np.float32).reshape(-1),
            np.asarray(bias, dtype=np.float32).reshape(-1),
        ]
    )


def unpack_gate(flat: np.ndarray, input_size: int, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ih = hidden_size * input_size
    hh = hidden_size * hidden_size
    w_ih = flat[:ih].reshape(hidden_size, input_size)
    w_hh = flat[ih : ih + hh].reshape(hidden_size, hidden_size)
    b = flat[ih + hh : ih + hh + hidden_size]
    return w_ih, w_hh, b


def init_loom_lstm_weights(
    input_size: int,
    hidden_size: int,
    seed: int,
    scale: float = 0.02,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gates = {}
    for name in ("i", "f", "g", "o"):
        w_ih = rng.standard_normal((hidden_size, input_size), dtype=np.float32) * scale
        w_hh = rng.standard_normal((hidden_size, hidden_size), dtype=np.float32) * scale
        b = np.zeros(hidden_size, dtype=np.float32)
        gates[name] = pack_gate(w_ih, w_hh, b)
    return gates


def loom_lstm_forward(
    x: np.ndarray,
    *,
    i_weights: np.ndarray,
    f_weights: np.ndarray,
    g_weights: np.ndarray,
    o_weights: np.ndarray,
    input_size: int,
    hidden_size: int,
) -> np.ndarray:
    """Single sequence [seq, input] -> [seq, hidden]."""
    x = np.asarray(x, dtype=np.float64)
    seq_len = x.shape[0]
    w_i, w_f, w_g, w_o = (
        unpack_gate(i_weights, input_size, hidden_size),
        unpack_gate(f_weights, input_size, hidden_size),
        unpack_gate(g_weights, input_size, hidden_size),
        unpack_gate(o_weights, input_size, hidden_size),
    )
    h_prev = np.zeros(hidden_size, dtype=np.float64)
    c_prev = np.zeros(hidden_size, dtype=np.float64)
    out = np.zeros((seq_len, hidden_size), dtype=np.float64)

    for t in range(seq_len):
        xt = x[t]
        pre_i = np.zeros(hidden_size, dtype=np.float64)
        pre_f = np.zeros(hidden_size, dtype=np.float64)
        pre_g = np.zeros(hidden_size, dtype=np.float64)
        pre_o = np.zeros(hidden_size, dtype=np.float64)
        for gate_pre, (w_ih, w_hh, b) in zip(
            (pre_i, pre_f, pre_g, pre_o),
            (w_i, w_f, w_g, w_o),
        ):
            for h in range(hidden_size):
                s = float(b[h])
                for i in range(input_size):
                    s += float(xt[i]) * float(w_ih[h, i])
                for hp in range(hidden_size):
                    s += float(h_prev[hp]) * float(w_hh[h, hp])
                gate_pre[h] = s
        i_g = np.array([_sigmoid(v) for v in pre_i])
        f_g = np.array([_sigmoid(v) for v in pre_f])
        g_g = np.tanh(pre_g)
        o_g = np.array([_sigmoid(v) for v in pre_o])
        c_curr = f_g * c_prev + i_g * g_g
        h_curr = o_g * np.tanh(c_curr)
        out[t] = h_curr
        h_prev, c_prev = h_curr, c_curr
    return out


def loom_lstm_forward_batch(
    x: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """[batch, seq, input] -> [batch, seq, hidden], independent per batch row."""
    x = np.asarray(x, dtype=np.float64)
    outs = [loom_lstm_forward(x[b], **kwargs) for b in range(x.shape[0])]
    return np.stack(outs, axis=0)
