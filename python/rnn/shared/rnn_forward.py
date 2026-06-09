"""Loom-compatible vanilla RNN forward (tanh, zero initial hidden)."""

from __future__ import annotations

import math

import numpy as np


def rnn_weight_size(input_size: int, hidden_size: int) -> int:
    return hidden_size * input_size + hidden_size * hidden_size + hidden_size


def pack_weights(w_ih: np.ndarray, w_hh: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Pack one RNN cell: ih [hidden,input] row-major, hh [hidden,hidden], bias [hidden]."""
    return np.concatenate(
        [
            np.asarray(w_ih, dtype=np.float32).reshape(-1),
            np.asarray(w_hh, dtype=np.float32).reshape(-1),
            np.asarray(bias, dtype=np.float32).reshape(-1),
        ]
    )


def unpack_weights(flat: np.ndarray, input_size: int, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ih = hidden_size * input_size
    hh = hidden_size * hidden_size
    w_ih = flat[:ih].reshape(hidden_size, input_size)
    w_hh = flat[ih : ih + hh].reshape(hidden_size, hidden_size)
    b = flat[ih + hh : ih + hh + hidden_size]
    return w_ih, w_hh, b


def init_loom_rnn_weights(
    input_size: int,
    hidden_size: int,
    seed: int,
    scale: float = 0.02,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w_ih = rng.standard_normal((hidden_size, input_size), dtype=np.float32) * scale
    w_hh = rng.standard_normal((hidden_size, hidden_size), dtype=np.float32) * scale
    b = np.zeros(hidden_size, dtype=np.float32)
    return pack_weights(w_ih, w_hh, b)


def loom_rnn_forward(
    x: np.ndarray,
    *,
    weights: np.ndarray,
    input_size: int,
    hidden_size: int,
) -> np.ndarray:
    """Single sequence [seq, input] -> [seq, hidden]."""
    x = np.asarray(x, dtype=np.float64)
    seq_len = x.shape[0]
    w_ih, w_hh, b = unpack_weights(weights, input_size, hidden_size)
    h_prev = np.zeros(hidden_size, dtype=np.float64)
    out = np.zeros((seq_len, hidden_size), dtype=np.float64)

    for t in range(seq_len):
        xt = x[t]
        pre = np.zeros(hidden_size, dtype=np.float64)
        for h in range(hidden_size):
            s = float(b[h])
            for i in range(input_size):
                s += float(xt[i]) * float(w_ih[h, i])
            for hp in range(hidden_size):
                s += float(h_prev[hp]) * float(w_hh[h, hp])
            pre[h] = s
        h_curr = np.tanh(pre)
        out[t] = h_curr
        h_prev = h_curr
    return out


def loom_rnn_forward_batch(
    x: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """[batch, seq, input] -> [batch, seq, hidden], independent per batch row."""
    x = np.asarray(x, dtype=np.float64)
    outs = [loom_rnn_forward(x[b], **kwargs) for b in range(x.shape[0])]
    return np.stack(outs, axis=0)
