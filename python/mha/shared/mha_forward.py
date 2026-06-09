"""Loom-compatible causal MHA forward (RoPE + softmax attention)."""

from __future__ import annotations

import math

import numpy as np

from .spec import ROPE_THETA


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """w row-major [out, in], x [..., in] -> [..., out]."""
    out = x @ w.T
    if b is not None and len(b) > 0:
        out = out + b
    return out


def _apply_rope(vec: np.ndarray, pos: int, num_heads: int, head_dim: int, theta: float) -> None:
    half = head_dim // 2
    for h in range(num_heads):
        base = h * head_dim
        for d in range(half):
            angle = pos / (theta ** (2 * d / head_dim))
            c, s = math.cos(angle), math.sin(angle)
            v0, v1 = vec[base + d], vec[base + d + half]
            vec[base + d] = v0 * c - v1 * s
            vec[base + d + half] = v0 * s + v1 * c


def loom_mha_forward(
    x: np.ndarray,
    *,
    q_w: np.ndarray,
    q_b: np.ndarray | None,
    k_w: np.ndarray,
    k_b: np.ndarray | None,
    v_w: np.ndarray,
    v_b: np.ndarray | None,
    o_w: np.ndarray,
    o_b: np.ndarray | None,
    num_heads: int,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    rope_theta: float = ROPE_THETA,
) -> np.ndarray:
    """Single sequence [seq, d_model] -> [seq, d_model]."""
    x = np.asarray(x, dtype=np.float64)
    seq_len, d_model = x.shape
    if head_dim is None:
        head_dim = d_model // num_heads
    if num_kv_heads is None:
        num_kv_heads = num_heads
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    cache_k = np.zeros((seq_len, kv_dim), dtype=np.float64)
    cache_v = np.zeros((seq_len, kv_dim), dtype=np.float64)
    out = np.zeros((seq_len, d_model), dtype=np.float64)

    for s in range(seq_len):
        q = _linear(x[s], q_w, q_b)
        k = _linear(x[s], k_w, k_b)
        v = _linear(x[s], v_w, v_b)
        _apply_rope(q, s, num_heads, head_dim, rope_theta)
        _apply_rope(k, s, num_kv_heads, head_dim, rope_theta)
        cache_k[s] = k
        cache_v[s] = v

        attn = np.zeros(q_dim, dtype=np.float64)
        for h in range(num_heads):
            kv_h = h // heads_per_kv
            scores = np.zeros(s + 1, dtype=np.float64)
            for kp in range(s + 1):
                dot = float(np.dot(q[h * head_dim : (h + 1) * head_dim], cache_k[kp, kv_h * head_dim : (kv_h + 1) * head_dim]))
                scores[kp] = dot * scale
            smax = float(np.max(scores))
            exp_s = np.exp(scores - smax)
            denom = float(np.sum(exp_s))
            for d in range(head_dim):
                acc = 0.0
                for kp in range(s + 1):
                    acc += exp_s[kp] * cache_v[kp, kv_h * head_dim + d]
                attn[h * head_dim + d] = acc / denom

        out[s] = _linear(attn, o_w, o_b)
    return out


def loom_mha_forward_batch(
    x: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """[batch, seq, d_model] -> [batch, seq, d_model], independent per batch row."""
    x = np.asarray(x, dtype=np.float64)
    batch = x.shape[0]
    outs = [loom_mha_forward(x[b], **kwargs) for b in range(batch)]
    return np.stack(outs, axis=0)
