"""Loom-compatible LayerNorm forward (eps=1e-5, per-token over dim)."""

from __future__ import annotations

import numpy as np


def init_layernorm_weights(dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal(dim, dtype=np.float32) * 0.02 + 1.0
    beta = rng.standard_normal(dim, dtype=np.float32) * 0.02
    return {"gamma": gamma, "beta": beta}


def loom_layernorm_forward(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """x [N, seq, dim] -> [N, seq*dim] matching Loom LayerNormForwardPolymorphic."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    n, seq, dim = x.shape
    out = np.zeros((n, seq, dim), dtype=np.float64)
    for i in range(n):
        for t in range(seq):
            row = x[i, t]
            mean = row.mean()
            var = (row * row).mean() - mean * mean
            std = np.sqrt(var + eps)
            out[i, t] = (row - mean) / std * gamma + beta
    return out.reshape(n, -1)
