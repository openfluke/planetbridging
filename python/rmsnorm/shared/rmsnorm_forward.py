"""Loom-compatible RMSNorm forward (eps=1e-6, per-token over dim)."""

from __future__ import annotations

import numpy as np


def init_rmsnorm_weights(dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gamma = rng.standard_normal(dim, dtype=np.float32) * 0.02 + 1.0
    return {"gamma": gamma}


def loom_rmsnorm_forward(
    x: np.ndarray,
    *,
    gamma: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """x [N, seq, dim] -> [N, seq*dim] matching Loom RMSNormForwardPolymorphic."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    n, seq, dim = x.shape
    out = np.zeros((n, seq, dim), dtype=np.float64)
    for i in range(n):
        for t in range(seq):
            row = x[i, t]
            rms = np.sqrt((row * row).mean() + eps)
            out[i, t] = row / rms * gamma
    return out.reshape(n, -1)
