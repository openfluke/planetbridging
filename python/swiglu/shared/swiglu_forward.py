"""Loom-compatible SwiGLU forward (SiLU gate, packed weight layout)."""

from __future__ import annotations

import numpy as np


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def init_swiglu_weights(input_dim: int, intermediate_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 0.02
    gate_w = rng.standard_normal((intermediate_dim, input_dim), dtype=np.float32) * scale
    up_w = rng.standard_normal((intermediate_dim, input_dim), dtype=np.float32) * scale
    down_w = rng.standard_normal((input_dim, intermediate_dim), dtype=np.float32) * scale
    gate_b = rng.standard_normal(intermediate_dim, dtype=np.float32) * scale
    up_b = rng.standard_normal(intermediate_dim, dtype=np.float32) * scale
    down_b = rng.standard_normal(input_dim, dtype=np.float32) * scale
    return {
        "gate_w": gate_w,
        "up_w": up_w,
        "down_w": down_w,
        "gate_b": gate_b,
        "up_b": up_b,
        "down_b": down_b,
    }


def pack_swiglu_weights(weights: dict[str, np.ndarray]) -> np.ndarray:
    """Flat pack: gateW | upW | downW | gateB | upB | downB (Loom WeightStore layout)."""
    parts = [
        weights["gate_w"].reshape(-1),
        weights["up_w"].reshape(-1),
        weights["down_w"].reshape(-1),
        weights["gate_b"].reshape(-1),
        weights["up_b"].reshape(-1),
        weights["down_b"].reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32)


def loom_swiglu_forward(
    x: np.ndarray,
    *,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
    gate_b: np.ndarray,
    up_b: np.ndarray,
    down_b: np.ndarray,
) -> np.ndarray:
    """x [N, seq, input_dim] -> [N, seq*input_dim] matching Loom SwiGLUForwardPolymorphic."""
    x = np.asarray(x, dtype=np.float64)
    in_dim = gate_w.shape[1]
    inter = gate_w.shape[0]
    gate_w = np.asarray(gate_w, dtype=np.float64).reshape(inter, in_dim)
    up_w = np.asarray(up_w, dtype=np.float64).reshape(inter, in_dim)
    down_w = np.asarray(down_w, dtype=np.float64).reshape(in_dim, inter)
    gate_b = np.asarray(gate_b, dtype=np.float64).reshape(inter)
    up_b = np.asarray(up_b, dtype=np.float64).reshape(inter)
    down_b = np.asarray(down_b, dtype=np.float64).reshape(in_dim)

    n, seq, _ = x.shape
    out = np.zeros((n, seq, in_dim), dtype=np.float64)
    for i in range(n):
        for t in range(seq):
            row = x[i, t]
            g = gate_w @ row + gate_b
            u = up_w @ row + up_b
            h = _silu(g) * u
            out[i, t] = down_w @ h + down_b
    return out.reshape(n, -1)
