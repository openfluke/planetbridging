"""Loom-compatible Residual forward (output = main + skip)."""

from __future__ import annotations

import numpy as np


def loom_residual_forward(main: np.ndarray, skip: np.ndarray) -> np.ndarray:
    """main, skip [N, seq, dim] -> [N, seq*dim] matching Loom ResidualForwardPolymorphic."""
    main = np.asarray(main, dtype=np.float64)
    skip = np.asarray(skip, dtype=np.float64)
    if main.shape != skip.shape:
        raise ValueError(f"shape mismatch: main {main.shape} skip {skip.shape}")
    return (main + skip).reshape(main.shape[0], -1)
