"""Compare native engine outputs vs Loom entity inference."""

from __future__ import annotations

import numpy as np

FP32_PASS_TOLERANCE = 1e-5


def compare_outputs(
    native: np.ndarray,
    loom: np.ndarray,
    *,
    tolerance: float = FP32_PASS_TOLERANCE,
) -> tuple[float, float, bool]:
    """Return (max_abs_diff, mean_abs_diff, exact_match)."""
    a = np.asarray(native, dtype=np.float64)
    b = np.asarray(loom, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: native {a.shape} vs loom {b.shape}")
    diff = np.abs(a - b)
    exact = bool(np.all(a == b))
    return float(diff.max()), float(diff.mean()), exact


def diff_label(
    max_abs_diff: float,
    *,
    exact: bool = False,
    tolerance: float = FP32_PASS_TOLERANCE,
) -> str:
    if exact or max_abs_diff == 0.0:
        return "EXACT"
    if max_abs_diff < tolerance:
        return "PASS"
    return "DIFF"
