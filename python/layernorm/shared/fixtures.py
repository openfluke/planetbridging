"""Deterministic LayerNorm fixtures [N, seq, dim]."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import Manifest, ModelSpec, load_manifest, model_output_dim
from .spec import FIXTURES_DIR


def fixture_path(manifest: Manifest | None = None) -> Path:
    manifest = manifest or load_manifest()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR / f"{manifest.fixture_version}.npz"


def _make_sequences(x_scalar: np.ndarray, seq_len: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    n = x_scalar.shape[0]
    out = np.zeros((n, seq_len, dim), dtype=np.float64)
    for i in range(n):
        phase = float(x_scalar[i, 0])
        for t in range(seq_len):
            for d in range(dim):
                out[i, t, d] = np.sin(0.11 * t + 0.07 * d + phase) + 0.05 * np.cos(0.03 * t + phase * 0.3)
        out[i] += rng.standard_normal((seq_len, dim)) * 0.01
    return out


def _targets(x: np.ndarray, width: int) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    parts = [
        np.sin(flat[:, 0:1]) + 0.25 * np.cos(flat[:, 1:2]),
        0.1 * flat[:, 2:3] - 0.05 * flat[:, 3:4],
        0.02 * np.tanh(flat[:, 4:5] + flat[:, 5:6]),
    ]
    y = np.concatenate(parts, axis=1)
    if width > y.shape[1]:
        reps = (width + y.shape[1] - 1) // y.shape[1]
        y = np.tile(y, (1, reps))
    return y[:, :width].astype(np.float64)


def ensure_fixtures(manifest: Manifest | None = None) -> dict[str, np.ndarray]:
    manifest = manifest or load_manifest()
    path = fixture_path(manifest)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(manifest.seed)
    n_train, n_test = manifest.train_samples, manifest.test_samples
    max_s, max_d = manifest.max_seq_len, manifest.max_dim

    x_scalar = rng.standard_normal((n_train + n_test, 1), dtype=np.float64)
    x_train = _make_sequences(x_scalar[:n_train], max_s, max_d, rng)
    x_test = _make_sequences(x_scalar[n_train:], max_s, max_d, rng)

    max_out = max(model_output_dim(m) for m in manifest.models)
    out_width = max(manifest.output_dim, max_out)
    y_train = _targets(x_train, out_width)
    y_test = _targets(x_test, out_width)

    np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def slice_model_inputs(
    data: dict[str, np.ndarray],
    model: ModelSpec,
    output_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq, dim = model.seq_len, model.dim
    x_train = data["x_train"][:, :seq, :dim]
    x_test = data["x_test"][:, :seq, :dim]
    y_train = data["y_train"]
    y_test = data["y_test"]
    if output_dim is not None:
        y_train = y_train[:, :output_dim]
        y_test = y_test[:, :output_dim]
    return x_train, y_train, x_test, y_test
