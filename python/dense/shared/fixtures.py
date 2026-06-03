"""Deterministic shared train/test fixtures for all engines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import Manifest, ModelSpec, load_manifest, max_model_output_dim
from .spec import FIXTURES_DIR


def fixture_path(manifest: Manifest | None = None) -> Path:
    manifest = manifest or load_manifest()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR / f"{manifest.fixture_version}.npz"


def ensure_fixtures(manifest: Manifest | None = None) -> dict[str, np.ndarray]:
    manifest = manifest or load_manifest()
    path = fixture_path(manifest)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(manifest.seed)
    x_full = rng.standard_normal(
        (manifest.train_samples + manifest.test_samples, manifest.max_input_dim),
        dtype=np.float64,
    )
    x_train = x_full[: manifest.train_samples]
    x_test = x_full[manifest.train_samples :]

    # Nonlinear but deterministic target shared by every engine.
    def targets(x: np.ndarray, width: int) -> np.ndarray:
        parts = [
            np.sin(x[:, 0:1]) + 0.25 * np.cos(x[:, 1:2]),
            0.1 * x[:, 2:3] - 0.05 * x[:, 3:4],
            0.02 * np.tanh(x[:, 4:5] + x[:, 5:6]),
            0.01 * (x[:, 6:7] ** 2 - x[:, 7:8] ** 2),
            0.005 * x[:, 8:9],
            0.005 * np.sin(x[:, 9:10]),
            0.005 * np.cos(x[:, 10:11]),
            0.002 * (x[:, 11:12] + x[:, 12:13]),
        ]
        y = np.concatenate(parts, axis=1)
        if y.shape[1] < width:
            raise ValueError("output_dim exceeds generated target width")
        return y[:, :width].astype(np.float64)

    out_width = max(manifest.output_dim, max_model_output_dim(manifest))
    y_train = targets(x_train, out_width)
    y_test = targets(x_test, out_width)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=x_train.astype(np.float64),
        y_train=y_train,
        x_test=x_test.astype(np.float64),
        y_test=y_test,
    )
    return {
        "x_train": x_train.astype(np.float64),
        "y_train": y_train,
        "x_test": x_test.astype(np.float64),
        "y_test": y_test,
    }


def slice_model_inputs(
    data: dict[str, np.ndarray],
    input_dim: int,
    output_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = data["y_train"]
    y_test = data["y_test"]
    if output_dim is not None:
        y_train = y_train[:, :output_dim]
        y_test = y_test[:, :output_dim]
    return (
        data["x_train"][:, :input_dim],
        y_train,
        data["x_test"][:, :input_dim],
        y_test,
    )
