"""Deterministic 5D mixer fixtures [N, C, D, H, W] (max spatial 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import Manifest, ModelSpec, load_manifest, model_input_dim, model_output_dim
from .mixer_spec import VOLUME_C, VOLUME_D, VOLUME_H, VOLUME_W
from .spec import FIXTURES_DIR


def fixture_path(manifest: Manifest | None = None) -> Path:
    manifest = manifest or load_manifest()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR / f"{manifest.fixture_version}.npz"


def _make_volumes(x_scalar: np.ndarray, spatial: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    n = x_scalar.shape[0]
    out = np.zeros((n, channels, spatial, spatial, spatial), dtype=np.float64)
    for i in range(n):
        phase = float(x_scalar[i, 0])
        for dep in range(spatial):
            for row in range(spatial):
                for col in range(spatial):
                    out[i, 0, dep, row, col] = (
                        np.sin(0.1 * dep + 0.08 * row + 0.06 * col + phase)
                        + 0.08 * np.cos(0.05 * dep - 0.04 * row + phase * 0.5)
                    )
        for c in range(1, channels):
            out[i, c, :, :, :] = out[i, 0, :, :, :] * (0.5 + 0.1 * c) + rng.standard_normal((spatial, spatial, spatial)) * 0.01
    return out


def _targets(x: np.ndarray, width: int) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    parts = [
        np.sin(flat[:, 0:1]) + 0.25 * np.cos(flat[:, 1:2]),
        0.1 * flat[:, 2:3] - 0.05 * flat[:, 3:4],
        0.02 * np.tanh(flat[:, 4:5] + flat[:, 5:6]),
        0.01 * (flat[:, 6:7] ** 2 - flat[:, 7:8] ** 2),
        0.005 * flat[:, 8:9],
        0.005 * np.sin(flat[:, 9:10]),
        0.005 * np.cos(flat[:, 10:11]),
        0.002 * (flat[:, 11:12] + flat[:, 12:13]),
    ]
    y = np.concatenate(parts, axis=1)
    return y[:, :width].astype(np.float64)


def ensure_fixtures(manifest: Manifest | None = None) -> dict[str, np.ndarray]:
    manifest = manifest or load_manifest()
    path = fixture_path(manifest)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(manifest.seed)
    n_train, n_test = manifest.train_samples, manifest.test_samples
    max_s = manifest.max_spatial

    x_scalar = rng.standard_normal((n_train + n_test, max_s ** 3), dtype=np.float64)
    x_train = _make_volumes(x_scalar[:n_train], max_s, manifest.input_channels, rng)
    x_test = _make_volumes(x_scalar[n_train:], max_s, manifest.input_channels, rng)

    out_width = max(manifest.output_dim, model_output_dim())
    y_train = _targets(x_train, out_width)
    y_test = _targets(x_test, out_width)

    np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def slice_model_inputs(
    data: dict[str, np.ndarray],
    model: ModelSpec | None = None,
    output_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _ = model
    c, d, h, w = VOLUME_C, VOLUME_D, VOLUME_H, VOLUME_W
    x_train = data["x_train"][:, :c, :d, :h, :w]
    x_test = data["x_test"][:, :c, :d, :h, :w]
    y_train = data["y_train"]
    y_test = data["y_test"]
    if output_dim is not None:
        y_train = y_train[:, :output_dim]
        y_test = y_test[:, :output_dim]
    return x_train, y_train, x_test, y_test


def volume_element_count() -> int:
    return model_input_dim()
