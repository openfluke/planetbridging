#!/usr/bin/env python3
"""scikit-learn MLPRegressor dense bedrock (classic dense baseline)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "sklearn"
REQUIRED = ["model.pkl"]


def hidden_layers(model: ModelSpec) -> tuple[int, ...]:
    linear_layers = [l for l in model.layers if l.activation == "linear"]
    if len(linear_layers) != 1:
        raise ValueError("sklearn engine supports one linear output head only")
    hidden = tuple(l.units for l in model.layers[:-1])
    return hidden


def activation(model: ModelSpec) -> str:
    if not model.layers[:-1]:
        return "identity"
    act = model.layers[0].activation.lower()
    mapping = {"relu": "relu", "tanh": "tanh", "sigmoid": "logistic", "linear": "identity"}
    for layer in model.layers[:-1]:
        if layer.activation.lower() != act:
            raise ValueError("sklearn engine requires uniform hidden activation")
    return mapping.get(act, act)


def train_or_load(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
):
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "model.pkl"

    if is_complete(PLANET, model.id, REQUIRED):
        with pkl_path.open("rb") as f:
            return pickle.load(f), True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model.input_dim, out_dim)
    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layers(model),
        activation=activation(model),
        solver="adam",
        alpha=0.0,
        batch_size=min(manifest.batch_size, x_train.shape[0]),
        learning_rate_init=manifest.learning_rate,
        max_iter=manifest.epochs,
        random_state=manifest.seed,
        early_stopping=False,
        verbose=False,
    )
    reg.fit(x_train.astype(np.float64), y_train.astype(np.float64))
    with pkl_path.open("wb") as f:
        pickle.dump(reg, f)
    import sklearn

    write_complete(PLANET, model.id, REQUIRED, framework_version=sklearn.__version__)
    return reg, False


def handler(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
    host: str = DEFAULT_HOST,
    skip_loom: bool = False,
):
    reg, skipped = train_or_load(
        model=model, manifest=manifest, data=data, models_dir=models_dir
    )
    _, _, x_test, _ = slice_model_inputs(data, model.input_dim, model_output_dim(model))
    out = reg.predict(x_test.astype(np.float64))
    if out.ndim == 1:
        out = out[:, None]
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="sklearn",
            outputs=out.astype(np.float64),
            artifact_paths=[str(model_dir(PLANET, model.id) / "model.pkl")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            net=reg,
            extractor="sklearn",
        )
        if loom is not None:
            results.append(loom)
    return results


if __name__ == "__main__":
    import sklearn

    raise SystemExit(run_planet(PLANET, sklearn.__version__, handler))
