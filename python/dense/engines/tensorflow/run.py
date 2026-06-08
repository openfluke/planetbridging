#!/usr/bin/env python3
"""TensorFlow dense bedrock: native Keras vs SavedModel export."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import LayerSpec, Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "tensorflow"
REQUIRED = ["model.keras", "saved_model"]


def layer_bias(layer: LayerSpec, model: ModelSpec) -> bool:
    if layer.bias is not None:
        return layer.bias
    return model.bias


def build_model(model: ModelSpec) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(model.input_dim,), name="input")
    x = inp
    for layer in model.layers:
        x = tf.keras.layers.Dense(
            layer.units,
            activation=None if layer.activation == "linear" else layer.activation,
            use_bias=layer_bias(layer, model),
        )(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=model.id)


def train_or_load(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
) -> tuple[tf.keras.Model, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras_path = out_dir / "model.keras"
    saved_path = out_dir / "saved_model"

    if is_complete(PLANET, model.id, REQUIRED):
        return tf.keras.models.load_model(keras_path), True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model.input_dim, out_dim)
    tf.keras.utils.set_random_seed(manifest.seed)
    net = build_model(model)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=manifest.learning_rate),
        loss="mse",
    )
    net.fit(
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        epochs=manifest.epochs,
        batch_size=manifest.batch_size,
        verbose=0,
    )
    net.save(keras_path)
    net.export(saved_path)
    write_complete(PLANET, model.id, REQUIRED, framework_version=tf.__version__)
    return net, False


def infer_saved_model(saved_path: Path, input_dim: int, x: np.ndarray) -> np.ndarray:
    """Keras 3: reload exported SavedModel via TFSMLayer (not load_model)."""
    inp = tf.keras.Input(shape=(input_dim,), name="input")
    sm_layer = tf.keras.layers.TFSMLayer(str(saved_path), call_endpoint="serve")
    export_model = tf.keras.Model(inputs=inp, outputs=sm_layer(inp))
    return export_model.predict(x, verbose=0).astype(np.float64)


def handler(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
    host: str = DEFAULT_HOST,
    skip_loom: bool = False,
):
    net, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    _, _, x_test, _ = slice_model_inputs(data, model.input_dim, model_output_dim(model))
    x = x_test.astype(np.float32)

    native_out = net.predict(x, verbose=0).astype(np.float64)
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="keras",
            outputs=native_out,
            artifact_paths=[str(out_dir / "model.keras")],
            train_skipped=skipped,
        )
    ]

    saved_path = out_dir / "saved_model"
    if saved_path.exists():
        export_out = infer_saved_model(saved_path, model.input_dim, x)
        results.append(
            VariantResult(
                planet=PLANET,
                stage="export",
                format="saved_model",
                outputs=export_out,
                artifact_paths=[str(saved_path)],
                train_skipped=True,
            )
        )

    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            net=net,
            extractor="keras",
        )
        if loom is not None:
            results.append(loom)

    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, tf.__version__, handler))
