#!/usr/bin/env python3
"""TensorFlow CNN3 bedrock."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "tensorflow"
REQUIRED = ["model.keras", "saved_model"]


def _keras_padding(padding: int) -> str:
    return "valid" if padding == 0 else "same"


def build_model(model: ModelSpec) -> tf.keras.Model:
    inp = tf.keras.Input(
        shape=(model.input_channels, model.depth, model.height, model.width),
        name="input",
    )
    x = inp
    for layer in model.layers:
        act = None if layer.activation == "linear" else layer.activation
        x = tf.keras.layers.Conv3D(
            layer.filters,
            layer.kernel_size,
            strides=layer.stride,
            padding=_keras_padding(layer.padding),
            use_bias=False,
            activation=act,
            data_format="channels_first",
        )(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=model.id)


def flatten_predict(pred: np.ndarray, output_dim: int) -> np.ndarray:
    return pred.reshape(pred.shape[0], -1)[:, :output_dim].astype(np.float64)


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[tf.keras.Model, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras_path = out_dir / "model.keras"
    saved_path = out_dir / "saved_model"
    if is_complete(PLANET, model.id, REQUIRED):
        return tf.keras.models.load_model(keras_path), True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model, out_dim)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    tf.keras.utils.set_random_seed(manifest.seed)
    net = build_model(model)
    net.compile(optimizer=tf.keras.optimizers.Adam(manifest.learning_rate), loss="mse")
    net.fit(x_train, y_train, epochs=manifest.epochs, batch_size=manifest.batch_size, verbose=0)
    net.save(keras_path)
    net.export(saved_path)
    write_complete(PLANET, model.id, REQUIRED, tf.__version__)
    return net, False


def infer_saved(saved_path: Path, x: np.ndarray) -> np.ndarray:
    inp = tf.keras.Input(shape=(x.shape[1], x.shape[2], x.shape[3], x.shape[4]), name="input")
    sm = tf.keras.layers.TFSMLayer(str(saved_path), call_endpoint="serve")
    m = tf.keras.Model(inputs=inp, outputs=sm(inp))
    return m.predict(x, verbose=0)


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    net, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)
    x = x_test.astype(np.float32)

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="keras",
            outputs=flatten_predict(net.predict(x, verbose=0), out_dim),
            artifact_paths=[str(out_dir / "model.keras")],
            train_skipped=skipped,
        )
    ]
    saved_path = out_dir / "saved_model"
    if saved_path.exists():
        results.append(
            VariantResult(
                planet=PLANET,
                stage="export",
                format="saved_model",
                outputs=flatten_predict(infer_saved(saved_path, x), out_dim),
                artifact_paths=[str(saved_path)],
                train_skipped=True,
            )
        )
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host, planet=PLANET, model=model, manifest=manifest, net=net, extractor="tensorflow"
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, tf.__version__, handler))
