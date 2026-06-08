#!/usr/bin/env python3
"""PaddlePaddle dense bedrock."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import LayerSpec, Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "paddle"
REQUIRED = ["model.pdparams"]


def layer_bias(layer: LayerSpec, model: ModelSpec) -> bool:
    if layer.bias is not None:
        return layer.bias
    return model.bias


class MLP(nn.Layer):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.layers = nn.LayerList()
        in_dim = spec.input_dim
        for layer in spec.layers:
            self.layers.append(
                nn.Linear(in_dim, layer.units, bias_attr=layer_bias(layer, spec))
            )
            in_dim = layer.units
        self.spec = spec

    def forward(self, x):
        for i, layer in enumerate(self.spec.layers):
            x = self.layers[i](x)
            act = layer.activation.lower()
            if act == "relu":
                x = nn.functional.relu(x)
            elif act == "tanh":
                x = nn.functional.tanh(x)
            elif act == "sigmoid":
                x = nn.functional.sigmoid(x)
            elif act != "linear":
                raise ValueError(f"unsupported activation: {layer.activation}")
        return x


def train_or_load(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
):
    paddle.seed(manifest.seed)
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "model.pdparams"

    net = MLP(model)
    if is_complete(PLANET, model.id, REQUIRED):
        state = paddle.load(str(params_path))
        net.set_state_dict(state)
        net.eval()
        return net, True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model.input_dim, out_dim)
    xt = paddle.to_tensor(x_train.astype(np.float32))
    yt = paddle.to_tensor(y_train.astype(np.float32))

    opt = paddle.optimizer.Adam(
        learning_rate=manifest.learning_rate, parameters=net.parameters()
    )
    net.train()
    n = xt.shape[0]
    for _ in range(manifest.epochs):
        perm = paddle.randperm(n)
        for start in range(0, n, manifest.batch_size):
            idx = perm[start : start + manifest.batch_size]
            opt.clear_grad()
            pred = net(xt[idx])
            loss = paddle.mean((pred - yt[idx]) ** 2)
            loss.backward()
            opt.step()

    net.eval()
    paddle.save(net.state_dict(), str(params_path))
    write_complete(PLANET, model.id, REQUIRED, framework_version=paddle.__version__)
    return net, False


def handler(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
    host: str = DEFAULT_HOST,
    skip_loom: bool = False,
):
    net, skipped = train_or_load(
        model=model, manifest=manifest, data=data, models_dir=models_dir
    )
    _, _, x_test, _ = slice_model_inputs(data, model.input_dim, model_output_dim(model))
    with paddle.no_grad():
        out = net(paddle.to_tensor(x_test.astype(np.float32))).numpy()
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="paddle",
            outputs=out.astype(np.float64),
            artifact_paths=[str(model_dir(PLANET, model.id) / "model.pdparams")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            net=net,
            extractor="paddle",
        )
        if loom is not None:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, paddle.__version__, handler))
