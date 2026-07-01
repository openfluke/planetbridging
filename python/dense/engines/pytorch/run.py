#!/usr/bin/env python3
"""PyTorch dense bedrock: native vs ONNX + Safetensors exports."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import LayerSpec, Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "pytorch"
REQUIRED = ["model.pt"]
OPTIONAL = ["model.safetensors", "model.onnx"]


def layer_bias(layer: LayerSpec, model: ModelSpec) -> bool:
    if layer.bias is not None:
        return layer.bias
    return model.bias


def build_model(model: ModelSpec) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = model.input_dim
    for layer in model.layers:
        use_bias = layer_bias(layer, model)
        layers.append(nn.Linear(in_dim, layer.units, bias=use_bias))
        act = layer.activation.lower()
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "tanh":
            layers.append(nn.Tanh())
        elif act == "sigmoid":
            layers.append(nn.Sigmoid())
        elif act != "linear":
            raise ValueError(f"unsupported activation: {layer.activation}")
        in_dim = layer.units
    return nn.Sequential(*layers)


def forward(net: nn.Sequential, x_test: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = net(torch.from_numpy(x_test.astype(np.float32))).cpu().numpy()
    return out.astype(np.float64)


def train_or_load(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
) -> tuple[nn.Sequential, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = build_model(model)
    if is_complete(PLANET, model.id, REQUIRED):
        state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        return net, True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model.input_dim, out_dim)
    x_t = torch.from_numpy(x_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))

    torch.manual_seed(manifest.seed)
    opt = torch.optim.Adam(net.parameters(), lr=manifest.learning_rate)
    loss_fn = nn.MSELoss()

    net.train()
    n = x_t.shape[0]
    for _ in range(manifest.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, manifest.batch_size):
            idx = perm[start : start + manifest.batch_size]
            opt.zero_grad()
            pred = net(x_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            opt.step()

    net.eval()
    torch.save(net.state_dict(), out_dir / "model.pt")

    try:
        from safetensors.torch import save_file

        save_file(net.state_dict(), out_dir / "model.safetensors")
    except Exception as exc:
        print(f"[pytorch] safetensors export skipped for {model.id}: {exc}")

    try:
        dummy = torch.zeros(1, model.input_dim, dtype=torch.float32)
        torch.onnx.export(
            net,
            dummy,
            out_dir / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
    except Exception as exc:
        print(f"[pytorch] onnx export skipped for {model.id}: {exc}")

    write_complete(
        PLANET,
        model.id,
        [p for p in REQUIRED + OPTIONAL if (out_dir / p).exists()],
        framework_version=torch.__version__,
    )
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
    net, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    _, _, x_test, _ = slice_model_inputs(data, model.input_dim, model_output_dim(model))

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="pytorch",
            outputs=forward(net, x_test),
            artifact_paths=[str(out_dir / "model.pt")],
            train_skipped=skipped,
        )
    ]

    st_path = out_dir / "model.safetensors"
    if st_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            print(f"[pytorch] safetensors reload skipped for {model.id}: {exc}")
        else:
            reloaded = build_model(model)
            reloaded.load_state_dict(load_file(st_path))
            reloaded.eval()
            results.append(
                VariantResult(
                    planet=PLANET,
                    stage="export",
                    format="safetensors",
                    outputs=forward(reloaded, x_test),
                    artifact_paths=[str(st_path)],
                    train_skipped=True,
                )
            )

    onnx_path = out_dir / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort
        except ImportError:
            print(
                f"[pytorch] {model.id}: onnx present but onnxruntime missing; "
                "run: conda run -n pb-dense-pytorch pip install onnxruntime"
            )
        else:
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            out = sess.run(None, {input_name: x_test.astype(np.float32)})[0]
            results.append(
                VariantResult(
                    planet=PLANET,
                    stage="export",
                    format="onnx",
                    outputs=out.astype(np.float64),
                    artifact_paths=[str(onnx_path)],
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
            extractor="pytorch",
        )
        if loom is not None:
            results.append(loom)

    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, torch.__version__, handler))
