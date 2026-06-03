#!/usr/bin/env python3
"""ONNX Runtime inference on PyTorch-exported dense models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_engine  # noqa: E402

ENGINE = "onnxruntime"
SOURCE_ENGINE = "pytorch"


def handler(*, model: ModelSpec, manifest: Manifest, data: dict[str, np.ndarray], models_dir: Path):
    onnx_path = models_dir / SOURCE_ENGINE / model.id / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"missing {onnx_path}; run pytorch engine first to export ONNX"
        )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    _, _, x_test, _ = slice_model_inputs(data, model.input_dim, model_output_dim(model))
    out = sess.run(None, {input_name: x_test.astype(np.float32)})[0]
    return ["model.onnx"], out.astype(np.float64), True


if __name__ == "__main__":
    raise SystemExit(run_engine(ENGINE, ort.__version__, handler))
