"""Tests for planetbridging PyPI package (no HTTP, uses loom-stream CLI)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetbridging._binary import default_binary_path, repo_root
from planetbridging.compare import compare_outputs, diff_label
from planetbridging.layers.dense import LayerSpec, layers_from_specs
from planetbridging.stream import stream_dense

LOOM_STREAM = default_binary_path()


@pytest.fixture(scope="session")
def loom_stream_binary() -> Path:
    if not LOOM_STREAM.is_file():
        pytest.skip(f"loom-stream not built: {LOOM_STREAM}")
    return LOOM_STREAM


def test_compare_exact():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    max_d, mean_d, exact = compare_outputs(a, a.copy())
    assert exact
    assert max_d == 0.0
    assert diff_label(max_d, exact=exact) == "EXACT"


def test_compare_pass():
    a = np.zeros((2, 2))
    b = a + 1e-7
    max_d, _, exact = compare_outputs(a, b)
    assert not exact
    assert diff_label(max_d) == "PASS"


def test_loom_stream_cli_synthetic(loom_stream_binary: Path, tmp_path: Path):
    payload = {
        "planet": "test",
        "model_id": "synthetic",
        "fixture_version": "dense_bedrock_v2",
        "input_dim": 4,
        "layers": [
            {
                "index": 0,
                "input_dim": 4,
                "output_dim": 2,
                "activation": "relu",
                "weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "bias": [0.01, 0.02],
            },
            {
                "index": 1,
                "input_dim": 2,
                "output_dim": 1,
                "activation": "linear",
                "weights": [1.0, 0.0],
                "bias": [0.5],
            },
        ],
    }
    envelope = {
        "bedrock": "dense",
        "root": str(repo_root()),
        "output_path": str(tmp_path / "synthetic.entity"),
        "fixture_version": "dense_bedrock_v2",
    }
    body = {**envelope, "payload": payload}
    proc = subprocess.run(
        [str(loom_stream_binary)],
        input=json.dumps(body).encode(),
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    resp = json.loads(proc.stdout)
    assert resp["status"] == "ok"
    assert Path(resp["entity_path"]).is_file()
    assert resp["sample_count"] == 100


def test_stream_dense_python_api(loom_stream_binary: Path, tmp_path: Path):
    specs = (LayerSpec(2, "relu"), LayerSpec(1, "linear"))
    layers = layers_from_specs(
        input_dim=4,
        specs=specs,
        kernels=[
            np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ],
        biases=[np.array([0.01, 0.02], dtype=np.float32), np.array([0.5], dtype=np.float32)],
    )
    result = stream_dense(
        planet="test",
        model_id="api_synthetic",
        layers=layers,
        input_dim=4,
        fixture_version="dense_bedrock_v2",
        output_path=tmp_path / "api.entity",
        root=repo_root(),
        binary=loom_stream_binary,
    )
    assert result.entity_path
    assert Path(result.entity_path).is_file()
    assert result.outputs.shape[0] == 100


@pytest.mark.pytorch
def test_absorb_pytorch(loom_stream_binary: Path, tmp_path: Path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from planetbridging import absorb

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    net.eval()
    x = np.random.randn(5, 4).astype(np.float32)
    with torch.no_grad():
        native = net(torch.from_numpy(x)).numpy()

    result = absorb.pytorch(
        net,
        model_id="pt_test",
        input_dim=4,
        layer_units=(2, 1),
        inputs=x,
        native_outputs=native,
        output_path=tmp_path / "pt.entity",
        root=repo_root(),
        binary=loom_stream_binary,
    )
    assert Path(result.entity_path).is_file()
    assert result.layer_count == 2
    assert result.outputs.shape[0] == len(x)
