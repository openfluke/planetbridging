"""Mixer stream integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MIXER = ROOT / "python" / "mixer"
sys.path.insert(0, str(MIXER))

from planetbridging._binary import default_binary_path, repo_root
from planetbridging.stream import stream_mixer

from shared.layer_stream import layers_json_from_weights
from shared.mixer_forward import loom_mixer_forward_batch
from shared.weights import init_mixer_weights

LOOM_STREAM = default_binary_path()
FIXTURE = ROOT / "python" / "mixer" / "fixtures" / "mixer_bedrock_v1.npz"


@pytest.fixture(scope="session")
def loom_stream_binary() -> Path:
    if not LOOM_STREAM.is_file():
        pytest.skip(f"loom-stream not built: {LOOM_STREAM}")
    return LOOM_STREAM


def test_mixer_stream_v1(loom_stream_binary: Path, tmp_path: Path):
    if not FIXTURE.is_file():
        pytest.skip("mixer fixture missing")

    weights = init_mixer_weights(42 + sum(ord(c) for c in "mixer_all_v1"))
    data = np.load(FIXTURE)
    x_test = data["x_test"]
    native = loom_mixer_forward_batch(x_test, weights, output_dim=8)
    layers = layers_json_from_weights(weights, "mixer_all_v1")

    result = stream_mixer(
        planet="test",
        model_id="mixer_all_v1",
        layers=layers,
        output_dim=8,
        output_path=tmp_path / "mixer_v1.entity",
        root=repo_root(),
        native_outputs=native,
        binary=loom_stream_binary,
    )

    assert len(layers) == 10
    assert Path(result.entity_path).is_file()
    assert result.compare_label in ("EXACT", "PASS", "DIFF")
    assert result.outputs.shape == native.shape
