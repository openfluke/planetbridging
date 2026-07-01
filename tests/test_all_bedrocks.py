"""Tests: all bedrock layer types stream via live AI engines (multi-layer models)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetbridging._binary import default_binary_path, repo_root
from planetbridging.bedrocks import BEDROCK_IDS, STREAM_LAYER_COUNTS, STREAM_MODEL_IDS
from planetbridging import engines

LOOM_STREAM = default_binary_path()


@pytest.fixture(scope="session")
def loom_stream_binary() -> Path:
    if not LOOM_STREAM.is_file():
        pytest.skip(f"loom-stream not built: {LOOM_STREAM}")
    return LOOM_STREAM


MIXER_POC_TOLERANCE = 5e-5


@pytest.mark.pytorch
@pytest.mark.parametrize("bedrock", BEDROCK_IDS)
def test_stream_bedrock_pytorch(loom_stream_binary: Path, bedrock: str, tmp_path: Path):
    torch = pytest.importorskip("torch")
    _ = torch  # noqa: F841
    if "pytorch" not in engines.available_planets(bedrock):
        pytest.skip(f"pytorch not enabled for {bedrock}")

    result = engines.stream(
        bedrock,
        "pytorch",
        root=repo_root(),
        binary=loom_stream_binary,
        out_dir=tmp_path / bedrock,
    )
    tol_label = result.native_vs_loom
    if bedrock == "mixer" and tol_label == "DIFF":
        from planetbridging.compare import compare_outputs

        max_d, _, _ = compare_outputs(result.native, result.loom_stream)
        assert max_d < MIXER_POC_TOLERANCE, f"mixer max diff {max_d}"
    else:
        assert tol_label in ("PASS", "EXACT"), (
            f"{bedrock}: {tol_label} entity={result.entity_path}"
        )
    assert result.layer_count == STREAM_LAYER_COUNTS[bedrock]
    assert result.model_id == STREAM_MODEL_IDS[bedrock]
    assert Path(result.entity_path).is_file()


@pytest.mark.pytorch
def test_stream_all_bedrocks_pytorch(loom_stream_binary: Path, tmp_path: Path):
    pytest.importorskip("torch")
    results = engines.stream_all_bedrocks(
        "pytorch",
        root=repo_root(),
        binary=loom_stream_binary,
    )
    assert len(results) == len(BEDROCK_IDS)
    ok = sum(
        1
        for r in results
        if r.native_vs_loom in ("PASS", "EXACT")
        or (r.bedrock == "mixer" and r.native_vs_loom == "DIFF")
    )
    assert ok == len(BEDROCK_IDS), f"failures: {[(r.bedrock, r.native_vs_loom) for r in results if r.native_vs_loom not in ('PASS','EXACT') and r.bedrock != 'mixer']}"


@pytest.mark.pytorch
@pytest.mark.parametrize(
    "bedrock,expected_layers",
    [
        ("dense", 4),
        ("cnn1", 2),
        ("cnn2", 2),
        ("cnn3", 2),
        ("mixer", 16),
    ],
)
def test_multi_layer_models(loom_stream_binary: Path, bedrock: str, expected_layers: int, tmp_path: Path):
    pytest.importorskip("torch")
    result = engines.stream(
        bedrock,
        "pytorch",
        root=repo_root(),
        binary=loom_stream_binary,
        out_dir=tmp_path,
    )
    assert result.layer_count == expected_layers
    if bedrock == "mixer":
        from planetbridging.compare import compare_outputs

        max_d, _, _ = compare_outputs(result.native, result.loom_stream)
        assert max_d < MIXER_POC_TOLERANCE
    else:
        assert result.native_vs_loom in ("PASS", "EXACT")
