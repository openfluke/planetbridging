"""Full compare ladder per bedrock: native → loom-stream → welvet reload."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bedrock_smoke import (
    _SMOKE_RUNNERS,
    _load_bedrock_module,
    _load_fixture_npz,
    _repo,
)
from .bedrocks import BEDROCK_IDS, BEDROCKS, BedrockInfo
from .compare import compare_outputs, diff_label
from .stream import StreamResult
from .welvet_infer import (
    WELVET_RELOAD_SKIP,
    try_infer_bedrock_entity,
)


@dataclass(frozen=True)
class LadderResult:
    bedrock: str
    model_id: str
    planet: str
    layer_count: int
    entity_path: str
    native: np.ndarray
    loom_stream: np.ndarray
    welvet: np.ndarray | None
    native_vs_loom: str
    native_vs_welvet: str | None
    loom_vs_welvet: str | None
    welvet_note: str | None = None


def _fixture_inputs(
    bedrock: str,
    root: Path,
    model: Any,
    manifest: Any,
    fixtures_mod: Any,
    data: dict[str, np.ndarray],
    out_dim: int,
) -> tuple[np.ndarray, int | None]:
    """Return (x_test, input_dim_for_dense)."""
    if bedrock == "dense":
        x = fixtures_mod.ensure_fixtures(manifest)["x_test"][:, : model.input_dim]
        return x, int(model.input_dim)
    if bedrock == "mixer":
        return data["x_test"], None
    if bedrock == "residual":
        _, _, _, xm, xs, _ = fixtures_mod.slice_model_inputs(data, model, out_dim)
        # Residual infer uses paired tensors; welvet skip uses main only for display.
        return xm, None
    sliced = fixtures_mod.slice_model_inputs(data, model, out_dim)
    return sliced[2], None


def run_bedrock_ladder(
    bedrock: str,
    *,
    root: Path | None = None,
    model_id: str | None = None,
    binary: Path | None = None,
    out_dir: Path | None = None,
    max_samples: int = 5,
    try_welvet: bool = True,
    planet: str = "numpy",
) -> LadderResult:
    """Stream one bedrock model and compare native / loom-stream / welvet reload."""
    bedrock = bedrock.lower()
    if bedrock not in BEDROCKS:
        raise ValueError(f"unknown bedrock {bedrock!r}")

    info = BEDROCKS[bedrock]
    mid = model_id or info.smoke_model_id
    root = _repo(root)
    runner = _SMOKE_RUNNERS.get(bedrock)
    if runner is None:
        raise ValueError(f"no smoke runner for {bedrock}")

    stream: StreamResult = runner(root, mid, info, binary=binary, out_dir=out_dir)
    native = stream.native_reference
    if native is None:
        raise RuntimeError(f"{bedrock}: stream result missing native_reference")

    manifest_mod = _load_bedrock_module(root, bedrock, "manifest")
    fixtures_mod = _load_bedrock_module(root, bedrock, "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == mid)
    data = _load_fixture_npz(root, bedrock, info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    x_test, input_dim = _fixture_inputs(bedrock, root, model, manifest, fixtures_mod, data, out_dim)

    n = min(max_samples, len(native), len(stream.outputs))
    native_s = native[:n]
    loom_s = stream.outputs[:n]
    x_s = x_test[:n]

    n_max, _, n_exact = compare_outputs(native_s, loom_s)
    native_vs_loom = diff_label(n_max, exact=n_exact)

    welvet_out: np.ndarray | None = None
    native_vs_welvet: str | None = None
    loom_vs_welvet: str | None = None
    welvet_note: str | None = None

    if bedrock in WELVET_RELOAD_SKIP:
        welvet_note = "welvet reload skipped (C ABI gap on this bedrock)"
    elif not try_welvet:
        welvet_note = "welvet not requested"
    else:
        welvet_out = try_infer_bedrock_entity(
            bedrock,
            stream.entity_path,
            x_s,
            output_dim=out_dim if bedrock != "dense" else native_s.shape[-1],
            input_dim=input_dim,
        )
        if welvet_out is None:
            welvet_note = "welvet reload failed or unavailable"
        else:
            w_max, _, w_exact = compare_outputs(native_s, welvet_out)
            lw_max, _, lw_exact = compare_outputs(loom_s, welvet_out)
            native_vs_welvet = diff_label(w_max, exact=w_exact)
            loom_vs_welvet = diff_label(lw_max, exact=lw_exact)

    return LadderResult(
        bedrock=bedrock,
        model_id=mid,
        planet=planet,
        layer_count=stream.layer_count,
        entity_path=stream.entity_path,
        native=native_s,
        loom_stream=loom_s,
        welvet=welvet_out,
        native_vs_loom=native_vs_loom,
        native_vs_welvet=native_vs_welvet,
        loom_vs_welvet=loom_vs_welvet,
        welvet_note=welvet_note,
    )


def run_all_bedrock_ladders(**kwargs: Any) -> list[LadderResult]:
    results: list[LadderResult] = []
    for bid in BEDROCK_IDS:
        try:
            results.append(run_bedrock_ladder(bid, **kwargs))
        except Exception as exc:
            info = BEDROCKS[bid]
            results.append(
                LadderResult(
                    bedrock=bid,
                    model_id=info.smoke_model_id,
                    planet=str(kwargs.get("planet", "numpy")),
                    layer_count=0,
                    entity_path=str(exc),
                    native=np.zeros((0, 0)),
                    loom_stream=np.zeros((0, 0)),
                    welvet=None,
                    native_vs_loom="ERROR",
                    native_vs_welvet=None,
                    loom_vs_welvet=None,
                    welvet_note=str(exc),
                )
            )
    return results
