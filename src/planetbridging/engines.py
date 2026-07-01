"""Stream live AI engine weights into Loom for every bedrock layer type.

Primary API — all 13 volumetric layer types, all POC planets
(pytorch, tensorflow, jax; sklearn on dense).

    from planetbridging import engines

    result = engines.stream("cnn1", "pytorch")
    ladder = engines.ladder("layernorm", "pytorch", try_welvet=True)
    all_results = engines.stream_all_bedrocks("pytorch")
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._binary import repo_root
from .bedrock_ladder import LadderResult, run_bedrock_ladder
from .bedrocks import BEDROCK_IDS, BEDROCK_PLANETS, BEDROCKS, STREAM_MODEL_IDS
from .compare import compare_outputs, diff_label
from .poc_engine import run_engine_handler
from .welvet_infer import try_infer_bedrock_entity

_PLANET_MODULES = {
    "pytorch": "torch",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "sklearn": "sklearn",
}


def available_planets(bedrock: str) -> tuple[str, ...]:
    """Planets enabled for a bedrock in POC manifests."""
    return BEDROCK_PLANETS.get(bedrock.lower(), ("pytorch", "tensorflow", "jax"))


def installed_planets(bedrock: str | None = None) -> tuple[str, ...]:
    """Planets with importable deps (optionally filtered by bedrock)."""
    candidates = available_planets(bedrock) if bedrock else tuple(_PLANET_MODULES)
    return tuple(p for p in candidates if _planet_available(p))


def _planet_available(planet: str) -> bool:
    mod = _PLANET_MODULES.get(planet, planet)
    return importlib.util.find_spec(mod) is not None


@dataclass(frozen=True)
class EngineStreamResult:
    """Native engine forward + loom-stream + optional welvet reload."""

    bedrock: str
    planet: str
    model_id: str
    layer_count: int
    entity_path: str
    native: np.ndarray
    loom_stream: np.ndarray
    native_vs_loom: str
    welvet: np.ndarray | None = None
    native_vs_welvet: str | None = None
    loom_vs_welvet: str | None = None
    welvet_note: str | None = None


def _variants_to_result(
    bedrock: str,
    planet: str,
    model_id: str,
    variants: list[Any],
    *,
    layer_count: int,
    try_welvet: bool = False,
    welvet_inputs: np.ndarray | None = None,
    output_dim: int | None = None,
    input_dim: int | None = None,
) -> EngineStreamResult:
    native_v = next(v for v in variants if v.stage == "native")
    loom_v = next((v for v in variants if v.stage == "loom"), None)
    if loom_v is None:
        raise RuntimeError(f"{bedrock}/{planet}: loom stream variant missing")

    native = np.asarray(native_v.outputs, dtype=np.float64)
    loom = np.asarray(loom_v.outputs, dtype=np.float64)
    n_max, _, n_exact = compare_outputs(native, loom)
    entity = loom_v.artifact_paths[0] if loom_v.artifact_paths else ""

    welvet_out = None
    native_vs_welvet = None
    loom_vs_welvet = None
    welvet_note = None
    if try_welvet and welvet_inputs is not None and output_dim is not None:
        welvet_out = try_infer_bedrock_entity(
            bedrock, entity, welvet_inputs, output_dim=output_dim, input_dim=input_dim
        )
        if welvet_out is None:
            welvet_note = "welvet reload unavailable for this bedrock"
        else:
            w_max, _, w_ex = compare_outputs(native[: len(welvet_out)], welvet_out)
            lw_max, _, lw_ex = compare_outputs(loom[: len(welvet_out)], welvet_out)
            native_vs_welvet = diff_label(w_max, exact=w_ex)
            loom_vs_welvet = diff_label(lw_max, exact=lw_ex)

    return EngineStreamResult(
        bedrock=bedrock,
        planet=planet,
        model_id=model_id,
        layer_count=layer_count,
        entity_path=entity,
        native=native,
        loom_stream=loom,
        native_vs_loom=diff_label(n_max, exact=n_exact),
        welvet=welvet_out,
        native_vs_welvet=native_vs_welvet,
        loom_vs_welvet=loom_vs_welvet,
        welvet_note=welvet_note,
    )


def _welvet_fixture_inputs(
    bedrock: str,
    root: Path,
    model_id: str,
    *,
    max_samples: int = 5,
) -> tuple[np.ndarray | None, int | None, int | None]:
    from .bedrock_ladder import _fixture_inputs
    from .bedrock_smoke import _load_bedrock_module
    from ._fixtures import load_fixture_arrays

    info = BEDROCKS[bedrock]
    manifest_mod = _load_bedrock_module(root, bedrock, "manifest")
    fixtures_mod = _load_bedrock_module(root, bedrock, "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    data = load_fixture_arrays(bedrock, root, info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    x_test, input_dim = _fixture_inputs(bedrock, root, model, manifest, fixtures_mod, data, out_dim)
    n = min(max_samples, len(x_test))
    od = out_dim
    if bedrock == "dense":
        layers = getattr(model, "layers", ())
        od = int(layers[-1].units) if layers else out_dim
    return x_test[:n], od, input_dim


def native_output_dim_from_model(model: Any) -> int:
    layers = getattr(model, "layers", None)
    if layers:
        return int(layers[-1].units)
    return int(getattr(model, "input_dim", 8))


def stream(
    bedrock: str,
    planet: str,
    *,
    model_id: str | None = None,
    root: Path | None = None,
    binary: Path | None = None,
    out_dir: Path | None = None,
    try_welvet: bool = False,
    max_samples: int = 5,
) -> EngineStreamResult:
    """Stream a live AI engine model into Loom .entity; compare native vs loom."""
    bedrock = bedrock.lower()
    planet = planet.lower()
    if bedrock not in BEDROCKS:
        raise ValueError(f"unknown bedrock {bedrock!r}")
    if planet not in available_planets(bedrock):
        raise ValueError(f"planet {planet!r} not enabled for bedrock {bedrock}")
    if not _planet_available(planet):
        raise ImportError(f"{planet} dependencies not installed")

    root = Path(root) if root else repo_root()
    mid = model_id or STREAM_MODEL_IDS[bedrock]
    info = BEDROCKS[bedrock]
    entity_dir = out_dir or root / ".planetbridging" / "stream" / bedrock / planet

    variants = run_engine_handler(
        bedrock,
        planet,
        model_id=mid,
        root=root,
        binary=binary,
        out_dir=entity_dir,
    )

    welvet_inputs = output_dim = input_dim = None
    if try_welvet:
        welvet_inputs, output_dim, input_dim = _welvet_fixture_inputs(
            bedrock, root, mid, max_samples=max_samples
        )

    return _variants_to_result(
        bedrock,
        planet,
        mid,
        variants,
        layer_count=info.layer_count,
        try_welvet=try_welvet,
        welvet_inputs=welvet_inputs,
        output_dim=output_dim,
        input_dim=input_dim,
    )


def ladder(bedrock: str, planet: str = "numpy", **kwargs: Any) -> LadderResult:
    """Compare ladder: numpy reference or live engine → loom-stream → welvet."""
    if planet == "numpy":
        return run_bedrock_ladder(bedrock, **kwargs)
    r = stream(bedrock, planet, try_welvet=kwargs.get("try_welvet", False), **kwargs)
    return LadderResult(
        bedrock=r.bedrock,
        model_id=r.model_id,
        planet=r.planet,
        layer_count=r.layer_count,
        entity_path=r.entity_path,
        native=r.native,
        loom_stream=r.loom_stream,
        welvet=r.welvet,
        native_vs_loom=r.native_vs_loom,
        native_vs_welvet=r.native_vs_welvet,
        loom_vs_welvet=r.loom_vs_welvet,
        welvet_note=r.welvet_note,
    )


def stream_all_bedrocks(
    planet: str,
    *,
    root: Path | None = None,
    binary: Path | None = None,
    try_welvet: bool = False,
) -> list[EngineStreamResult]:
    """Stream every bedrock layer type from one AI engine."""
    if not _planet_available(planet):
        raise ImportError(f"{planet} dependencies not installed")
    results: list[EngineStreamResult] = []
    for bid in BEDROCK_IDS:
        if planet not in available_planets(bid):
            continue
        try:
            results.append(stream(bid, planet, root=root, binary=binary, try_welvet=try_welvet))
        except Exception as exc:
            results.append(
                EngineStreamResult(
                    bedrock=bid,
                    planet=planet,
                    model_id=STREAM_MODEL_IDS[bid],
                    layer_count=0,
                    entity_path=str(exc),
                    native=np.zeros((0, 0)),
                    loom_stream=np.zeros((0, 0)),
                    native_vs_loom="ERROR",
                    welvet_note=str(exc),
                )
            )
    return results


def stream_all_planets(
    bedrock: str,
    *,
    root: Path | None = None,
    binary: Path | None = None,
    model_id: str | None = None,
    try_welvet: bool = False,
) -> list[EngineStreamResult]:
    """Stream one bedrock from every installed AI engine."""
    results: list[EngineStreamResult] = []
    for planet in available_planets(bedrock):
        if not _planet_available(planet):
            continue
        try:
            results.append(
                stream(
                    bedrock,
                    planet,
                    model_id=model_id,
                    root=root,
                    binary=binary,
                    try_welvet=try_welvet,
                )
            )
        except Exception as exc:
            results.append(
                EngineStreamResult(
                    bedrock=bedrock,
                    planet=planet,
                    model_id=model_id or STREAM_MODEL_IDS[bedrock],
                    layer_count=0,
                    entity_path=str(exc),
                    native=np.zeros((0, 0)),
                    loom_stream=np.zeros((0, 0)),
                    native_vs_loom="ERROR",
                    welvet_note=str(exc),
                )
            )
    return results
