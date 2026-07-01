"""Smoke-test every POC bedrock layer type via loom-stream (no HTTP)."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ._binary import repo_root
from .bedrocks import BEDROCK_IDS, BEDROCKS, BedrockInfo
from .stream import StreamResult, stream_bedrock, stream_dense, stream_mixer


@dataclass(frozen=True)
class SmokeResult:
    bedrock: str
    model_id: str
    layer_count: int
    compare_label: str
    max_abs_diff: float | None
    entity_path: str


def _repo(root: Path | None) -> Path:
    return Path(root) if root else repo_root()


def _bedrock_pkg(root: Path, bedrock: str) -> str:
    """Register python/<bedrock>/shared as an isolated package (avoids shared.* collisions)."""
    shared_dir = root / "python" / bedrock / "shared"
    parent = f"_pb_{bedrock}"
    pkg = f"{parent}.shared"
    if parent not in sys.modules:
        parent_mod = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(parent, loader=None, is_package=True)
        )
        parent_mod.__path__ = [str(root / "python" / bedrock)]  # type: ignore[attr-defined]
        sys.modules[parent] = parent_mod
    if pkg not in sys.modules:
        mod = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(pkg, loader=None, is_package=True)
        )
        mod.__path__ = [str(shared_dir)]  # type: ignore[attr-defined]
        sys.modules[pkg] = mod
        # Pre-load spec.py if present (many modules import from .spec)
        spec_py = shared_dir / "spec.py"
        if spec_py.is_file():
            _load_bedrock_module(root, bedrock, "spec")
    return pkg


def _load_bedrock_module(root: Path, bedrock: str, stem: str):
    """Import one module from python/<bedrock>/shared/<stem>.py."""
    pkg = _bedrock_pkg(root, bedrock)
    name = f"{pkg}.{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = root / "python" / bedrock / "shared" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(root / "python" / bedrock / "shared")]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _add_bedrock_path(root: Path, bedrock: str) -> Path:
    _bedrock_pkg(root, bedrock)
    return root / "python" / bedrock


def _load_manifest(root: Path, bedrock: str) -> dict:
    return yaml.safe_load((root / "python" / bedrock / "manifest.yaml").read_text())


def _model_entry(manifest: dict, model_id: str) -> dict:
    for m in manifest["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(model_id)


def _model_seed(manifest: dict, model_id: str) -> int:
    return int(manifest["seed"]) + sum(ord(c) for c in model_id)


def _load_fixture_npz(root: Path, bedrock: str, fixture_version: str) -> dict[str, np.ndarray]:
    path = root / "python" / bedrock / "fixtures" / f"{fixture_version}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_bedrock_smoke(
    bedrock: str,
    *,
    root: Path | None = None,
    model_id: str | None = None,
    binary: Path | None = None,
    out_dir: Path | None = None,
) -> SmokeResult:
    """Stream one representative model per bedrock and compare native vs loom-stream."""
    bedrock = bedrock.lower()
    if bedrock not in BEDROCKS:
        raise ValueError(f"unknown bedrock {bedrock!r}")
    info = BEDROCKS[bedrock]
    mid = model_id or info.stream_model_id
    root = _repo(root)
    runner = _SMOKE_RUNNERS.get(bedrock)
    if runner is None:
        raise ValueError(f"no smoke runner for {bedrock}")
    result = runner(root, mid, info, binary=binary, out_dir=out_dir)
    return SmokeResult(
        bedrock=bedrock,
        model_id=mid,
        layer_count=result.layer_count,
        compare_label=result.compare_label or "PENDING",
        max_abs_diff=result.max_abs_diff,
        entity_path=result.entity_path,
    )


def run_all_bedrock_smokes(**kwargs: Any) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for bid in BEDROCK_IDS:
        try:
            results.append(run_bedrock_smoke(bid, **kwargs))
        except Exception as exc:
            results.append(
                SmokeResult(
                    bedrock=bid,
                    model_id=BEDROCKS[bid].stream_model_id,
                    layer_count=0,
                    compare_label="ERROR",
                    max_abs_diff=None,
                    entity_path=str(exc),
                )
            )
    return results


def _flatten_out(y: np.ndarray, out_dim: int) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)[:, :out_dim]


def _smoke_dense(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "dense")
    fixtures = _load_bedrock_module(root, "dense", "fixtures")
    manifest_mod = _load_bedrock_module(root, "dense", "manifest")
    extractors = _load_bedrock_module(root, "dense", "extractors")

    import torch
    import torch.nn as nn

    from planetbridging.layers.dense import LayerSpec as PLayerSpec

    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    data = fixtures.ensure_fixtures(manifest)
    x_test = data["x_test"][:, : model.input_dim]

    torch.manual_seed(manifest.seed)
    layers_mod: list[nn.Module] = []
    in_d = model.input_dim
    for spec in model.layers:
        layers_mod.append(nn.Linear(in_d, spec.units))
        if spec.activation == "relu":
            layers_mod.append(nn.ReLU())
        in_d = spec.units
    net = nn.Sequential(*layers_mod)
    net.eval()
    with torch.no_grad():
        native = net(torch.from_numpy(x_test.astype(np.float32))).numpy()

    dense_layers = extractors.extract_pytorch_sequential(net, model)
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "dense"
    out.mkdir(parents=True, exist_ok=True)
    return stream_dense(
        planet="smoke",
        model_id=model_id,
        layers=dense_layers,
        input_dim=model.input_dim,
        fixture_version=info.fixture_version,
        inputs=x_test,
        native_outputs=native,
        output_path=out / f"{model_id}.entity",
        root=root,
        binary=kw.get("binary"),
    )


def _smoke_layernorm(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "layernorm")
    fwd = _load_bedrock_module(root, "layernorm", "layernorm_forward")
    layer_stream = _load_bedrock_module(root, "layernorm", "layer_stream")
    manifest_mod = _load_bedrock_module(root, "layernorm", "manifest")
    fixtures = _load_bedrock_module(root, "layernorm", "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    weights = fwd.init_layernorm_weights(model.dim, seed)
    data = _load_fixture_npz(root, "layernorm", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = fwd.loom_layernorm_forward(x_test, gamma=weights["gamma"], beta=weights["beta"])[:, :out_dim]
    layers = [layer_stream.layer_stream_from_weights(model, weights=weights).to_json_dict()]
    payload = {
        "bedrock": "layernorm", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "dim": model.dim, "seq_len": model.seq_len, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "layernorm"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("layernorm", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_rmsnorm(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "rmsnorm")
    fwd = _load_bedrock_module(root, "rmsnorm", "rmsnorm_forward")
    layer_stream = _load_bedrock_module(root, "rmsnorm", "layer_stream")
    manifest_mod = _load_bedrock_module(root, "rmsnorm", "manifest")
    fixtures = _load_bedrock_module(root, "rmsnorm", "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    weights = fwd.init_rmsnorm_weights(model.dim, seed)
    data = _load_fixture_npz(root, "rmsnorm", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = fwd.loom_rmsnorm_forward(x_test, gamma=weights["gamma"])[:, :out_dim]
    layers = [layer_stream.layer_stream_from_weights(model, weights=weights).to_json_dict()]
    payload = {
        "bedrock": "rmsnorm", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "dim": model.dim, "seq_len": model.seq_len, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "rmsnorm"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("rmsnorm", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_embedding(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "embedding")
    fwd = _load_bedrock_module(root, "embedding", "embedding_forward")
    layer_stream = _load_bedrock_module(root, "embedding", "layer_stream")
    manifest_mod = _load_bedrock_module(root, "embedding", "manifest")
    fixtures = _load_bedrock_module(root, "embedding", "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    weights = fwd.init_embedding_weights(model.vocab_size, model.embed_dim, seed)
    data = _load_fixture_npz(root, "embedding", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = fwd.loom_embedding_forward(x_test, table=weights["table"])[:, :out_dim]
    layers = [layer_stream.layer_stream_from_weights(model, weights=weights).to_json_dict()]
    payload = {
        "bedrock": "embedding", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "vocab_size": model.vocab_size, "seq_len": model.seq_len,
        "embed_dim": model.embed_dim, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "embedding"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("embedding", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_swiglu(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "swiglu")
    fwd = _load_bedrock_module(root, "swiglu", "swiglu_forward")
    layer_stream = _load_bedrock_module(root, "swiglu", "layer_stream")
    manifest_mod = _load_bedrock_module(root, "swiglu", "manifest")
    fixtures = _load_bedrock_module(root, "swiglu", "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    weights = fwd.init_swiglu_weights(model.input_dim, model.intermediate_dim, seed)
    data = _load_fixture_npz(root, "swiglu", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = fwd.loom_swiglu_forward(x_test, **weights)[:, :out_dim]
    layers = [layer_stream.layer_stream_from_weights(model, weights=weights).to_json_dict()]
    payload = {
        "bedrock": "swiglu", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "input_dim": model.input_dim, "intermediate_dim": model.intermediate_dim,
        "seq_len": model.seq_len, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "swiglu"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("swiglu", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_residual(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "residual")
    manifest_mod = _load_bedrock_module(root, "residual", "manifest")
    fixtures = _load_bedrock_module(root, "residual", "fixtures")
    fwd = _load_bedrock_module(root, "residual", "residual_forward")
    layer_stream = _load_bedrock_module(root, "residual", "layer_stream")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    data = _load_fixture_npz(root, "residual", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, _, x_main, x_skip, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = fwd.loom_residual_forward(x_main, x_skip)[:, :out_dim]
    layers = [layer_stream.layer_stream_from_model(model).to_json_dict()]
    payload = {
        "bedrock": "residual", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "dim": model.dim, "seq_len": model.seq_len, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "residual"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("residual", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_recurrent(root: Path, model_id: str, info: BedrockInfo, bedrock: str, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, bedrock)
    fwd_mod = _load_bedrock_module(root, bedrock, f"{bedrock}_forward")
    layer_mod = _load_bedrock_module(root, bedrock, "layer_stream")
    manifest_mod = _load_bedrock_module(root, bedrock, "manifest")
    fixtures = _load_bedrock_module(root, bedrock, "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    data = _load_fixture_npz(root, bedrock, info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)

    if bedrock == "lstm":
        gates = fwd_mod.init_loom_lstm_weights(model.input_size, model.hidden_size, seed)
        native = _flatten_out(
            fwd_mod.loom_lstm_forward_batch(
                x_test, i_weights=gates["i"], f_weights=gates["f"],
                g_weights=gates["g"], o_weights=gates["o"],
                input_size=model.input_size, hidden_size=model.hidden_size,
            ),
            out_dim,
        )
        layer = layer_mod.layer_stream_from_weights(
            model, i_weights=gates["i"], f_weights=gates["f"],
            g_weights=gates["g"], o_weights=gates["o"],
        )
    else:
        w = fwd_mod.init_loom_rnn_weights(model.input_size, model.hidden_size, seed)
        native = _flatten_out(
            fwd_mod.loom_rnn_forward_batch(
                x_test, weights=w, input_size=model.input_size, hidden_size=model.hidden_size,
            ),
            out_dim,
        )
        layer = layer_mod.layer_stream_from_weights(model, weights=w)

    layers = [layer.to_json_dict()]
    payload = {
        "bedrock": bedrock, "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "input_size": model.input_size, "hidden_size": model.hidden_size,
        "seq_len": model.seq_len, "output_dim": out_dim, "layers": layers,
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / bedrock
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock(bedrock, payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_mha(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "mha")
    fwd = _load_bedrock_module(root, "mha", "mha_forward")
    layer_stream = _load_bedrock_module(root, "mha", "layer_stream")
    manifest_mod = _load_bedrock_module(root, "mha", "manifest")
    fixtures = _load_bedrock_module(root, "mha", "fixtures")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    seed = _model_seed({"seed": manifest.seed}, model_id)
    rng = np.random.default_rng(seed)
    d, h, hd = model.d_model, model.num_heads, model.head_dim
    q_dim = h * hd
    kv = model.num_kv
    kv_dim = kv * hd
    scale = 0.02
    q_w = (rng.standard_normal((q_dim, d), dtype=np.float32) * scale)
    k_w = (rng.standard_normal((kv_dim, d), dtype=np.float32) * scale)
    v_w = (rng.standard_normal((kv_dim, d), dtype=np.float32) * scale)
    o_w = (rng.standard_normal((d, q_dim), dtype=np.float32) * scale)
    z = np.zeros(q_dim, np.float32)
    zkv = np.zeros(kv_dim, np.float32)
    zd = np.zeros(d, np.float32)
    data = _load_fixture_npz(root, "mha", info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures.slice_model_inputs(data, model, out_dim)
    native = _flatten_out(
        fwd.loom_mha_forward_batch(
            x_test, q_w=q_w, q_b=z, k_w=k_w, k_b=zkv, v_w=v_w, v_b=zkv, o_w=o_w, o_b=zd,
            num_heads=h, num_kv_heads=kv, head_dim=hd,
        ),
        out_dim,
    )
    layer = layer_stream.layer_stream_from_weights(
        model, q_w=q_w, q_b=z, k_w=k_w, k_b=zkv, v_w=v_w, v_b=zkv, o_w=o_w, o_b=zd,
    )
    payload = {
        "bedrock": "mha", "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "d_model": d, "seq_len": model.seq_len, "output_dim": out_dim,
        "layers": [layer.to_json_dict()],
    }
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "mha"
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock("mha", payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_cnn(root: Path, model_id: str, info: BedrockInfo, bedrock: str, **kw: Any) -> StreamResult:
    import torch
    import torch.nn as nn

    _add_bedrock_path(root, bedrock)
    manifest_mod = _load_bedrock_module(root, bedrock, "manifest")
    fixtures_mod = _load_bedrock_module(root, bedrock, "fixtures")
    extract_mod = _load_bedrock_module(root, bedrock, "extractors")
    manifest = manifest_mod.load_manifest()
    model = next(m for m in manifest.models if m.id == model_id)
    data = _load_fixture_npz(root, bedrock, info.fixture_version)
    out_dim = manifest_mod.model_output_dim(model)
    _, _, x_test, _ = fixtures_mod.slice_model_inputs(data, model, out_dim)
    torch.manual_seed(manifest.seed)
    mods: list[nn.Module] = []
    in_ch = model.input_channels
    for spec in model.layers:
        if bedrock == "cnn1":
            mods.append(nn.Conv1d(in_ch, spec.filters, spec.kernel_size, spec.stride, spec.padding, bias=False))
        elif bedrock == "cnn2":
            mods.append(nn.Conv2d(in_ch, spec.filters, spec.kernel_size, spec.stride, spec.padding, bias=False))
        else:
            mods.append(nn.Conv3d(in_ch, spec.filters, spec.kernel_size, spec.stride, spec.padding, bias=False))
        in_ch = spec.filters
    net = nn.Sequential(*mods, nn.Flatten())
    net.eval()
    with torch.no_grad():
        native = net(torch.from_numpy(x_test.astype(np.float32))).numpy()[:, :out_dim]
    if bedrock == "cnn1":
        layers = extract_mod.extract_pytorch_conv1d(net, model)
    elif bedrock == "cnn2":
        layers = extract_mod.extract_pytorch_conv2d(net, model)
    else:
        layers = extract_mod.extract_pytorch_conv3d(net, model)
    layer_dicts = [l.to_json_dict() for l in layers]
    payload: dict[str, Any] = {
        "bedrock": bedrock, "planet": "smoke", "model_id": model_id,
        "fixture_version": info.fixture_version,
        "input_channels": model.input_channels,
        "output_dim": out_dim, "layers": layer_dicts,
    }
    if bedrock == "cnn1":
        payload["seq_len"] = model.seq_len
    elif bedrock == "cnn2":
        payload["height"] = model.height
        payload["width"] = model.width
    else:
        payload["depth"] = model.depth
        payload["height"] = model.height
        payload["width"] = model.width
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / bedrock
    out.mkdir(parents=True, exist_ok=True)
    return stream_bedrock(bedrock, payload, fixture_version=info.fixture_version,
                          output_path=out / f"{model_id}.entity", root=root,
                          native_outputs=native, binary=kw.get("binary"))


def _smoke_mixer(root: Path, model_id: str, info: BedrockInfo, **kw: Any) -> StreamResult:
    _add_bedrock_path(root, "mixer")
    layer_stream = _load_bedrock_module(root, "mixer", "layer_stream")
    mixer_fwd = _load_bedrock_module(root, "mixer", "mixer_forward")
    weights_mod = _load_bedrock_module(root, "mixer", "weights")
    manifest_mod = _load_bedrock_module(root, "mixer", "manifest")
    manifest = manifest_mod.load_manifest()
    model = manifest_mod.ModelSpec(id=model_id)
    seed = weights_mod.model_seed(manifest, model)
    weights = weights_mod.init_mixer_weights(seed)
    data = _load_fixture_npz(root, "mixer", info.fixture_version)
    x_test = data["x_test"]
    native = mixer_fwd.loom_mixer_forward_batch(x_test, weights, output_dim=manifest.output_dim)
    layers = layer_stream.layers_json_from_weights(weights, model_id)
    out = kw.get("out_dir") or root / ".planetbridging" / "smoke" / "mixer"
    out.mkdir(parents=True, exist_ok=True)
    return stream_mixer(
        planet="smoke", model_id=model_id, layers=layers, output_dim=manifest.output_dim,
        fixture_version=info.fixture_version, output_path=out / f"{model_id}.entity",
        root=root, native_outputs=native, binary=kw.get("binary"),
    )


_SMOKE_RUNNERS: dict[str, Callable[..., StreamResult]] = {
    "dense": _smoke_dense,
    "cnn1": lambda r, m, i, **kw: _smoke_cnn(r, m, i, "cnn1", **kw),
    "cnn2": lambda r, m, i, **kw: _smoke_cnn(r, m, i, "cnn2", **kw),
    "cnn3": lambda r, m, i, **kw: _smoke_cnn(r, m, i, "cnn3", **kw),
    "mha": _smoke_mha,
    "lstm": lambda r, m, i, **kw: _smoke_recurrent(r, m, i, "lstm", **kw),
    "rnn": lambda r, m, i, **kw: _smoke_recurrent(r, m, i, "rnn", **kw),
    "layernorm": _smoke_layernorm,
    "embedding": _smoke_embedding,
    "rmsnorm": _smoke_rmsnorm,
    "swiglu": _smoke_swiglu,
    "residual": _smoke_residual,
    "mixer": _smoke_mixer,
}
