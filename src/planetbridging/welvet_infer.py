"""Reload planetbridging .entity files via welvet and run Loom inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .compare import compare_outputs, diff_label

# Bedrocks where welvet reload + forward_polymorphic matches loom-stream on fixtures.
WELVET_RELOAD_BEDROCKS: frozenset[str] = frozenset(
    {
        "dense",
        "cnn1",
        "cnn2",
        "cnn3",
        "mha",
        "layernorm",
        "embedding",
        "rmsnorm",
        "swiglu",
    }
)

# Known C ABI gaps — entity loads but forward may panic or diverge.
WELVET_RELOAD_SKIP: frozenset[str] = frozenset({"lstm", "rnn", "mixer", "residual"})


def welvet_src_path() -> Path | None:
    """Monorepo path: loom/welvet/python/src."""
    root = Path(__file__).resolve().parents[2]  # planetbridging repo root
    candidate = root.parent / "welvet" / "python" / "src"
    if (candidate / "welvet" / "__init__.py").is_file():
        return candidate
    return None


def import_welvet() -> Any:
    """Import welvet.Network; adds monorepo src to sys.path if needed."""
    try:
        from welvet import Network  # type: ignore

        return Network
    except ImportError:
        src = welvet_src_path()
        if src is None:
            raise ImportError(
                "welvet not installed. From the loom monorepo:\n"
                "  pip install -e ../welvet/python\n"
                "Or set PYTHONPATH to loom/welvet/python/src"
            ) from None
        path = str(src)
        if path not in sys.path:
            sys.path.insert(0, path)
        from welvet import Network  # type: ignore

        return Network


def load_entity_network(entity_path: str | Path) -> Any:
    Network = import_welvet()
    blob = Path(entity_path).read_bytes()
    net = Network.deserialize_entity(blob)
    net.sync_inference_weights()
    return net


def _forward_polymorphic_row(net: Any, sample: np.ndarray, output_dim: int) -> np.ndarray:
    x = np.asarray(sample, dtype=np.float64)
    flat = x.reshape(-1).tolist()
    shape = list(x.shape) if x.ndim > 0 else [len(flat)]
    out = net.forward_polymorphic(flat, shape)
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    return arr[:output_dim]


def infer_bedrock_entity(
    bedrock: str,
    entity_path: str | Path,
    inputs: np.ndarray,
    *,
    output_dim: int,
    input_dim: int | None = None,
) -> np.ndarray:
    """Run welvet forward on a streamed .entity for common bedrock input layouts."""
    bedrock = bedrock.lower()
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if bedrock == "dense":
        return infer_dense_entity(entity_path, x, input_dim=input_dim or x.shape[-1])

    net = load_entity_network(entity_path)
    try:
        rows = [_forward_polymorphic_row(net, x[i], output_dim) for i in range(x.shape[0])]
        return np.stack(rows, axis=0)
    finally:
        net.free()


def try_infer_bedrock_entity(
    bedrock: str,
    entity_path: str | Path,
    inputs: np.ndarray,
    *,
    output_dim: int,
    input_dim: int | None = None,
) -> np.ndarray | None:
    """Best-effort welvet reload; returns None when unsupported or forward fails."""
    bedrock = bedrock.lower()
    if bedrock in WELVET_RELOAD_SKIP:
        return None
    try:
        return infer_bedrock_entity(
            bedrock, entity_path, inputs, output_dim=output_dim, input_dim=input_dim
        )
    except Exception:
        return None


def infer_dense_entity(
    entity_path: str | Path,
    inputs: np.ndarray,
    *,
    input_dim: int | None = None,
) -> np.ndarray:
    """Run welvet forward on a dense .entity for each row of inputs."""
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    dim = input_dim or x.shape[1]

    net = load_entity_network(entity_path)
    try:
        return np.stack([_forward_dense_row(net, row[:dim]) for row in x], axis=0)
    finally:
        net.free()


def _forward_dense_row(net: Any, row: np.ndarray) -> np.ndarray:
    flat = row.astype(np.float64).tolist()
    out = net.forward_polymorphic(flat, [1, len(flat)])
    return np.asarray(out, dtype=np.float64)


def infer_mixer_v1_entity(
    entity_path: str | Path,
    volumes: np.ndarray,
    *,
    output_dim: int = 8,
) -> np.ndarray:
    """Run welvet forward on mixer v1 .entity. volumes: [N, C, D, H, W]."""
    x = np.asarray(volumes, dtype=np.float64)
    if x.ndim == 4:
        x = x[np.newaxis, ...]

    net = load_entity_network(entity_path)
    try:
        outs = [_forward_mixer_v1_row(net, row, output_dim=output_dim) for row in x]
        return np.stack(outs, axis=0)
    finally:
        net.free()


def _forward_mixer_v1_row(net: Any, volume: np.ndarray, *, output_dim: int) -> np.ndarray:
    flat = volume.reshape(-1).astype(np.float64).tolist()
    # 5D volume input [C, D, H, W] = [1, 2, 2, 2]
    c, d, h, w = volume.shape
    out = net.forward_polymorphic(flat, [1, c, d, h, w])
    return np.asarray(out[:output_dim], dtype=np.float64)


def compare_native_loom_welvet(
    native: np.ndarray,
    loom_stream: np.ndarray,
    welvet_out: np.ndarray,
) -> dict[str, str]:
    """Three-way compare labels: native vs loom-stream, native vs welvet, loom vs welvet."""
    n_loom_max, _, n_loom_exact = compare_outputs(native, loom_stream)
    n_wel_max, _, n_wel_exact = compare_outputs(native, welvet_out)
    l_w_max, _, l_w_exact = compare_outputs(loom_stream, welvet_out)
    return {
        "native_vs_loom_stream": diff_label(n_loom_max, exact=n_loom_exact),
        "native_vs_welvet": diff_label(n_wel_max, exact=n_wel_exact),
        "loom_stream_vs_welvet": diff_label(l_w_max, exact=l_w_exact),
    }


def print_compare_ladder(
    *,
    title: str,
    native: np.ndarray,
    loom_stream: np.ndarray,
    welvet_out: np.ndarray | None = None,
) -> None:
    n_loom_max, n_loom_mean, n_loom_exact = compare_outputs(native, loom_stream)
    print(f"\n=== {title} ===")
    print(f"  native → loom-stream:  {diff_label(n_loom_max, exact=n_loom_exact):<6}  max={n_loom_max:.6e}  mean={n_loom_mean:.6e}")
    if welvet_out is not None:
        n_wel_max, n_wel_mean, n_wel_exact = compare_outputs(native, welvet_out)
        l_w_max, l_w_mean, l_w_exact = compare_outputs(loom_stream, welvet_out)
        print(f"  native → welvet:       {diff_label(n_wel_max, exact=n_wel_exact):<6}  max={n_wel_max:.6e}  mean={n_wel_mean:.6e}")
        print(f"  loom-stream → welvet:  {diff_label(l_w_max, exact=l_w_exact):<6}  max={l_w_max:.6e}  mean={l_w_mean:.6e}")
