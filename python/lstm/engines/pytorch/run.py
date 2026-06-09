#!/usr/bin/env python3
"""PyTorch LSTM bedrock (Loom gate layout)."""

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
from shared.lstm_forward import unpack_gate  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402
from shared.weights import init_or_load_weights, layer_from_gates, save_weights  # noqa: E402

PLANET = "pytorch"
REQUIRED = ["weights.npz"]


def load_into_pytorch_lstm(cell: nn.LSTM, gates: dict[str, np.ndarray], model: ModelSpec) -> None:
    inp, hid = model.input_size, model.hidden_size
    w_ih = np.zeros((4 * hid, inp), dtype=np.float32)
    w_hh = np.zeros((4 * hid, hid), dtype=np.float32)
    b_ih = np.zeros(4 * hid, dtype=np.float32)
    for g, key in enumerate(("i", "f", "g", "o")):
        w_ih_g, w_hh_g, b_g = unpack_gate(gates[key], inp, hid)
        sl = slice(g * hid, (g + 1) * hid)
        w_ih[sl] = w_ih_g
        w_hh[sl] = w_hh_g
        b_ih[sl] = b_g
    cell.weight_ih_l0.data = torch.from_numpy(w_ih)
    cell.weight_hh_l0.data = torch.from_numpy(w_hh)
    cell.bias_ih_l0.data = torch.from_numpy(b_ih)
    cell.bias_hh_l0.data = torch.zeros(4 * hid, dtype=torch.float32)


def flatten_out(y: np.ndarray, out_dim: int) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)[:, :out_dim]


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[nn.LSTM, dict[str, np.ndarray], bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    skipped = is_complete(PLANET, model.id, REQUIRED)
    gates = init_or_load_weights(weights_path, model, manifest, skipped=skipped)

    cell = nn.LSTM(model.input_size, model.hidden_size, batch_first=True, bias=True)
    load_into_pytorch_lstm(cell, gates, model)
    cell.eval()

    if not skipped:
        save_weights(weights_path, gates)
        write_complete(PLANET, model.id, REQUIRED, torch.__version__)
    return cell, gates, skipped


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    cell, gates, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)

    with torch.no_grad():
        out, _ = cell(torch.from_numpy(x_test.astype(np.float32)))
        native = flatten_out(out.numpy(), out_dim)

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="pytorch",
            outputs=native,
            artifact_paths=[str(out_dir / "weights.npz")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host, planet=PLANET, model=model, manifest=manifest, layer=layer_from_gates(model, gates)
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, torch.__version__, handler))
