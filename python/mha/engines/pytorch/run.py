#!/usr/bin/env python3
"""PyTorch MHA bedrock (Loom causal + RoPE semantics)."""

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
from shared.layer_stream import layer_stream_from_weights, pytorch_linear_to_loom  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.mha_forward_torch import loom_mha_forward_torch  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "pytorch"
REQUIRED = ["model.pt"]


class LoomMHAModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        q_dim = num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, q_dim, bias=True)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=True)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return loom_mha_forward_torch(
            x,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            o_proj=self.o_proj,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )


def flatten_out(y: np.ndarray | torch.Tensor, out_dim: int) -> np.ndarray:
    flat = np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)
    return flat[:, :out_dim]


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[LoomMHAModule, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = LoomMHAModule(model.d_model, model.num_heads, model.num_kv)
    if is_complete(PLANET, model.id, REQUIRED):
        state = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        return net, True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model, out_dim)
    x_t = torch.from_numpy(x_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))

    torch.manual_seed(manifest.seed)
    if manifest.epochs > 0:
        opt = torch.optim.Adam(net.parameters(), lr=manifest.learning_rate)
        loss_fn = nn.MSELoss()
        net.train()
        n = x_t.shape[0]
        for _ in range(manifest.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, manifest.batch_size):
                idx = perm[start : start + manifest.batch_size]
                opt.zero_grad()
                pred = net(x_t[idx]).reshape(idx.shape[0], -1)
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                opt.step()
    net.eval()
    torch.save(net.state_dict(), out_dir / "model.pt")
    write_complete(PLANET, model.id, REQUIRED, torch.__version__)
    return net, False


def extract_layer(net: LoomMHAModule, model: ModelSpec):
    return layer_stream_from_weights(
        model,
        q_w=net.q_proj.weight.detach().cpu().numpy(),
        q_b=net.q_proj.bias.detach().cpu().numpy(),
        k_w=net.k_proj.weight.detach().cpu().numpy(),
        k_b=net.k_proj.bias.detach().cpu().numpy(),
        v_w=net.v_proj.weight.detach().cpu().numpy(),
        v_b=net.v_proj.bias.detach().cpu().numpy(),
        o_w=net.o_proj.weight.detach().cpu().numpy(),
        o_b=net.o_proj.bias.detach().cpu().numpy(),
    )


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    net, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)

    with torch.no_grad():
        native = flatten_out(net(torch.from_numpy(x_test.astype(np.float32))), out_dim)

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="pytorch",
            outputs=native,
            artifact_paths=[str(out_dir / "model.pt")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host, planet=PLANET, model=model, manifest=manifest, layer=extract_layer(net, model)
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, torch.__version__, handler))
