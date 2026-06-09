#!/usr/bin/env python3
"""PyTorch mixer bedrock (10-layer stack matching mixer_forward)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared import mixer_spec as ms  # noqa: E402
from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402
from shared.mixer_forward import load_mha_torch_forward  # noqa: E402
from shared.weights import init_or_load_weights, save_weights  # noqa: E402

PLANET = "pytorch"
REQUIRED = ["weights.npz"]


class MixerModule(nn.Module):
    def __init__(self, weights: dict[str, np.ndarray]):
        super().__init__()
        self.weights = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in weights.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weights
        t = x
        t = self._cnn3(t, w["cnn3"])
        t = t.reshape(t.shape[0], -1)
        t = self._dense(t, w["dense1_w"], w.get("dense1_b"), "linear")
        t = t.reshape(t.shape[0], 1, ms.CNN2_H, ms.CNN2_W)
        t = self._cnn2(t, w["cnn2"])
        t = t.reshape(t.shape[0], -1)
        t = self._dense(t, w["dense2_w"], w.get("dense2_b"), "relu")
        t = t.reshape(t.shape[0], 1, ms.CNN1_LEN)
        t = self._cnn1(t, w["cnn1"])
        t = t.reshape(t.shape[0], -1)
        t = self._dense(t, w["dense3_w"], w.get("dense3_b"), "linear")
        t = t.reshape(t.shape[0], ms.MHA_SEQ, ms.MHA_D_MODEL)
        t = self._mha(t, w)
        t = self._rnn(t, w["rnn"])
        t = self._lstm(t, w)
        t = t.reshape(t.shape[0], -1)
        t = self._dense(t, w["dense4_w"], w.get("dense4_b"), "linear")
        return t[:, : ms.OUTPUT_DIM]

    def _dense(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = weight.numel() // in_dim
        w = weight.reshape(out_dim, in_dim)
        out = x @ w.T
        if bias is not None:
            out = out + bias
        if activation == "relu":
            out = torch.relu(out)
        return out

    def _cnn3(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = weight.reshape(ms.CNN3_FILTERS, ms.VOLUME_C, ms.CNN3_KERNEL, ms.CNN3_KERNEL, ms.CNN3_KERNEL)
        return nn.functional.conv3d(x, w, stride=1, padding=0)

    def _cnn2(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = weight.reshape(ms.CNN2_FILTERS, 1, ms.CNN2_KERNEL, ms.CNN2_KERNEL)
        return nn.functional.conv2d(x, w, stride=1, padding=0)

    def _cnn1(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = weight.reshape(ms.CNN1_FILTERS, 1, ms.CNN1_KERNEL)
        return nn.functional.conv1d(x, w, stride=1, padding=0)

    def _mha(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        loom_mha_forward_torch = load_mha_torch_forward()
        q_dim = ms.MHA_HEADS * ms.MHA_HEAD_DIM
        d = ms.MHA_D_MODEL
        q_proj = nn.Linear(d, q_dim, bias=True)
        k_proj = nn.Linear(d, q_dim, bias=True)
        v_proj = nn.Linear(d, q_dim, bias=True)
        o_proj = nn.Linear(q_dim, d, bias=True)
        q_proj.weight.data = w["mha_q_w"].reshape(q_dim, d)
        q_proj.bias.data = w["mha_q_b"]
        k_proj.weight.data = w["mha_k_w"].reshape(q_dim, d)
        k_proj.bias.data = w["mha_k_b"]
        v_proj.weight.data = w["mha_v_w"].reshape(q_dim, d)
        v_proj.bias.data = w["mha_v_b"]
        o_proj.weight.data = w["mha_o_w"].reshape(d, q_dim)
        o_proj.bias.data = w["mha_o_b"]
        return loom_mha_forward_torch(
            x,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            num_heads=ms.MHA_HEADS,
            num_kv_heads=ms.MHA_HEADS,
            head_dim=ms.MHA_HEAD_DIM,
        )

    def _rnn(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch, seq, inp = x.shape
        hid = ms.RECURRENT_HID
        flat = weight.detach().cpu().numpy()
        ih = hid * inp
        hh = hid * hid
        w_ih = flat[:ih].reshape(hid, inp)
        w_hh = flat[ih : ih + hh].reshape(hid, hid)
        b = flat[ih + hh : ih + hh + hid]
        h = torch.zeros(batch, hid, dtype=x.dtype, device=x.device)
        outs = []
        for t in range(seq):
            xt = x[:, t, :]
            pre = xt @ w_ih.T + h @ w_hh.T + b
            h = torch.tanh(pre)
            outs.append(h)
        return torch.stack(outs, dim=1)

    def _lstm(self, x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
        batch, seq, inp = x.shape
        hid = ms.RECURRENT_HID
        gates = {}
        for name in ("i", "f", "g", "o"):
            flat = w[f"lstm_{name}"].detach().cpu().numpy()
            ih = hid * inp
            hh = hid * hid
            gates[name] = (
                flat[:ih].reshape(hid, inp),
                flat[ih : ih + hh].reshape(hid, hid),
                flat[ih + hh : ih + hh + hid],
            )
        h = torch.zeros(batch, hid, dtype=x.dtype, device=x.device)
        c = torch.zeros(batch, hid, dtype=x.dtype, device=x.device)
        outs = []
        for t in range(seq):
            xt = x[:, t, :]
            pre_i = xt @ gates["i"][0].T + h @ gates["i"][1].T + gates["i"][2]
            pre_f = xt @ gates["f"][0].T + h @ gates["f"][1].T + gates["f"][2]
            pre_g = xt @ gates["g"][0].T + h @ gates["g"][1].T + gates["g"][2]
            pre_o = xt @ gates["o"][0].T + h @ gates["o"][1].T + gates["o"][2]
            i_g = torch.sigmoid(pre_i)
            f_g = torch.sigmoid(pre_f)
            g_g = torch.tanh(pre_g)
            o_g = torch.sigmoid(pre_o)
            c = f_g * c + i_g * g_g
            h = o_g * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[MixerModule, dict[str, np.ndarray], bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    skipped = is_complete(PLANET, model.id, REQUIRED)
    weights = init_or_load_weights(weights_path, model, manifest, skipped=skipped)
    net = MixerModule(weights)
    net.eval()
    if not skipped:
        save_weights(weights_path, weights)
        write_complete(PLANET, model.id, REQUIRED, torch.__version__)
    return net, weights, skipped


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    net, weights, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)

    with torch.no_grad():
        native = net(torch.from_numpy(x_test.astype(np.float32))).cpu().numpy().astype(np.float64)

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
            host=host, planet=PLANET, model=model, manifest=manifest, weights=weights
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, torch.__version__, handler))
