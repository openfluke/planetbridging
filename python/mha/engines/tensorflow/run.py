#!/usr/bin/env python3
"""TensorFlow MHA bedrock (Loom causal + RoPE semantics)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.layer_stream import keras_dense_to_loom, layer_stream_from_weights  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST, ROPE_THETA  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "tensorflow"
REQUIRED = ["weights.npz"]


def _apply_rope_tf(vec: tf.Tensor, pos: int, num_heads: int, head_dim: int) -> tf.Tensor:
    half = head_dim // 2
    out = tf.identity(vec)
    for h in range(num_heads):
        base = h * head_dim
        for d in range(half):
            angle = pos / (ROPE_THETA ** (2 * d / head_dim))
            c, s = math.cos(angle), math.sin(angle)
            v0 = float(out[base + d].numpy())
            v1 = float(out[base + d + half].numpy())
            rot0 = v0 * c - v1 * s
            rot1 = v0 * s + v1 * c
            out = tf.tensor_scatter_nd_update(out, [[base + d]], [rot0])
            out = tf.tensor_scatter_nd_update(out, [[base + d + half]], [rot1])
    return out


class LoomMHAModule(tf.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        q_dim = num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = tf.keras.layers.Dense(q_dim, use_bias=True, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(kv_dim, use_bias=True, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(kv_dim, use_bias=True, name="v_proj")
        self.o_proj = tf.keras.layers.Dense(d_model, use_bias=True, name="o_proj")

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build((None, self.num_heads * self.head_dim))

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        batch = int(x.shape[0])
        seq = int(x.shape[1])
        heads_per_kv = self.num_heads // self.num_kv_heads
        scale = 1.0 / math.sqrt(self.head_dim)
        outs = []
        for bi in range(batch):
            cache_k: list[tf.Tensor] = []
            cache_v: list[tf.Tensor] = []
            rows = []
            for s in range(seq):
                tok = x[bi, s][None, :]
                q = _apply_rope_tf(self.q_proj(tok)[0], s, self.num_heads, self.head_dim)
                k = _apply_rope_tf(self.k_proj(tok)[0], s, self.num_kv_heads, self.head_dim)
                v = self.v_proj(tok)[0]
                cache_k.append(k)
                cache_v.append(v)
                attn_chunks = []
                for h in range(self.num_heads):
                    kv_h = h // heads_per_kv
                    scores = []
                    for kp in range(s + 1):
                        dot = tf.reduce_sum(
                            q[h * self.head_dim : (h + 1) * self.head_dim]
                            * cache_k[kp][kv_h * self.head_dim : (kv_h + 1) * self.head_dim]
                        )
                        scores.append(dot * scale)
                    smax = tf.reduce_max(tf.stack(scores))
                    exp_s = tf.exp(tf.stack(scores) - smax)
                    denom = tf.reduce_sum(exp_s)
                    for d in range(self.head_dim):
                        acc = tf.reduce_sum(
                            tf.stack(
                                [exp_s[kp] * cache_v[kp][kv_h * self.head_dim + d] for kp in range(s + 1)]
                            )
                        )
                        attn_chunks.append(acc / denom)
                rows.append(self.o_proj(tf.stack(attn_chunks)[None, :])[0])
            outs.append(tf.stack(rows))
        return tf.stack(outs)


def flatten_out(y: np.ndarray | tf.Tensor, out_dim: int) -> np.ndarray:
    flat = np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)
    return flat[:, :out_dim]


def save_weights(net: LoomMHAModule, path: Path) -> None:
    np.savez(
        path,
        q_w=net.q_proj.kernel.numpy(),
        q_b=net.q_proj.bias.numpy(),
        k_w=net.k_proj.kernel.numpy(),
        k_b=net.k_proj.bias.numpy(),
        v_w=net.v_proj.kernel.numpy(),
        v_b=net.v_proj.bias.numpy(),
        o_w=net.o_proj.kernel.numpy(),
        o_b=net.o_proj.bias.numpy(),
    )


def load_weights(net: LoomMHAModule, path: Path) -> None:
    data = np.load(path)
    net.q_proj.set_weights([data["q_w"], data["q_b"]])
    net.k_proj.set_weights([data["k_w"], data["k_b"]])
    net.v_proj.set_weights([data["v_w"], data["v_b"]])
    net.o_proj.set_weights([data["o_w"], data["o_b"]])


def extract_from_net(net: LoomMHAModule, model: ModelSpec):
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv * model.head_dim
    return layer_stream_from_weights(
        model,
        q_w=keras_dense_to_loom(net.q_proj.kernel.numpy()).reshape(q_dim, model.d_model),
        q_b=net.q_proj.bias.numpy(),
        k_w=keras_dense_to_loom(net.k_proj.kernel.numpy()).reshape(kv_dim, model.d_model),
        k_b=net.k_proj.bias.numpy(),
        v_w=keras_dense_to_loom(net.v_proj.kernel.numpy()).reshape(kv_dim, model.d_model),
        v_b=net.v_proj.bias.numpy(),
        o_w=keras_dense_to_loom(net.o_proj.kernel.numpy()).reshape(model.d_model, q_dim),
        o_b=net.o_proj.bias.numpy(),
    )


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[LoomMHAModule, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.npz"
    net = LoomMHAModule(model.d_model, model.num_heads, model.num_kv)
    net.build((None, model.d_model))

    if is_complete(PLANET, model.id, REQUIRED):
        load_weights(net, weights_path)
        return net, True

    out_dim = model_output_dim(model)
    x_train, y_train, _, _ = slice_model_inputs(data, model, out_dim)
    x_tf = tf.convert_to_tensor(x_train.astype(np.float32))
    y_tf = tf.convert_to_tensor(y_train.astype(np.float32))
    tf.keras.utils.set_random_seed(manifest.seed)
    opt = tf.keras.optimizers.Adam(manifest.learning_rate)
    n = x_tf.shape[0]
    rng = np.random.default_rng(manifest.seed)

    vars_ = (
        net.q_proj.kernel,
        net.q_proj.bias,
        net.k_proj.kernel,
        net.k_proj.bias,
        net.v_proj.kernel,
        net.v_proj.bias,
        net.o_proj.kernel,
        net.o_proj.bias,
    )

    if manifest.epochs > 0:
        for _ in range(manifest.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, manifest.batch_size):
                idx = perm[start : start + manifest.batch_size]
                with tf.GradientTape() as tape:
                    pred = net.forward(x_tf[idx])
                    loss = tf.reduce_mean(tf.square(tf.reshape(pred, (len(idx), -1)) - y_tf[idx]))
                grads = tape.gradient(loss, vars_)
                opt.apply_gradients(zip(grads, vars_))

    save_weights(net, weights_path)
    write_complete(PLANET, model.id, REQUIRED, tf.__version__)
    return net, False


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    net, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dir = model_dir(PLANET, model.id)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)
    native = flatten_out(net.forward(tf.convert_to_tensor(x_test.astype(np.float32))).numpy(), out_dim)

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="keras",
            outputs=native,
            artifact_paths=[str(out_dir / "weights.npz")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host, planet=PLANET, model=model, manifest=manifest, layer=extract_from_net(net, model)
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, tf.__version__, handler))
