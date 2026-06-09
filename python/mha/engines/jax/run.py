#!/usr/bin/env python3
"""JAX/Flax MHA bedrock (Loom causal + RoPE semantics)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.artifacts import is_complete, model_dir, write_complete  # noqa: E402
from shared.fixtures import slice_model_inputs  # noqa: E402
from shared.layer_stream import layer_stream_from_weights  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST, ROPE_THETA  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "jax"
REQUIRED = ["params.msgpack"]


def flax_dense_to_loom(kernel: np.ndarray) -> np.ndarray:
    k = np.asarray(kernel, dtype=np.float32)
    return np.ascontiguousarray(k.T).reshape(-1)


def _apply_rope_jax(vec: jnp.ndarray, pos: int, num_heads: int, head_dim: int) -> jnp.ndarray:
    half = head_dim // 2
    out = vec
    for h in range(num_heads):
        base = h * head_dim
        for d in range(half):
            angle = pos / (ROPE_THETA ** (2 * d / head_dim))
            c, s = math.cos(angle), math.sin(angle)
            v0 = float(out[base + d])
            v1 = float(out[base + d + half])
            out = out.at[base + d].set(v0 * c - v1 * s)
            out = out.at[base + d + half].set(v0 * s + v1 * c)
    return out


class LoomMHAModule(nn.Module):
    d_model: int
    num_heads: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q_proj = nn.Dense(q_dim, use_bias=True, name="q_proj")
        k_proj = nn.Dense(kv_dim, use_bias=True, name="k_proj")
        v_proj = nn.Dense(kv_dim, use_bias=True, name="v_proj")
        o_proj = nn.Dense(self.d_model, use_bias=True, name="o_proj")

        batch = x.shape[0]
        seq = x.shape[1]
        heads_per_kv = self.num_heads // self.num_kv_heads
        scale = 1.0 / math.sqrt(self.head_dim)
        outs = []
        for bi in range(batch):
            cache_k = []
            cache_v = []
            rows = []
            for s in range(seq):
                q = _apply_rope_jax(q_proj(x[bi, s]), s, self.num_heads, self.head_dim)
                k = _apply_rope_jax(k_proj(x[bi, s]), s, self.num_kv_heads, self.head_dim)
                v = v_proj(x[bi, s])
                cache_k.append(k)
                cache_v.append(v)
                attn_chunks = []
                for h in range(self.num_heads):
                    kv_h = h // heads_per_kv
                    scores = []
                    for kp in range(s + 1):
                        dot = jnp.sum(
                            q[h * self.head_dim : (h + 1) * self.head_dim]
                            * cache_k[kp][kv_h * self.head_dim : (kv_h + 1) * self.head_dim]
                        )
                        scores.append(dot * scale)
                    smax = jnp.max(jnp.stack(scores))
                    exp_s = jnp.exp(jnp.stack(scores) - smax)
                    denom = jnp.sum(exp_s)
                    for d in range(self.head_dim):
                        acc = jnp.sum(
                            jnp.stack(
                                [exp_s[kp] * cache_v[kp][kv_h * self.head_dim + d] for kp in range(s + 1)]
                            )
                        )
                        attn_chunks.append(acc / denom)
                rows.append(o_proj(jnp.stack(attn_chunks)))
            outs.append(jnp.stack(rows))
        return jnp.stack(outs)


@dataclass
class TrainBundle:
    module: LoomMHAModule
    state: train_state.TrainState

    def apply(self, x):
        return self.module.apply({"params": self.state.params}, x)


def extract_from_params(params: dict, model: ModelSpec):
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv * model.head_dim
    return layer_stream_from_weights(
        model,
        q_w=flax_dense_to_loom(params["q_proj"]["kernel"]).reshape(q_dim, model.d_model),
        q_b=np.asarray(params["q_proj"]["bias"], dtype=np.float32),
        k_w=flax_dense_to_loom(params["k_proj"]["kernel"]).reshape(kv_dim, model.d_model),
        k_b=np.asarray(params["k_proj"]["bias"], dtype=np.float32),
        v_w=flax_dense_to_loom(params["v_proj"]["kernel"]).reshape(kv_dim, model.d_model),
        v_b=np.asarray(params["v_proj"]["bias"], dtype=np.float32),
        o_w=flax_dense_to_loom(params["o_proj"]["kernel"]).reshape(model.d_model, q_dim),
        o_b=np.asarray(params["o_proj"]["bias"], dtype=np.float32),
    )


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[TrainBundle, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.msgpack"
    out_dim = model_output_dim(model)
    module = LoomMHAModule(model.d_model, model.num_heads, model.num_kv)
    x_train, y_train, _, _ = slice_model_inputs(data, model, out_dim)
    xj = jnp.asarray(x_train.astype(np.float32))
    yj = jnp.asarray(y_train.astype(np.float32))

    if is_complete(PLANET, model.id, REQUIRED):
        from flax.serialization import from_bytes

        template = module.init(jax.random.PRNGKey(0), xj[:1])["params"]
        params = from_bytes(template, params_path.read_bytes())
        state = train_state.TrainState.create(
            apply_fn=module.apply, params=params, tx=optax.adam(manifest.learning_rate)
        )
        return TrainBundle(module, state), True

    key = jax.random.PRNGKey(manifest.seed)
    params = module.init(key, xj[:1])["params"]
    state = train_state.TrainState.create(
        apply_fn=module.apply, params=params, tx=optax.adam(manifest.learning_rate)
    )

    @jax.jit
    def step(st, xb, yb):
        def loss_fn(p):
            pred = module.apply({"params": p}, xb)
            return jnp.mean((pred.reshape(xb.shape[0], -1) - yb) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    st = state
    if manifest.epochs > 0:
        n = xj.shape[0]
        for _ in range(manifest.epochs):
            key, sub = jax.random.split(key)
            perm = jax.random.permutation(sub, n)
            for start in range(0, n, manifest.batch_size):
                idx = perm[start : start + manifest.batch_size]
                st, _ = step(st, xj[idx], yj[idx])
    state = st

    from flax.serialization import to_bytes

    params_path.write_bytes(to_bytes(state.params))
    write_complete(PLANET, model.id, REQUIRED, jax.__version__)
    return TrainBundle(module, state), False


def handler(*, model: ModelSpec, manifest: Manifest, data: dict, models_dir: Path, host: str = DEFAULT_HOST, skip_loom: bool = False):
    bundle, skipped = train_or_load(model=model, manifest=manifest, data=data)
    out_dim = model_output_dim(model)
    _, _, x_test, _ = slice_model_inputs(data, model, out_dim)
    out = np.asarray(bundle.apply(jnp.asarray(x_test.astype(np.float32))), dtype=np.float64)
    out = out.reshape(out.shape[0], -1)[:, :out_dim]
    out_dir = model_dir(PLANET, model.id)

    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="jax",
            outputs=out,
            artifact_paths=[str(out_dir / "params.msgpack")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            layer=extract_from_params(bundle.state.params, model),
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, jax.__version__, handler))
