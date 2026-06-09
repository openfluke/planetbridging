#!/usr/bin/env python3
"""JAX/Flax CNN2 bedrock."""

from __future__ import annotations

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
from shared.manifest import Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "jax"
REQUIRED = ["params.msgpack"]


def layers_cfg(model: ModelSpec) -> tuple[tuple[int, int, int, int, str], ...]:
    return tuple(
        (layer.filters, layer.kernel_size, layer.stride, layer.padding, layer.activation.lower())
        for layer in model.layers
    )


class CNN2Stack(nn.Module):
    layers_cfg: tuple[tuple[int, int, int, int, str], ...]
    in_channels: int

    @nn.compact
    def __call__(self, x):
        for filters, k, stride, pad, act in self.layers_cfg:
            x = nn.Conv(
                features=filters,
                kernel_size=(k, k),
                strides=(stride, stride),
                padding=((pad, pad), (pad, pad)),
                use_bias=False,
            )(x)
            if act == "relu":
                x = nn.relu(x)
            elif act == "tanh":
                x = nn.tanh(x)
            elif act == "sigmoid":
                x = nn.sigmoid(x)
            elif act != "linear":
                raise ValueError(f"unsupported activation: {act}")
        return x.reshape((x.shape[0], -1))


@dataclass
class TrainBundle:
    module: CNN2Stack
    state: train_state.TrainState

    def apply(self, x):
        return self.module.apply({"params": self.state.params}, x)


def to_nhwc(x: np.ndarray) -> np.ndarray:
    """Fixture is NCHW; Flax Conv expects NHWC."""
    return np.transpose(x, (0, 2, 3, 1))


def train_or_load(*, model: ModelSpec, manifest: Manifest, data: dict) -> tuple[TrainBundle, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.msgpack"
    out_dim = model_output_dim(model)
    module = CNN2Stack(layers_cfg(model), model.input_channels)
    x_train, y_train, _, _ = slice_model_inputs(data, model, out_dim)
    xj = jnp.asarray(to_nhwc(x_train).astype(np.float32))
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
            return jnp.mean((pred - yb) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    n = xj.shape[0]
    st = state
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
    out = np.asarray(bundle.apply(jnp.asarray(to_nhwc(x_test).astype(np.float32))), dtype=np.float64)
    out = out[:, :out_dim]
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
            net=bundle,
            extractor="jax",
            params=bundle.state.params,
        )
        if loom:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, jax.__version__, handler))
