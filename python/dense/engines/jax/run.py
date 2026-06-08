#!/usr/bin/env python3
"""JAX/Flax dense bedrock."""

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
from shared.manifest import LayerSpec, Manifest, ModelSpec, model_output_dim  # noqa: E402
from shared.loom_bridge import stream_planet_to_loom  # noqa: E402
from shared.runner import run_planet  # noqa: E402
from shared.spec import DEFAULT_HOST  # noqa: E402
from shared.variants import VariantResult  # noqa: E402

PLANET = "jax"
REQUIRED = ["params.msgpack"]


def layer_bias(layer: LayerSpec, model: ModelSpec) -> bool:
    if layer.bias is not None:
        return layer.bias
    return model.bias


def layers_cfg(model: ModelSpec) -> tuple[tuple[int, str, bool], ...]:
    return tuple(
        (layer.units, layer.activation.lower(), layer_bias(layer, model))
        for layer in model.layers
    )


class MLP(nn.Module):
    layers_cfg: tuple[tuple[int, str, bool], ...]

    @nn.compact
    def __call__(self, x):
        for units, act, use_bias in self.layers_cfg:
            x = nn.Dense(units, use_bias=use_bias)(x)
            if act == "relu":
                x = nn.relu(x)
            elif act == "tanh":
                x = nn.tanh(x)
            elif act == "sigmoid":
                x = nn.sigmoid(x)
            elif act != "linear":
                raise ValueError(f"unsupported activation: {act}")
        return x


@dataclass
class TrainBundle:
    module: MLP
    state: train_state.TrainState

    def apply(self, x):
        return self.module.apply({"params": self.state.params}, x)


def train_or_load(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
) -> tuple[TrainBundle, bool]:
    out_dir = model_dir(PLANET, model.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.msgpack"
    out_dim = model_output_dim(model)

    module = MLP(layers_cfg(model))
    x_train, y_train, _, _ = slice_model_inputs(data, model.input_dim, out_dim)
    xj = jnp.asarray(x_train.astype(np.float32))
    yj = jnp.asarray(y_train.astype(np.float32))

    if is_complete(PLANET, model.id, REQUIRED):
        from flax.serialization import from_bytes

        template = module.init(jax.random.PRNGKey(0), xj[:1])["params"]
        params = from_bytes(template, params_path.read_bytes())
        state = train_state.TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=optax.adam(manifest.learning_rate),
        )
        return TrainBundle(module, state), True

    key = jax.random.PRNGKey(manifest.seed)
    params = module.init(key, xj[:1])["params"]
    state = train_state.TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optax.adam(manifest.learning_rate),
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
    for epoch in range(manifest.epochs):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, n)
        for start in range(0, n, manifest.batch_size):
            idx = perm[start : start + manifest.batch_size]
            st, _ = step(st, xj[idx], yj[idx])

    from flax.serialization import to_bytes

    params_path.write_bytes(to_bytes(st.params))
    write_complete(
        PLANET,
        model.id,
        REQUIRED,
        framework_version=f"jax-{jax.__version__}",
        extra={"fixture_version": manifest.fixture_version},
    )
    return TrainBundle(module, st), False


def handler(
    *,
    model: ModelSpec,
    manifest: Manifest,
    data: dict[str, np.ndarray],
    models_dir: Path,
    host: str = DEFAULT_HOST,
    skip_loom: bool = False,
):
    bundle, skipped = train_or_load(
        model=model, manifest=manifest, data=data, models_dir=models_dir
    )
    _, _, x_test, _ = slice_model_inputs(
        data, model.input_dim, model_output_dim(model)
    )
    out = np.asarray(bundle.apply(jnp.asarray(x_test.astype(np.float32))))
    results = [
        VariantResult(
            planet=PLANET,
            stage="native",
            format="jax",
            outputs=out.astype(np.float64),
            artifact_paths=[str(model_dir(PLANET, model.id) / "params.msgpack")],
            train_skipped=skipped,
        )
    ]
    if not skip_loom:
        loom = stream_planet_to_loom(
            host=host,
            planet=PLANET,
            model=model,
            manifest=manifest,
            net=bundle.module,
            extractor="jax",
            params=bundle.state.params,
        )
        if loom is not None:
            results.append(loom)
    return results


if __name__ == "__main__":
    raise SystemExit(run_planet(PLANET, f"jax-{jax.__version__}", handler))
