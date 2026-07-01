"""All planetbridging bedrock layer types (matches compare-host tabs)."""

from __future__ import annotations

from dataclasses import dataclass

# Tab order matches host/compare.go AllBedrockIDs and PROGRESS.md
BEDROCK_IDS: tuple[str, ...] = (
    "dense",
    "cnn1",
    "cnn2",
    "cnn3",
    "mha",
    "lstm",
    "rnn",
    "layernorm",
    "embedding",
    "rmsnorm",
    "swiglu",
    "residual",
    "mixer",
)

# Default fixture npz tag per bedrock (python/<bedrock>/fixtures/<version>.npz)
FIXTURE_VERSIONS: dict[str, str] = {
    "dense": "dense_bedrock_v2",
    "cnn1": "cnn1_bedrock_v1",
    "cnn2": "cnn2_bedrock_v1",
    "cnn3": "cnn3_bedrock_v1",
    "mha": "mha_bedrock_v1",
    "lstm": "lstm_bedrock_v1",
    "rnn": "rnn_bedrock_v1",
    "layernorm": "layernorm_bedrock_v1",
    "embedding": "embedding_bedrock_v1",
    "rmsnorm": "rmsnorm_bedrock_v1",
    "swiglu": "swiglu_bedrock_v1",
    "residual": "residual_bedrock_v1",
    "mixer": "mixer_bedrock_v1",
}

# Canonical multi-layer model per bedrock for streaming tests and engine API.
# dense/cnn/mixer use explicit multi-layer stacks; other tabs are single-type layers
# (heterogeneous multi-layer stacks use mixer v2).
STREAM_MODEL_IDS: dict[str, str] = {
    "dense": "mlp_32_64_32_16_8_relu",       # 4 dense layers
    "cnn1": "conv1_32_8_4_2layer",           # 2 conv1d layers
    "cnn2": "conv2_8_4_2layer",              # 2 conv2d layers
    "cnn3": "conv3_4_4_2layer",              # 2 conv3d layers
    "mha": "mha_8_2_4",
    "lstm": "lstm_4_4_4",
    "rnn": "rnn_4_4_4",
    "layernorm": "layernorm_8",
    "embedding": "embedding_16_4_4",
    "rmsnorm": "rmsnorm_8",
    "swiglu": "swiglu_8_16_4",
    "residual": "residual_8",
    "mixer": "mixer_all_v2",                 # 16 layers, all 12 types
}

# Layer count for the stream model (for reporting).
STREAM_LAYER_COUNTS: dict[str, int] = {
    "dense": 4,
    "cnn1": 2,
    "cnn2": 2,
    "cnn3": 2,
    "mha": 1,
    "lstm": 1,
    "rnn": 1,
    "layernorm": 1,
    "embedding": 1,
    "rmsnorm": 1,
    "swiglu": 1,
    "residual": 1,
    "mixer": 16,
}

# Back-compat alias
SMOKE_MODEL_IDS = STREAM_MODEL_IDS

# AI engine planets supported per bedrock (from POC manifests).
BEDROCK_PLANETS: dict[str, tuple[str, ...]] = {
    "dense": ("pytorch", "tensorflow", "jax", "sklearn"),
    "mixer": ("pytorch", "tensorflow", "jax"),
}
_DEFAULT_PLANETS = ("pytorch", "tensorflow", "jax")
for _bid in BEDROCK_IDS:
    BEDROCK_PLANETS.setdefault(_bid, _DEFAULT_PLANETS)


@dataclass(frozen=True)
class BedrockInfo:
    id: str
    fixture_version: str
    stream_model_id: str
    layer_count: int

    @property
    def smoke_model_id(self) -> str:
        return self.stream_model_id


BEDROCKS: dict[str, BedrockInfo] = {
    bid: BedrockInfo(
        id=bid,
        fixture_version=FIXTURE_VERSIONS[bid],
        stream_model_id=STREAM_MODEL_IDS[bid],
        layer_count=STREAM_LAYER_COUNTS[bid],
    )
    for bid in BEDROCK_IDS
}
