#!/usr/bin/env python3
"""Multi-layer stacks: 4-layer MLP, 2-layer CNNs, 16-layer Mixer v2.

    python examples/04_multi_layer_models.py
"""

from __future__ import annotations

from _helpers import print_result, require_loom_stream

require_loom_stream()

from planetbridging import engines  # noqa: E402
from planetbridging._binary import repo_root  # noqa: E402
from planetbridging.bedrocks import STREAM_MODEL_IDS  # noqa: E402

MIXER_POC_TOLERANCE = 5e-5

MULTI_LAYER = (
    ("dense", 4, "mlp_32_64_32_16_8_relu"),
    ("cnn1", 2, "conv1_32_8_4_2layer"),
    ("cnn2", 2, "conv2_8_4_2layer"),
    ("cnn3", 2, "conv3_4_4_2layer"),
    ("mixer", 16, "mixer_all_v2"),
)


def _ok(result) -> bool:
    if result.native_vs_loom in ("PASS", "EXACT"):
        return True
    if result.bedrock == "mixer":
        from planetbridging.compare import compare_outputs

        max_d, _, _ = compare_outputs(result.native, result.loom_stream)
        return max_d < MIXER_POC_TOLERANCE
    return False


def main() -> None:
    print("Multi-layer models (PyTorch → Loom)\n")
    failures = []

    for bedrock, expected_layers, model_id in MULTI_LAYER:
        assert STREAM_MODEL_IDS[bedrock] == model_id
        result = engines.stream(bedrock, "pytorch", model_id=model_id, root=repo_root())
        print_result(result, title=f"{bedrock} — {expected_layers} layers")
        if result.layer_count != expected_layers:
            failures.append(f"{bedrock}: expected {expected_layers} layers, got {result.layer_count}")
        if not _ok(result):
            failures.append(f"{bedrock}: {result.native_vs_loom}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nAll multi-layer models passed.")


if __name__ == "__main__":
    main()
