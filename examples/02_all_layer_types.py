#!/usr/bin/env python3
"""Stream all 13 Loom layer bedrocks from a live PyTorch model.

    python examples/02_all_layer_types.py
"""

from __future__ import annotations

from _helpers import require_loom_stream

require_loom_stream()

from planetbridging import engines  # noqa: E402
from planetbridging._binary import repo_root  # noqa: E402
from planetbridging.bedrocks import BEDROCK_IDS, STREAM_LAYER_COUNTS  # noqa: E402

MIXER_POC_TOLERANCE = 5e-5


def main() -> None:
    print("Streaming all 13 bedrock layer types (PyTorch) …\n")
    results = engines.stream_all_bedrocks("pytorch", root=repo_root())

    ok = 0
    for r in results:
        label = r.native_vs_loom
        extra = ""
        if r.bedrock == "mixer" and label == "DIFF":
            from planetbridging.compare import compare_outputs

            max_d, _, _ = compare_outputs(r.native, r.loom_stream)
            label = "PASS" if max_d < MIXER_POC_TOLERANCE else "DIFF"
            extra = f"  (mixer POC max={max_d:.2e})"
        layers = STREAM_LAYER_COUNTS.get(r.bedrock, r.layer_count)
        mark = "✓" if label in ("PASS", "EXACT") else "✗"
        print(f"  {mark} {r.bedrock:<12} {layers:>2} layers  {label:<6}{extra}")
        if label in ("PASS", "EXACT"):
            ok += 1

    print(f"\n{ok}/{len(BEDROCK_IDS)} passed")
    if ok < len(BEDROCK_IDS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
