#!/usr/bin/env python3
"""Same bedrock layer on every installed AI engine (PyTorch / TensorFlow / JAX).

    python examples/03_cross_engine.py
    python examples/03_cross_engine.py cnn2
"""

from __future__ import annotations

import sys

from _helpers import print_result, require_loom_stream

require_loom_stream()

from planetbridging import engines  # noqa: E402


def main() -> None:
    bedrock = (sys.argv[1] if len(sys.argv) > 1 else "layernorm").lower()
    installed = engines.installed_planets(bedrock)
    if not installed:
        print(f"No engines installed for {bedrock}. Try: pip install -e \".[pytorch,tensorflow,jax]\"")
        raise SystemExit(1)

    print(f"Bedrock: {bedrock}")
    print(f"Engines: {', '.join(installed)}\n")

    results = engines.stream_all_planets(bedrock)
    failures = []
    for r in results:
        print_result(r)
        if r.native_vs_loom not in ("PASS", "EXACT"):
            failures.append((r.planet, r.native_vs_loom))

    if failures:
        print(f"\nFailed: {failures}")
        raise SystemExit(1)
    print("\nAll installed engines match Loom.")


if __name__ == "__main__":
    main()
