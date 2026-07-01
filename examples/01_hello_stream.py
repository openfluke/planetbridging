#!/usr/bin/env python3
"""Smallest demo: stream one live PyTorch model into a Loom .entity file.

    python examples/01_hello_stream.py
    python examples/01_hello_stream.py mha
"""

from __future__ import annotations

import sys

from _helpers import print_result, require_loom_stream

require_loom_stream()

from planetbridging import engines  # noqa: E402
from planetbridging._binary import repo_root  # noqa: E402


def main() -> None:
    bedrock = (sys.argv[1] if len(sys.argv) > 1 else "dense").lower()
    print(f"Streaming {bedrock} (PyTorch) → loom-stream → .entity …")

    result = engines.stream(bedrock, "pytorch", root=repo_root())
    print_result(result)

    if result.native_vs_loom not in ("PASS", "EXACT"):
        raise SystemExit(1)
    print("\nDone — native and Loom outputs match.")


if __name__ == "__main__":
    main()
