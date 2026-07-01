#!/usr/bin/env python3
"""Three-way compare: native → loom-stream → welvet entity reload.

Welvet reload works for most single-type bedrocks (cnn, mha, norms, …).
LSTM/RNN/mixer/residual and dense welvet are still flaky — the script skips
gracefully when welvet is missing or unsupported.

    python examples/05_welvet_ladder.py
    python examples/05_welvet_ladder.py cnn1
    python examples/05_welvet_ladder.py mha layernorm swiglu
"""

from __future__ import annotations

import sys

from _helpers import require_loom_stream

require_loom_stream()

from planetbridging import engines  # noqa: E402
from planetbridging._binary import repo_root  # noqa: E402
from planetbridging.welvet_infer import (  # noqa: E402
    WELVET_RELOAD_BEDROCKS,
    WELVET_RELOAD_SKIP,
    import_welvet,
    print_compare_ladder,
)


def main() -> None:
    if len(sys.argv) > 1:
        bedrocks = [a.lower() for a in sys.argv[1:]]
    else:
        bedrocks = sorted(WELVET_RELOAD_BEDROCKS)

    try:
        import_welvet()
    except ImportError as exc:
        print(exc)
        print("\nInstall welvet:  pip install -e ../welvet/python")
        raise SystemExit(1) from exc

    print("Welvet ladder (native → loom-stream → welvet reload)\n")
    print(f"Supported: {', '.join(sorted(WELVET_RELOAD_BEDROCKS))}")
    print(f"Skipped:   {', '.join(sorted(WELVET_RELOAD_SKIP))}\n")

    for bedrock in bedrocks:
        if bedrock in WELVET_RELOAD_SKIP:
            print(f"  skip {bedrock} (known C ABI gap)")
            continue

        result = engines.stream(
            bedrock,
            "pytorch",
            root=repo_root(),
            try_welvet=True,
        )
        n = len(result.welvet) if result.welvet is not None else len(result.native)
        print_compare_ladder(
            title=f"{bedrock} / {result.model_id}",
            native=result.native[:n],
            loom_stream=result.loom_stream[:n],
            welvet_out=result.welvet,
        )
        if result.welvet_note and result.welvet is None:
            print(f"  note: {result.welvet_note}")


if __name__ == "__main__":
    main()
