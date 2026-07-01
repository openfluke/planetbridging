#!/usr/bin/env python3
"""Generate missing bedrock fixture npz files in a dev checkout (numpy only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetbridging._fixtures import ensure_bedrock_fixture
from planetbridging.bedrocks import BEDROCK_IDS, FIXTURE_VERSIONS


def main() -> None:
    for bedrock in BEDROCK_IDS:
        fv = FIXTURE_VERSIONS[bedrock]
        out = ROOT / "python" / bedrock / "fixtures"
        path = out / f"{fv}.npz"
        if path.is_file():
            print(f"  ok  {bedrock}  {path.name}")
            continue
        ensure_bedrock_fixture(bedrock, ROOT, out_dir=out)
        print(f"  gen {bedrock}  {path.name}")


if __name__ == "__main__":
    main()
