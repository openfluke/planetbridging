#!/usr/bin/env python3
"""Generate missing bedrock fixture npz files before PyPI bundle (numpy only)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planetbridging.bedrocks import BEDROCK_IDS, FIXTURE_VERSIONS


def _load_fixtures_module(bedrock: str):
    from planetbridging.bedrock_smoke import _load_bedrock_module

    return _load_bedrock_module(ROOT, bedrock, "fixtures")


def _load_manifest_module(bedrock: str):
    from planetbridging.bedrock_smoke import _load_bedrock_module

    return _load_bedrock_module(ROOT, bedrock, "manifest")


def main() -> None:
    for bedrock in BEDROCK_IDS:
        fv = FIXTURE_VERSIONS[bedrock]
        path = ROOT / "python" / bedrock / "fixtures" / f"{fv}.npz"
        if path.is_file():
            print(f"  ok  {bedrock}  {path.name}")
            continue
        fixtures = _load_fixtures_module(bedrock)
        manifest = _load_manifest_module(bedrock).load_manifest()
        if hasattr(fixtures, "ensure_fixtures"):
            fixtures.ensure_fixtures(manifest)
            print(f"  gen {bedrock}  {path.name}")
        elif path.is_file():
            print(f"  ok  {bedrock}  {path.name}")
        else:
            raise SystemExit(f"missing fixtures for {bedrock}: {path}")


if __name__ == "__main__":
    main()
