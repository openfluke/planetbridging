"""Shared setup for planetbridging examples."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_imports() -> None:
    """Allow running examples before `pip install -e .`."""
    src = ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def require_loom_stream() -> Path:
    ensure_imports()
    from planetbridging._binary import default_binary_path, find_loom_stream

    try:
        return find_loom_stream()
    except FileNotFoundError as exc:
        print(exc)
        print("\nBuild first:  go build -o bin/loom-stream ./cmd/loom-stream/")
        raise SystemExit(1) from exc


def print_result(result, *, title: str | None = None) -> None:
    """Pretty-print an EngineStreamResult."""
    head = title or f"{result.bedrock} / {result.planet}"
    print(f"\n{head}")
    print(f"  model:     {result.model_id} ({result.layer_count} layers)")
    print(f"  entity:    {result.entity_path}")
    print(f"  native→loom: {result.native_vs_loom}")
    if result.welvet is not None:
        print(f"  native→welvet: {result.native_vs_welvet}")
        print(f"  loom→welvet:   {result.loom_vs_welvet}")
    elif result.welvet_note:
        print(f"  welvet:    {result.welvet_note}")
