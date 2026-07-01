"""Shared setup for planetbridging examples (pip install or git checkout)."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
# Git checkout: planetbridging/ ; pip wheel: site-packages/planetbridging/
PACKAGE_OR_REPO = EXAMPLES_DIR.parent


def is_dev_checkout() -> bool:
    """Running from a planetbridging git clone (not an installed wheel)."""
    return (PACKAGE_OR_REPO / "src" / "planetbridging").is_dir() and (PACKAGE_OR_REPO / "go.mod").is_file()


def ensure_imports() -> None:
    """Editable dev install: prefer src/ before site-packages."""
    if not is_dev_checkout():
        return
    src = PACKAGE_OR_REPO / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def output_dir(*parts: str) -> Path:
    """Writable cache under cwd (never writes into site-packages)."""
    base = Path.cwd() / ".planetbridging" / "examples"
    path = base.joinpath(*parts) if parts else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_loom_stream() -> Path:
    """Resolve loom-stream (bundled in pip wheel, or bin/ in dev checkout)."""
    ensure_imports()
    from planetbridging._binary import find_loom_stream

    try:
        return find_loom_stream()
    except FileNotFoundError as exc:
        print(exc)
        if is_dev_checkout():
            print("\nDev checkout:  go build -o bin/loom-stream ./cmd/loom-stream/")
        else:
            print("\nReinstall:  pip install --force-reinstall planetbridging")
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
