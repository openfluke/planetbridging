"""Resolve planetbridging data root and platform tags (dev checkout vs PyPI wheel)."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def package_dir() -> Path:
    """Installed ``planetbridging`` package directory."""
    return Path(__file__).resolve().parent


def platform_tag() -> str:
    """Directory name for bundled native binaries (matches welvet layout)."""
    plat = sys.platform
    arch = platform.machine().lower()
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    arch_key = arch_map.get(arch, arch)
    if plat.startswith("linux"):
        return f"linux_{arch_key}"
    if plat == "darwin":
        bundled = package_dir() / "_bin" / f"macos_{arch_key}"
        if (bundled / "loom-stream").is_file():
            return f"macos_{arch_key}"
        universal = package_dir() / "_bin" / "macos_universal" / "loom-stream"
        if universal.is_file():
            return "macos_universal"
        return f"macos_{arch_key}"
    if plat == "win32":
        return f"windows_{arch_key}"
    return f"{plat}_{arch_key}"


def repo_root() -> Path:
    """Root containing ``python/<bedrock>/`` (dev repo or wheel ``_data/``)."""
    env = os.environ.get("PLANETBRIDGING_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if (root / "python" / "dense" / "manifest.yaml").is_file():
            return root
        raise FileNotFoundError(f"PLANETBRIDGING_ROOT invalid (no python/dense): {root}")

    pkg = package_dir()
    bundled = pkg / "_data"
    if (bundled / "python" / "dense" / "manifest.yaml").is_file():
        return bundled

    # Editable / monorepo dev: .../planetbridging/src/planetbridging → repo root
    dev = pkg.parents[1]
    if (dev / "python" / "dense" / "manifest.yaml").is_file():
        return dev

    raise FileNotFoundError(
        "planetbridging bedrock data not found. Reinstall with:\n"
        "  pip install --force-reinstall planetbridging\n"
        "Or set PLANETBRIDGING_ROOT to a planetbridging repo checkout."
    )


def python_bedrock_dir(bedrock: str) -> Path:
    return repo_root() / "python" / bedrock.lower()


def bundled_loom_stream() -> Path | None:
    """Platform loom-stream inside the wheel, if present."""
    name = "loom-stream.exe" if sys.platform == "win32" else "loom-stream"
    pkg = package_dir()
    tag = platform_tag()
    candidate = pkg / "_bin" / tag / name
    if candidate.is_file():
        return candidate
    # Fallback: scan any bundled platform (e.g. partial wheel)
    bin_root = pkg / "_bin"
    if bin_root.is_dir():
        for sub in sorted(bin_root.iterdir()):
            alt = sub / name
            if alt.is_file():
                return alt
    return None


def bundled_platforms() -> tuple[str, ...]:
    """Platform tags shipped in this install (for diagnostics)."""
    pkg = package_dir() / "_bin"
    if not pkg.is_dir():
        return ()
    name = "loom-stream.exe" if sys.platform == "win32" else "loom-stream"
    return tuple(
        sorted(
            d.name
            for d in pkg.iterdir()
            if d.is_dir() and (d / name).is_file()
        )
    )
