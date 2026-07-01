"""Bedrock fixture npz — dev checkout paths or user cache for pip installs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .bedrocks import BEDROCK_IDS, FIXTURE_VERSIONS


def fixtures_cache_root() -> Path:
    root = os.environ.get("PLANETBRIDGING_FIXTURES_CACHE")
    if root:
        return Path(root).expanduser().resolve()
    return Path.home() / ".planetbridging" / "fixtures"


def _fixture_npz(bedrock: str, fixture_version: str, directory: Path) -> Path:
    return directory / f"{fixture_version}.npz"


def ensure_bedrock_fixture(bedrock: str, data_root: Path, *, out_dir: Path) -> Path:
    """Generate one bedrock fixture npz into ``out_dir`` (numpy only)."""
    from .bedrock_smoke import _load_bedrock_module

    bedrock = bedrock.lower()
    fv = FIXTURE_VERSIONS[bedrock]
    target = _fixture_npz(bedrock, fv, out_dir)
    if target.is_file():
        return target

    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_mod = _load_bedrock_module(data_root, bedrock, "fixtures")
    manifest_mod = _load_bedrock_module(data_root, bedrock, "manifest")

    # POC modules bind FIXTURES_DIR at import — redirect for read-only pip installs.
    fixtures_mod.FIXTURES_DIR = out_dir
    try:
        spec_mod = _load_bedrock_module(data_root, bedrock, "spec")
        if hasattr(spec_mod, "FIXTURES_DIR"):
            spec_mod.FIXTURES_DIR = out_dir
    except Exception:
        pass

    manifest = manifest_mod.load_manifest()
    if not hasattr(fixtures_mod, "ensure_fixtures"):
        raise RuntimeError(f"{bedrock}: no ensure_fixtures() in POC fixtures module")
    fixtures_mod.ensure_fixtures(manifest)

    if not target.is_file():
        raise RuntimeError(f"{bedrock}: expected fixture not created: {target}")
    return target


def resolve_fixtures_dir(
    bedrock: str,
    data_root: Path,
    fixture_version: str,
) -> Path:
    """Directory containing ``<fixture_version>.npz`` for loom-stream."""
    bedrock = bedrock.lower()
    fv = fixture_version or FIXTURE_VERSIONS[bedrock]

    bundled = _fixture_npz(bedrock, fv, data_root / "python" / bedrock / "fixtures")
    if bundled.is_file():
        return bundled.parent

    cache = fixtures_cache_root() / bedrock
    ensure_bedrock_fixture(bedrock, data_root, out_dir=cache)
    return cache


def load_fixture_arrays(
    bedrock: str,
    data_root: Path,
    fixture_version: str | None = None,
) -> dict[str, np.ndarray]:
    """Load bedrock fixture arrays (bundled npz or user cache)."""
    bedrock = bedrock.lower()
    fv = fixture_version or FIXTURE_VERSIONS[bedrock]
    directory = resolve_fixtures_dir(bedrock, data_root, fv)
    path = directory / f"{fv}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}


def ensure_all_bedrock_fixtures(data_root: Path | None = None) -> None:
    """Warm the fixture cache for every bedrock (first pip use)."""
    from ._paths import repo_root

    root = data_root or repo_root()
    cache = fixtures_cache_root()
    for bedrock in BEDROCK_IDS:
        ensure_bedrock_fixture(bedrock, root, out_dir=cache / bedrock)
