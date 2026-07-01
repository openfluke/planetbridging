"""Run POC bedrock engine handlers without compare-host HTTP."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ._binary import repo_root
from ._fixtures import load_fixture_arrays
from .bedrock_smoke import _bedrock_pkg, _load_bedrock_module
from .stream import stream_bedrock


class _MockHTTPResponse:
    """Minimal urllib response for patched loom stream calls."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@contextmanager
def cli_loom_stream(
    *,
    root: Path | None = None,
    binary: Path | None = None,
    out_dir: Path | None = None,
) -> Iterator[None]:
    """Redirect POC ``post_*_stream`` HTTP calls to ``stream_bedrock`` (no :9876)."""
    root = Path(root) if root else repo_root()
    real_urlopen = urllib.request.urlopen

    def patched_urlopen(req: Any, *args: Any, **kwargs: Any) -> _MockHTTPResponse:
        url = str(getattr(req, "full_url", req))
        if "/api/v1/loom/stream" not in url:
            return real_urlopen(req, *args, **kwargs)

        body = json.loads(req.data.decode("utf-8"))
        tail = url.rstrip("/").split("/")[-1].lower()
        bedrock = str(body.get("bedrock") or tail).lower()
        if bedrock in ("", "stream"):
            bedrock = "dense"
        entity_path = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            entity_path = out_dir / f"{body.get('model_id', 'model')}.entity"

        result = stream_bedrock(
            bedrock,
            body,
            fixture_version=str(body.get("fixture_version", "")),
            output_path=entity_path,
            root=root,
            binary=binary,
        )
        return _MockHTTPResponse(
            {
                "status": "ok",
                "entity_path": result.entity_path,
                "layer_count": result.layer_count,
                "weight_bytes": result.weight_bytes,
                "outputs": result.outputs.tolist(),
                "max_abs_diff": result.max_abs_diff,
                "mean_abs_diff": result.mean_abs_diff,
                "exact_match": result.exact_match,
            }
        )

    urllib.request.urlopen = patched_urlopen  # type: ignore[assignment]
    try:
        yield
    finally:
        urllib.request.urlopen = real_urlopen  # type: ignore[assignment]


def _prepare_bedrock_imports(root: Path, bedrock: str) -> Path:
    """Isolate python/<bedrock>/shared imports (same strategy as bedrock_smoke)."""
    bed = root / "python" / bedrock
    if not bed.is_dir():
        raise FileNotFoundError(bed)

    # Drop cached shared.* from other bedrocks.
    for key in list(sys.modules):
        if key == "shared" or key.startswith("shared."):
            del sys.modules[key]

    # Keep only one bedrock root on sys.path front.
    bed_str = str(bed)
    others = _other_bedrocks(bedrock, root)
    sys.path = [p for p in sys.path if not any(p.endswith(f"/python/{b}") for b in others)]
    if bed_str not in sys.path:
        sys.path.insert(0, bed_str)

    _bedrock_pkg(root, bedrock)
    return bed


def _other_bedrocks(current: str, root: Path) -> list[str]:
    py = root / "python"
    if not py.is_dir():
        return []
    return [d.name for d in py.iterdir() if d.is_dir() and d.name != current]


def load_engine_handler(bedrock: str, planet: str, *, root: Path | None = None) -> Any:
    """Import ``handler`` from ``python/<bedrock>/engines/<planet>/run.py``."""
    root = Path(root) if root else repo_root()
    _prepare_bedrock_imports(root, bedrock)
    path = root / "python" / bedrock / "engines" / planet / "run.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"_poc_{bedrock}_{planet}"
    for key in list(sys.modules):
        if key == name or key.startswith(f"{name}."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "handler"):
        raise AttributeError(f"{path} has no handler()")
    return mod.handler


def run_engine_handler(
    bedrock: str,
    planet: str,
    *,
    model_id: str | None = None,
    root: Path | None = None,
    binary: Path | None = None,
    out_dir: Path | None = None,
    skip_loom: bool = False,
) -> list[Any]:
    """Run one POC planet handler (native + optional loom stream via CLI)."""
    root = Path(root) if root else repo_root()
    _prepare_bedrock_imports(root, bedrock)
    manifest_mod = _load_bedrock_module(root, bedrock, "manifest")
    fixtures_mod = _load_bedrock_module(root, bedrock, "fixtures")
    manifest = manifest_mod.load_manifest()
    fv = getattr(manifest, "fixture_version", None)
    data = load_fixture_arrays(bedrock, root, fv)

    model = None
    if model_id:
        model = next((m for m in manifest.models if m.id == model_id), None)
        if model is None:
            raise KeyError(model_id)
    else:
        model = manifest.models[0]

    handler = load_engine_handler(bedrock, planet, root=root)
    models_dir = root / "python" / bedrock / "models"
    entity_dir = out_dir or root / ".planetbridging" / "stream" / bedrock / planet

    with cli_loom_stream(root=root, binary=binary, out_dir=entity_dir):
        return handler(
            model=model,
            manifest=manifest,
            data=data,
            models_dir=models_dir,
            host="http://127.0.0.1:9876",
            skip_loom=skip_loom,
        )
