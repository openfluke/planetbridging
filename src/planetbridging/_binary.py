"""Locate the loom-stream Go binary (stdlib subprocess backend)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path

from ._paths import bundled_loom_stream, bundled_platforms, package_dir, platform_tag, repo_root

__all__ = ["repo_root", "default_binary_path", "find_loom_stream", "run_loom_stream"]


def default_binary_path() -> Path:
    bundled = bundled_loom_stream()
    if bundled is not None:
        return bundled
    name = "loom-stream.exe" if platform.system() == "Windows" else "loom-stream"
    return repo_root() / "bin" / name


def find_loom_stream(explicit: str | Path | None = None) -> Path:
    """Resolve loom-stream: explicit → env → bundled wheel → PATH → dev bin/."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"loom-stream not found: {path}")
        return path

    env = os.environ.get("PLANETBRIDGING_LOOM_STREAM")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path

    bundled = bundled_loom_stream()
    if bundled is not None:
        return bundled

    which = shutil.which("loom-stream")
    if which:
        return Path(which)

    name = "loom-stream.exe" if platform.system() == "Windows" else "loom-stream"
    dev_bin = repo_root() / "bin" / name
    if dev_bin.is_file():
        return dev_bin

    tag = platform_tag()
    platforms = ", ".join(bundled_platforms()) or "(none)"
    raise FileNotFoundError(
        "loom-stream binary not found for this platform.\n"
        f"  wanted: planetbridging/_bin/{tag}/{name}\n"
        f"  bundled: {platforms}\n"
        "  Options:\n"
        "    pip install --force-reinstall planetbridging\n"
        "    go build -o bin/loom-stream ./cmd/loom-stream/\n"
        "    export PLANETBRIDGING_LOOM_STREAM=/path/to/loom-stream"
    )


def ensure_executable(path: Path) -> None:
    if not os.access(path, os.X_OK):
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_loom_stream(
    *,
    binary: Path | None,
    envelope: dict,
    payload: dict,
    timeout: float = 120.0,
) -> dict:
    """Invoke loom-stream with JSON on stdin, parse JSON stdout."""
    exe = find_loom_stream(binary)
    ensure_executable(exe)

    body = {**envelope, "payload": payload}
    proc = subprocess.run(
        [str(exe)],
        input=json.dumps(body).encode("utf-8"),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or f"exit {proc.returncode}"
        raise RuntimeError(f"loom-stream failed: {detail}")

    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"loom-stream returned invalid JSON: {proc.stdout!r}") from exc
