"""Locate the loom-stream Go binary (stdlib subprocess backend)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path


def repo_root() -> Path:
    """Planetbridging repo root (../../ from src/planetbridging/)."""
    return Path(__file__).resolve().parents[2]


def default_binary_path() -> Path:
    name = "loom-stream.exe" if platform.system() == "Windows" else "loom-stream"
    return repo_root() / "bin" / name


def find_loom_stream(explicit: str | Path | None = None) -> Path:
    """Resolve loom-stream: explicit → env → PATH → repo bin/."""
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

    which = shutil.which("loom-stream")
    if which:
        return Path(which)

    candidate = default_binary_path()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        "loom-stream binary not found. Build from the planetbridging repo:\n"
        "  go build -o bin/loom-stream ./cmd/loom-stream/\n"
        "Or set PLANETBRIDGING_LOOM_STREAM=/path/to/loom-stream"
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
