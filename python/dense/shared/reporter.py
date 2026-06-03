"""Push 100-sample outputs to the planetbridging compare-host."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from .spec import DEFAULT_HOST


def outputs_to_json(outputs: np.ndarray) -> list[list[float]]:
    arr = np.asarray(outputs, dtype=np.float64)
    return arr.tolist()


def post_report(
    *,
    host: str,
    planet: str,
    stage: str,
    format: str,
    model_id: str,
    framework_version: str,
    fixture_version: str,
    input_dim: int,
    output_dim: int,
    outputs: np.ndarray,
    artifact_paths: list[str] | None = None,
    train_skipped: bool = False,
) -> dict[str, Any]:
    host = host.rstrip("/")
    payload = {
        "planet": planet,
        "stage": stage,
        "format": format,
        "engine": planet,
        "model_id": model_id,
        "framework_version": framework_version,
        "fixture_version": fixture_version,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "sample_count": int(outputs.shape[0]),
        "outputs": outputs_to_json(outputs),
        "artifact_paths": artifact_paths or [],
        "train_skipped": train_skipped,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/v1/report",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"compare-host unreachable at {host}; start it with "
            f"'go run .' from planetbridging root"
        ) from exc


def host_reachable(host: str = DEFAULT_HOST) -> bool:
    host = host.rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False
