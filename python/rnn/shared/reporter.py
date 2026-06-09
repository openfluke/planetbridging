"""Push RNN reports to compare-host."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import numpy as np

from .spec import BEDROCK, DEFAULT_HOST


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
        "bedrock": BEDROCK,
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
        "outputs": np.asarray(outputs, dtype=np.float64).tolist(),
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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def host_reachable(host: str = DEFAULT_HOST) -> bool:
    host = host.rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False
