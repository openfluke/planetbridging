from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VariantResult:
    planet: str
    stage: str
    format: str
    outputs: np.ndarray
    artifact_paths: list[str]
    train_skipped: bool = False
