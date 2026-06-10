"""Loom-compatible Embedding forward (token lookup)."""

from __future__ import annotations

import numpy as np


def init_embedding_weights(vocab_size: int, embed_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((vocab_size, embed_dim), dtype=np.float32) * 0.02
    return {"table": table}


def loom_embedding_forward(
    token_ids: np.ndarray,
    *,
    table: np.ndarray,
) -> np.ndarray:
    """token_ids [N, seq] -> [N, seq*embed_dim] matching Loom EmbeddingForwardPolymorphic."""
    ids = np.asarray(token_ids, dtype=np.int64)
    table = np.asarray(table, dtype=np.float64)
    n, seq = ids.shape
    dim = table.shape[1]
    out = np.zeros((n, seq, dim), dtype=np.float64)
    for i in range(n):
        for t in range(seq):
            tid = int(ids[i, t])
            if 0 <= tid < table.shape[0]:
                out[i, t] = table[tid]
    return out.reshape(n, -1)
