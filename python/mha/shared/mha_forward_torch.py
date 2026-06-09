"""Torch MHA forward matching loom_mha_forward (causal + RoPE)."""

from __future__ import annotations

import math

import torch

from .spec import ROPE_THETA


def _apply_rope(vec: torch.Tensor, pos: int, num_heads: int, head_dim: int, theta: float) -> torch.Tensor:
    out = vec.clone()
    half = head_dim // 2
    for h in range(num_heads):
        base = h * head_dim
        for d in range(half):
            angle = pos / (theta ** (2 * d / head_dim))
            c, s = math.cos(angle), math.sin(angle)
            v0 = out[base + d].item()
            v1 = out[base + d + half].item()
            out[base + d] = v0 * c - v1 * s
            out[base + d + half] = v0 * s + v1 * c
    return out


def loom_mha_forward_torch(
    x: torch.Tensor,
    *,
    q_proj: torch.nn.Linear,
    k_proj: torch.nn.Linear,
    v_proj: torch.nn.Linear,
    o_proj: torch.nn.Linear,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float = ROPE_THETA,
) -> torch.Tensor:
    """[batch, seq, d_model] -> [batch, seq, d_model]. Mirrors shared/mha_forward.py."""
    batch, seq_len, _ = x.shape
    q_dim = num_heads * head_dim
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    outs = []
    for bi in range(batch):
        cache_k: list[torch.Tensor] = []
        cache_v: list[torch.Tensor] = []
        rows = []
        for s in range(seq_len):
            q = _apply_rope(q_proj(x[bi, s]), s, num_heads, head_dim, rope_theta)
            k = _apply_rope(k_proj(x[bi, s]), s, num_kv_heads, head_dim, rope_theta)
            v = v_proj(x[bi, s])
            cache_k.append(k)
            cache_v.append(v)

            attn_parts = []
            for h in range(num_heads):
                kv_h = h // heads_per_kv
                scores = []
                for kp in range(s + 1):
                    dot = torch.dot(
                        q[h * head_dim : (h + 1) * head_dim],
                        cache_k[kp][kv_h * head_dim : (kv_h + 1) * head_dim],
                    )
                    scores.append(dot * scale)
                smax = torch.stack(scores).max()
                exp_s = torch.exp(torch.stack(scores) - smax)
                denom = exp_s.sum()
                for d in range(head_dim):
                    acc = torch.zeros((), dtype=x.dtype, device=x.device)
                    for kp in range(s + 1):
                        acc = acc + exp_s[kp] * cache_v[kp][kv_h * head_dim + d]
                    attn_parts.append(acc / denom)
            rows.append(o_proj(torch.stack(attn_parts)))
        outs.append(torch.stack(rows))
    return torch.stack(outs)
