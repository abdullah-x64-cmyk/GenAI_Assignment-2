"""Reference and tiled online-softmax scaled dot-product attention.

Tensor layout is [batch, heads, sequence, feature]. Boolean masks use True
for positions that may be attended to. Floating masks are additive score
biases (use -inf to block a position). Causal attention is bottom-right
aligned when query and key lengths differ, as for a cached query suffix.
"""
from __future__ import annotations

import math

import torch


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have [batch, heads, sequence, feature] shape")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("batch/head dimensions and key/value sequence lengths must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if q.device != k.device or k.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")


def _scores(q: torch.Tensor, k: torch.Tensor, scale: float,
            attn_mask: torch.Tensor | None, causal: bool,
            q_start: int = 0, k_start: int = 0) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, -torch.inf)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype)
    if causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        # Align the last query with the last key for suffix-query/cache use.
        offset = max(k_len - (q_len + q_start), 0) + q_start - k_start
        q_pos = torch.arange(q_len, device=q.device)[:, None] + offset
        k_pos = torch.arange(k_len, device=q.device)[None, :] + k_start
        scores = scores.masked_fill(k_pos > q_pos, -torch.inf)
    return scores


def reference_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Materialized scaled dot-product attention reference.

    The full [B,H,Q,K] score and probability tensors are materialized. Fully
    masked query rows produce zeros, matching common SDPA behavior.
    """
    _validate(q, k, v)
    output_dtype = q.dtype
    if q.dtype in (torch.float16, torch.bfloat16):
        q, k, v = q.float(), k.float(), v.float()
    factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    scores = _scores(q, k, factor, attn_mask, causal)
    row_max = scores.amax(dim=-1, keepdim=True)
    valid = torch.isfinite(row_max)
    safe_max = torch.where(valid, row_max, torch.zeros_like(row_max))
    weights = torch.exp(scores - safe_max)
    weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
    denom = weights.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, weights / denom.clamp_min(torch.finfo(weights.dtype).tiny),
                        torch.zeros_like(weights))
    return torch.matmul(probs, v).to(output_dtype)


def tiled_online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tile_q: int = 32,
    tile_k: int = 64,
    attn_mask: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute attention in query/key tiles using online log-sum-exp updates.

    For each query tile, running row maxima m, normalizers l, and weighted
    value accumulator a are updated as each key/value tile is visited:
      m' = max(m, max(s)); alpha = exp(m-m')
      l' = alpha*l + sum(exp(s-m')); a' = alpha*a + exp(s-m')@v
    The final result is a/l. Only one score tile is live at a time, reducing
    score workspace from O(B*H*Q*K) to O(B*H*tile_q*tile_k).
    """
    _validate(q, k, v)
    if tile_q <= 0 or tile_k <= 0:
        raise ValueError("tile_q and tile_k must be positive")
    output_dtype = q.dtype
    if q.dtype in (torch.float16, torch.bfloat16):
        q, k, v = q.float(), k.float(), v.float()
    bsz, heads, q_len, _ = q.shape
    k_len = k.shape[-2]
    value_dim = v.shape[-1]
    factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    out = torch.zeros((bsz, heads, q_len, value_dim), device=q.device, dtype=q.dtype)

    for qs in range(0, q_len, tile_q):
        qe = min(qs + tile_q, q_len)
        q_block = q[:, :, qs:qe, :]
        rows = qe - qs
        running_max = torch.full((bsz, heads, rows, 1), -torch.inf, device=q.device, dtype=q.dtype)
        running_sum = torch.zeros_like(running_max)
        running_value = torch.zeros((bsz, heads, rows, value_dim), device=q.device, dtype=q.dtype)

        for ks in range(0, k_len, tile_k):
            ke = min(ks + tile_k, k_len)
            k_block = k[:, :, ks:ke, :]
            v_block = v[:, :, ks:ke, :]
            block_mask = None
            if attn_mask is not None:
                # Broadcasting to the local block handles [Q,K], [B,1,Q,K],
                # and any other standard broadcastable SDPA mask.
                block_mask = attn_mask[..., qs:qe, ks:ke]
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * factor
            if block_mask is not None:
                if block_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~block_mask, -torch.inf)
                else:
                    scores = scores + block_mask.to(dtype=scores.dtype)
            if causal:
                # Bottom-right alignment for a query suffix into a longer KV.
                offset = max(k_len - q_len, 0)
                q_positions = torch.arange(qs, qe, device=q.device) + offset
                k_positions = torch.arange(ks, ke, device=q.device)
                scores = scores.masked_fill(k_positions[None, :] > q_positions[:, None], -torch.inf)

            tile_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, tile_max)
            row_valid = torch.isfinite(new_max)
            safe_new_max = torch.where(row_valid, new_max, torch.zeros_like(new_max))
            alpha = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - safe_new_max),
                torch.zeros_like(running_max),
            )
            exp_scores = torch.exp(scores - safe_new_max)
            exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
            running_sum = alpha * running_sum + exp_scores.sum(dim=-1, keepdim=True)
            running_value = alpha * running_value + torch.matmul(exp_scores, v_block)
            running_max = new_max

        out[:, :, qs:qe, :] = torch.where(
            running_sum > 0,
            running_value / running_sum.clamp_min(torch.finfo(running_sum.dtype).tiny),
            torch.zeros_like(running_value),
        )
    return out.to(output_dtype)
