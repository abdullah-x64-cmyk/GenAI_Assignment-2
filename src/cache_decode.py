"""Explicit greedy decoding with and without incremental KV-cache reuse.

The cached path runs the full prompt once, then feeds exactly the newly
generated token together with the returned ``past_key_values``. The reference
path recomputes logits for the entire prompt-plus-generated prefix at every
step. Both paths use greedy argmax and identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import torch


@dataclass
class DecodeResult:
    prompt_tokens: int
    generated_token_ids: list[int]
    prefill_seconds: float
    decode_seconds: float
    cache_sequence_lengths: list[int] | None = None

    @property
    def tokens_per_second(self) -> float:
        return len(self.generated_token_ids) / self.decode_seconds if self.decode_seconds else 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _greedy(logits: torch.Tensor) -> torch.Tensor:
    # argmax has deterministic tie handling (first maximum index), shared by
    # both implementations. No sampling or logits processors are applied.
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def _cache_length(cache: Any) -> int:
    """Inspect DynamicCache or legacy tuple caches to verify actual growth."""
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, torch.Tensor):  # deterministic unit-test cache
        return int(cache.shape[-1])
    if isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        key = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer
        if isinstance(key, torch.Tensor) and key.ndim >= 3:
            return int(key.shape[-2])
    raise TypeError(f"unsupported key/value cache type: {type(cache).__name__}")


def cached_greedy(model: Any, input_ids: torch.Tensor, max_new_tokens: int,
                  eos_token_id: int | None = None) -> DecodeResult:
    """Greedy decode with explicit ``past_key_values`` reuse and cache growth."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("cache benchmark expects one unpadded prompt (shape [1, seq])")
    device = input_ids.device
    generated: list[int] = []
    attention_mask = torch.ones_like(input_ids)

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        # Prefill: compute K/V for all prompt positions in one forward pass.
        output = model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=True, return_dict=True)
        next_token = _greedy(output.logits)
        past_key_values = output.past_key_values
    cache_sequence_lengths = [_cache_length(past_key_values)]
    if cache_sequence_lengths[-1] != input_ids.shape[-1]:
        raise AssertionError(f"prefill cache has length {cache_sequence_lengths[-1]}, expected {input_ids.shape[-1]}")
    _sync(device)
    prefill = time.perf_counter() - start

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            if step + 1 == max_new_tokens:
                break
            # Append the current token to the key mask. The model's causal
            # mask plus cache_position/position_ids place it after cached tokens.
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=-1
            )
            position_ids = torch.tensor([[input_ids.shape[-1] + step]], device=device, dtype=torch.long)
            cache_position = torch.tensor([input_ids.shape[-1] + step], device=device, dtype=torch.long)
            output = model(input_ids=next_token, attention_mask=attention_mask,
                           position_ids=position_ids, cache_position=cache_position,
                           past_key_values=past_key_values,
                           use_cache=True, return_dict=True)
            past_key_values = output.past_key_values
            actual_cache_length = _cache_length(past_key_values)
            expected_cache_length = input_ids.shape[-1] + step + 1
            cache_sequence_lengths.append(actual_cache_length)
            if actual_cache_length != expected_cache_length:
                raise AssertionError(f"KV cache length {actual_cache_length}, expected {expected_cache_length}")
            next_token = _greedy(output.logits)
    _sync(device)
    decode = time.perf_counter() - start
    return DecodeResult(input_ids.shape[-1], generated, prefill, decode, cache_sequence_lengths)


def uncached_greedy(model: Any, input_ids: torch.Tensor, max_new_tokens: int,
                    eos_token_id: int | None = None) -> DecodeResult:
    """Reference greedy path: recompute the complete prefix each step."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("cache benchmark expects one unpadded prompt (shape [1, seq])")
    device = input_ids.device
    prefix = input_ids
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=prefix, attention_mask=attention_mask,
                       use_cache=False, return_dict=True)
        next_token = _greedy(output.logits)
    _sync(device)
    prefill = time.perf_counter() - start

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            token_id = int(next_token.item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            if step + 1 == max_new_tokens:
                break
            prefix = torch.cat([prefix, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=-1
            )
            output = model(input_ids=prefix, attention_mask=attention_mask,
                           use_cache=False, return_dict=True)
            next_token = _greedy(output.logits)
    _sync(device)
    decode = time.perf_counter() - start
    return DecodeResult(input_ids.shape[-1], generated, prefill, decode)
