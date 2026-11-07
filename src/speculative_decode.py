"""Greedy speculative decoding with explicit draft/target KV-cache handling.

The draft proposes a short block. The target evaluates that block in one call,
then we accept the matching greedy prefix. On rejection, both caches are
cropped to the accepted prefix and the target's replacement token is fed to
both models to restore synchronized state.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from cache_decode import _cache_length, _greedy, _sync


@dataclass
class SpeculativeResult:
    prompt_tokens: int
    generated_token_ids: list[int]
    prefill_seconds: float
    decode_seconds: float
    drafted_tokens: int
    accepted_draft_tokens: int
    rejected_rounds: int
    rounds: int
    target_cache_lengths: list[int]
    draft_cache_lengths: list[int]

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_draft_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def tokens_per_second(self) -> float:
        return len(self.generated_token_ids) / self.decode_seconds if self.decode_seconds else 0.0


def _crop_cache(cache: Any, length: int) -> Any:
    """Roll a Transformers cache or legacy tuple back to an accepted prefix."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    if isinstance(cache, (tuple, list)):
        layers = []
        for layer in cache:
            # Qwen2 legacy entries are (key, value), with sequence at -2.
            layers.append(tuple(x[..., :length, :] if isinstance(x, torch.Tensor) and x.ndim >= 3
                                else x for x in layer))
        return tuple(layers)
    if isinstance(cache, torch.Tensor):
        return cache[..., :length]
    raise TypeError(f"unsupported key/value cache type: {type(cache).__name__}")


def _step(model: Any, token_ids: torch.Tensor, cache: Any, mask: torch.Tensor,
          cache_position: torch.Tensor):
    out = model(input_ids=token_ids, attention_mask=mask,
                cache_position=cache_position, past_key_values=cache,
                use_cache=True, return_dict=True)
    return out.past_key_values, out.logits


def speculative_greedy(target: Any, draft: Any, input_ids: torch.Tensor,
                       max_new_tokens: int, draft_tokens: int = 4,
                       eos_token_id: int | None = None) -> SpeculativeResult:
    """Greedy speculative generation for one unpadded prompt.

    Both models must have the same vocabulary/token IDs. `draft_tokens` is the
    maximum number of candidate tokens per target verification call.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("speculative decoding expects one unpadded prompt (shape [1, seq])")
    if max_new_tokens < 0 or draft_tokens < 1:
        raise ValueError("max_new_tokens must be nonnegative and draft_tokens must be positive")
    device = input_ids.device
    prompt_len = input_ids.shape[-1]
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    target_cache_lengths: list[int] = []
    draft_cache_lengths: list[int] = []
    drafted = accepted_total = rejected = rounds = 0

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        t_out = target(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=True, return_dict=True)
        d_out = draft(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=True, return_dict=True)
        t_cache, d_cache = t_out.past_key_values, d_out.past_key_values
        t_next, d_next = _greedy(t_out.logits), _greedy(d_out.logits)
    if _cache_length(t_cache) != prompt_len or _cache_length(d_cache) != prompt_len:
        raise AssertionError("target and draft prefill cache lengths must equal prompt length")
    target_cache_lengths.append(prompt_len)
    draft_cache_lengths.append(prompt_len)
    _sync(device)
    prefill = time.perf_counter() - start

    _sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            rounds += 1
            round_limit = min(draft_tokens, max_new_tokens - len(generated))
            context_len = _cache_length(d_cache)
            candidates: list[torch.Tensor] = []
            # Draft autoregressively; caches include all candidate tokens.
            for _ in range(round_limit):
                candidate = d_next
                candidates.append(candidate)
                drafted += 1
                next_pos = _cache_length(d_cache)
                new_mask = torch.ones((1, next_pos + 1), device=device, dtype=attention_mask.dtype)
                d_cache, d_logits = _step(
                    draft, candidate, d_cache, new_mask,
                    torch.tensor([next_pos], device=device, dtype=torch.long),
                )
                d_next = _greedy(d_logits)
                if eos_token_id is not None and int(candidate.item()) == eos_token_id:
                    break

            candidate_ids = torch.cat(candidates, dim=-1)
            n_candidates = candidate_ids.shape[-1]
            # Verify every proposed token in one target call. Its previous
            # next-token distribution scores candidate zero; each following
            # position's logits score the next candidate.
            verify_mask = torch.ones((1, context_len + n_candidates), device=device,
                                     dtype=attention_mask.dtype)
            positions = torch.arange(context_len, context_len + n_candidates,
                                     device=device, dtype=torch.long)
            t_cache, verify_logits = _step(target, candidate_ids, t_cache, verify_mask, positions)

            accepted = 0
            correction: torch.Tensor | None = None
            for index, candidate in enumerate(candidates):
                target_choice = t_next if index == 0 else _greedy(verify_logits[:, index - 1:index, :])
                if torch.equal(candidate, target_choice):
                    accepted += 1
                    generated.append(int(candidate.item()))
                    if eos_token_id is not None and int(candidate.item()) == eos_token_id:
                        break
                else:
                    correction = target_choice
                    break
                if len(generated) >= max_new_tokens:
                    break

            accepted_total += accepted
            rejected_this_round = accepted < n_candidates and correction is not None
            if rejected_this_round:
                rejected += 1
                kept_len = context_len + accepted
                t_cache = _crop_cache(t_cache, kept_len)
                d_cache = _crop_cache(d_cache, kept_len)
                # Replace the first rejected token with the target's greedy
                # choice and advance both models so caches remain synchronized.
                if len(generated) < max_new_tokens:
                    generated.append(int(correction.item()))
                    if eos_token_id is None or int(correction.item()) != eos_token_id:
                        next_pos = kept_len
                        corr_mask = torch.ones((1, next_pos + 1), device=device, dtype=attention_mask.dtype)
                        t_cache, t_logits = _step(target, correction, t_cache, corr_mask,
                                                  torch.tensor([next_pos], device=device, dtype=torch.long))
                        d_cache, d_logits = _step(draft, correction, d_cache, corr_mask,
                                                  torch.tensor([next_pos], device=device, dtype=torch.long))
                        t_next, d_next = _greedy(t_logits), _greedy(d_logits)
            else:
                # The whole candidate block matched. The target's final
                # verification logits provide the extra greedy bonus token.
                if len(generated) < max_new_tokens and (eos_token_id is None or generated[-1] != eos_token_id):
                    bonus = _greedy(verify_logits[:, n_candidates - 1:n_candidates, :])
                    generated.append(int(bonus.item()))
                    if len(generated) < max_new_tokens and (eos_token_id is None or int(bonus.item()) != eos_token_id):
                        next_pos = _cache_length(t_cache)
                        bonus_mask = torch.ones((1, next_pos + 1), device=device, dtype=attention_mask.dtype)
                        t_cache, t_logits = _step(target, bonus, t_cache, bonus_mask,
                                                  torch.tensor([next_pos], device=device, dtype=torch.long))
                        d_cache, d_logits = _step(draft, bonus, d_cache, bonus_mask,
                                                  torch.tensor([next_pos], device=device, dtype=torch.long))
                        t_next, d_next = _greedy(t_logits), _greedy(d_logits)
                    else:
                        t_next = d_next = bonus
                elif n_candidates and len(generated) < max_new_tokens:
                    # No bonus is emitted when a candidate was EOS.
                    t_next = _greedy(verify_logits[:, -1:, :])
                    d_next = d_next

            if _cache_length(t_cache) != _cache_length(d_cache):
                raise AssertionError("target and draft cache lengths diverged")
            target_cache_lengths.append(_cache_length(t_cache))
            draft_cache_lengths.append(_cache_length(d_cache))
            if eos_token_id is not None and generated and generated[-1] == eos_token_id:
                break

    _sync(device)
    decode = time.perf_counter() - start
    return SpeculativeResult(prompt_len, generated, prefill, decode, drafted,
                             accepted_total, rejected, rounds,
                             target_cache_lengths, draft_cache_lengths)
