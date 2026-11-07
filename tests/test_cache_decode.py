import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cache_decode import cached_greedy, uncached_greedy


class TinyCausalModel:
    """Deterministic toy model that exposes cache growth for loop testing."""
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, position_ids=None, cache_position=None,
                 past_key_values=None, use_cache=False, return_dict=True):
        self.calls.append({
            "input_length": input_ids.shape[-1],
            "past_length": 0 if past_key_values is None else past_key_values.shape[-1],
            "use_cache": use_cache,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "mask_length": None if attention_mask is None else attention_mask.shape[-1],
        })
        sequence = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], dim=-1)
        next_id = (sequence[:, -1] + 1) % 7
        logits = torch.full((1, input_ids.shape[-1], 7), -100.0)
        logits[:, -1, next_id.item()] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=sequence if use_cache else None)


class CacheDecodeTests(unittest.TestCase):
    def test_cached_and_full_prefix_greedy_tokens_agree(self):
        for prompt_length in (1, 4, 9):
            for output_length in (1, 3, 6):
                ids = (torch.arange(prompt_length).remainder(7) + 2).reshape(1, -1)
                cached = cached_greedy(TinyCausalModel(), ids, output_length)
                reference = uncached_greedy(TinyCausalModel(), ids, output_length)
                self.assertEqual(cached.generated_token_ids, reference.generated_token_ids)

    def test_cached_path_prefills_once_then_feeds_one_token(self):
        model = TinyCausalModel()
        result = cached_greedy(model, torch.tensor([[1, 2, 3, 4]]), 4)
        self.assertEqual(len(result.generated_token_ids), 4)
        self.assertEqual([call["input_length"] for call in model.calls], [4, 1, 1, 1])
        self.assertEqual([call["past_length"] for call in model.calls], [0, 4, 5, 6])
        self.assertEqual([call["mask_length"] for call in model.calls], [4, 5, 6, 7])
        self.assertEqual([call["position_ids"].item() for call in model.calls[1:]], [4, 5, 6])
        self.assertEqual([call["cache_position"].item() for call in model.calls[1:]], [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
