import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cache_decode import cached_greedy
from speculative_decode import speculative_greedy


class TransitionModel:
    """Toy causal model with a controllable greedy next-token transition."""
    def __init__(self, delta=1, vocab=17):
        self.delta = delta
        self.vocab = vocab

    def __call__(self, input_ids, attention_mask=None, cache_position=None,
                 past_key_values=None, use_cache=False, return_dict=True, **kwargs):
        context = input_ids if past_key_values is None else torch.cat((past_key_values, input_ids), dim=-1)
        logits = torch.full((1, input_ids.shape[-1], self.vocab), -100.0)
        for i in range(input_ids.shape[-1]):
            nxt = int(input_ids[0, i]) + self.delta
            logits[0, i, nxt % self.vocab] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=context if use_cache else None)


class SpeculativeDecodeTests(unittest.TestCase):
    def test_all_draft_tokens_accepted_and_matches_greedy(self):
        prompt = torch.tensor([[2, 3, 4]])
        target, draft = TransitionModel(), TransitionModel()
        result = speculative_greedy(target, draft, prompt, 11, draft_tokens=4)
        reference = cached_greedy(TransitionModel(), prompt, 11)
        self.assertEqual(result.generated_token_ids, reference.generated_token_ids)
        self.assertEqual(result.accepted_draft_tokens, result.drafted_tokens)
        self.assertEqual(result.acceptance_rate, 1.0)
        self.assertEqual(result.target_cache_lengths, result.draft_cache_lengths)

    def test_rejection_uses_target_correction_and_resynchronizes(self):
        prompt = torch.tensor([[6, 7]])
        result = speculative_greedy(TransitionModel(), TransitionModel(delta=2), prompt,
                                    9, draft_tokens=3)
        reference = cached_greedy(TransitionModel(), prompt, 9)
        self.assertEqual(result.generated_token_ids, reference.generated_token_ids)
        self.assertGreater(result.rejected_rounds, 0)
        self.assertEqual(result.accepted_draft_tokens, 0)
        self.assertEqual(result.target_cache_lengths, result.draft_cache_lengths)

    def test_eos_candidate_keeps_caches_aligned(self):
        result = speculative_greedy(TransitionModel(), TransitionModel(),
                                    torch.tensor([[3]]), 8, draft_tokens=4,
                                    eos_token_id=4)
        self.assertEqual(result.generated_token_ids, [4])
        self.assertEqual(result.target_cache_lengths, result.draft_cache_lengths)

    def test_rejects_invalid_batch_or_draft_length(self):
        with self.assertRaises(ValueError):
            speculative_greedy(TransitionModel(), TransitionModel(), torch.ones((2, 3), dtype=torch.long), 4)
        with self.assertRaises(ValueError):
            speculative_greedy(TransitionModel(), TransitionModel(), torch.ones((1, 3), dtype=torch.long), 4, 0)


if __name__ == "__main__":
    unittest.main()
