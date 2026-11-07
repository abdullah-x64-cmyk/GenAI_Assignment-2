import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_attention import (
    reference_scaled_dot_product_attention,
    tiled_online_softmax_attention,
)


class OnlineAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.q = torch.randn(2, 3, 11, 8, dtype=torch.float64)
        self.k = torch.randn(2, 3, 11, 8, dtype=torch.float64)
        self.v = torch.randn(2, 3, 11, 5, dtype=torch.float64)

    def test_tiled_matches_reference_causal_uneven_tiles(self):
        expected = reference_scaled_dot_product_attention(self.q, self.k, self.v, causal=True)
        for tq, tk in ((1, 1), (3, 4), (5, 7), (16, 32)):
            actual = tiled_online_softmax_attention(
                self.q, self.k, self.v, causal=True, tile_q=tq, tile_k=tk
            )
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_matches_pytorch_sdpa_causal(self):
        expected = F.scaled_dot_product_attention(self.q, self.k, self.v, is_causal=True)
        actual = tiled_online_softmax_attention(self.q, self.k, self.v, causal=True,
                                                 tile_q=4, tile_k=3)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_boolean_and_additive_masks(self):
        mask = torch.ones(11, 11, dtype=torch.bool).tril()
        mask[4, :] = False
        expected = reference_scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=mask)
        actual = tiled_online_softmax_attention(self.q, self.k, self.v, attn_mask=mask,
                                                 tile_q=4, tile_k=3)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        additive = torch.zeros(11, 11, dtype=torch.float64).masked_fill(~mask, -torch.inf)
        actual_additive = tiled_online_softmax_attention(
            self.q, self.k, self.v, attn_mask=additive, tile_q=4, tile_k=3
        )
        torch.testing.assert_close(actual_additive, expected, rtol=1e-10, atol=1e-10)

    def test_fully_masked_query_returns_zero(self):
        mask = torch.ones(11, 11, dtype=torch.bool)
        mask[2, :] = False
        output = tiled_online_softmax_attention(self.q, self.k, self.v, attn_mask=mask)
        torch.testing.assert_close(output[:, :, 2], torch.zeros_like(output[:, :, 2]))

    def test_suffix_queries_use_bottom_right_causal_alignment(self):
        q = self.q[:, :, -3:, :]
        expected = F.scaled_dot_product_attention(
            q, self.k, self.v,
            attn_mask=(torch.arange(11)[None, :] <= torch.arange(8, 11)[:, None])
        )
        actual = tiled_online_softmax_attention(q, self.k, self.v, causal=True,
                                                 tile_q=2, tile_k=4)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_invalid_tile_size_is_rejected(self):
        with self.assertRaises(ValueError):
            tiled_online_softmax_attention(self.q, self.k, self.v, tile_q=0)

    def test_half_precision_uses_stable_float32_reductions(self):
        q, k, v = self.q.float().half(), self.k.float().half(), self.v.float().half()
        expected = reference_scaled_dot_product_attention(q, k, v, causal=True)
        actual = tiled_online_softmax_attention(q, k, v, causal=True, tile_q=3, tile_k=4)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
