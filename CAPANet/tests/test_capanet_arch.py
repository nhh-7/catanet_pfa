import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicsr.archs.capanet_arch import (  # noqa: E402
    CAPABlock,
    CAPANet,
    CATANet,
    DynamicPrototypeRouter,
    LowToHighMultiLevelReconstruction,
    PrototypeCenterInteraction,
    TAB,
    align_attention_state,
    restore_sorted_tokens,
    segment_pool_features,
    segment_pool_labels,
)


class CAPANetModuleTest(unittest.TestCase):
    def test_dynamic_prototype_router_shapes_and_restore(self):
        torch.manual_seed(0)
        router = DynamicPrototypeRouter(dim=40, router_dim=36, num_tokens=8)
        x = torch.randn(2, 17, 40)

        sorted_x, idx_last, labels, scores, prototypes = router(x)

        self.assertEqual(sorted_x.shape, (2, 17, 40))
        self.assertEqual(idx_last.shape, (2, 17, 1))
        self.assertEqual(labels.shape, (2, 17))
        self.assertEqual(scores.shape, (2, 17))
        self.assertEqual(prototypes.shape, (2, 8, 40))

        restored = restore_sorted_tokens(sorted_x, idx_last)
        self.assertEqual(restored.shape, x.shape)
        self.assertTrue(torch.allclose(restored.gather(1, idx_last.expand_as(sorted_x)), sorted_x))
        self.assertTrue(torch.allclose(prototypes.norm(dim=-1), torch.ones_like(scores[:, :8]), atol=1e-4, rtol=1e-4))

    def test_segment_pooling_and_label_tie_break_follow_docs(self):
        x = torch.tensor(
            [[[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]]],
            dtype=torch.float32,
        )
        scores = torch.tensor([[0.1, 0.9, 0.8, 0.2]], dtype=torch.float32)
        labels = torch.tensor([[2, 3, 1, 1]], dtype=torch.long)

        pooled_x, pooled_scores = segment_pool_features(x, scores, scale=2)
        pooled_labels = segment_pool_labels(labels, scores, scale=2)

        expected_x = torch.tensor([[[2.8, 28.0], [5.4, 54.0]]], dtype=torch.float32)
        expected_scores = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        expected_labels = torch.tensor([[3, 1]], dtype=torch.long)

        self.assertTrue(torch.allclose(pooled_x, expected_x, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(pooled_scores, expected_scores, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(pooled_labels, expected_labels))

        tie_labels = torch.tensor([[4, 5]], dtype=torch.long)
        tie_scores = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        self.assertTrue(torch.equal(segment_pool_labels(tie_labels, tie_scores, scale=2), torch.tensor([[4]])))

    def test_prototype_center_interaction_splits_levels(self):
        branch = PrototypeCenterInteraction(
            dim_splits={"hf": 20, "mf": 10, "lf": 10},
            qk_dim_splits={"hf": 18, "mf": 9, "lf": 9},
            head_splits={"hf": 2, "mf": 1, "lf": 1},
            enabled_levels=("mf", "hf"),
        )
        prototypes = torch.randn(2, 6, 40)
        outputs = branch(prototypes)

        self.assertEqual(outputs["hf"]["k"].shape, (2, 2, 6, 9))
        self.assertEqual(outputs["hf"]["v"].shape, (2, 2, 6, 10))
        self.assertEqual(outputs["mf"]["k"].shape, (2, 1, 6, 9))
        self.assertEqual(outputs["mf"]["v"].shape, (2, 1, 6, 10))
        self.assertIsNone(outputs["lf"])

    def test_attention_state_alignment_resizes_and_renormalizes(self):
        prev_state = torch.rand(1, 3, 2, 4, 8, dtype=torch.float32)
        prev_state = prev_state / prev_state.sum(dim=-1, keepdim=True)

        aligned = align_attention_state(prev_state, target_shape=(1, 5, 1, 2, 4))

        self.assertEqual(aligned.shape, (1, 5, 1, 2, 4))
        self.assertTrue(torch.all(aligned >= 0))
        self.assertTrue(torch.allclose(aligned.sum(dim=-1), torch.ones_like(aligned.sum(dim=-1)), atol=1e-5, rtol=1e-5))

    def test_lmr_uses_explicit_cross_scale_projection(self):
        lmr = LowToHighMultiLevelReconstruction(
            dim=40,
            qk_dim=36,
            heads=4,
            group_size=16,
            block_index=0,
            total_blocks=8,
            level_head_split=(2, 1, 1),
        )

        self.assertEqual(lmr.mf_to_hf_proj.in_features, 10)
        self.assertEqual(lmr.mf_to_hf_proj.out_features, 20)
        self.assertEqual(lmr.lf_to_mf_proj.in_features, 10)
        self.assertEqual(lmr.lf_to_mf_proj.out_features, 10)

    def test_block_returns_three_level_attention_state(self):
        torch.manual_seed(0)
        self.assertIs(TAB, CAPABlock)
        block = CAPABlock(40, 36, 96, 4, num_tokens=8, group_size=16)
        x = torch.randn(1, 40, 8, 8)

        y, state = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(set(state.keys()), {"lf", "mf", "hf"})

        y_next, state_next = block(x, state)
        self.assertEqual(y_next.shape, x.shape)
        self.assertEqual(set(state_next.keys()), {"lf", "mf", "hf"})

    def test_capanet_supports_project_ablation_knobs(self):
        torch.manual_seed(0)
        model = CAPANet(
            upscale=2,
            n_iters=[1] * 8,
            num_tokens=[8] * 8,
            group_size=[16] * 8,
            level_head_split=[1, 1, 2],
            focus_mode="fixed",
            fixed_focus_ratio=0.25,
            use_sparse_pfsa=False,
            global_branch_levels=["hf", "mf", "lf"],
            attn_state_mode="shared",
            routing_mode="identity",
        )
        model.eval()
        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 3, 32, 32))

    def test_legacy_catanet_alias_still_builds(self):
        torch.manual_seed(0)
        model = CATANet(
            upscale=2,
            n_iters=[1] * 8,
            num_tokens=[8] * 8,
            group_size=[16] * 8,
        )
        model.eval()
        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
