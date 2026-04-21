import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicsr.archs.catanet_arch import CAPANet, DynamicPrototypeRouter, TAB  # noqa: E402


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

        restored = torch.zeros_like(sorted_x).scatter(1, idx_last.expand_as(sorted_x), sorted_x)
        self.assertEqual(restored.shape, x.shape)

    def test_block_returns_three_level_attention_state(self):
        torch.manual_seed(0)
        block = TAB(40, 36, 96, 4, num_tokens=8, group_size=16)
        x = torch.randn(1, 40, 8, 8)

        y, state = block(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(set(state.keys()), {"lf", "mf", "hf"})

        y_next, state_next = block(x, state)
        self.assertEqual(y_next.shape, x.shape)
        self.assertEqual(set(state_next.keys()), {"lf", "mf", "hf"})

    def test_capanet_x2_output_shape(self):
        torch.manual_seed(0)
        model = CAPANet(
            upscale=2,
            n_iters=[1] * 8,
            num_tokens=[8] * 8,
            group_size=[16] * 8,
        )
        model.eval()
        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
