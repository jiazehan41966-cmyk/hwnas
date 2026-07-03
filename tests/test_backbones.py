import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.models import (
    build_backbone,
    default_backbone_candidates,
    get_macro_template,
    list_macro_templates,
)
from hwnas_fpga.models.backbones import _adapt_conv_weight


class BackboneBuildTests(unittest.TestCase):
    def test_default_candidates_are_declared(self) -> None:
        candidates = default_backbone_candidates()
        self.assertGreaterEqual(len(candidates), 6)
        self.assertEqual(candidates[0].arch_id, "simplecnn")
        self.assertIn("fbnet_a", {candidate.arch_id for candidate in candidates})

    def test_build_all_backbones_forward_single_channel(self) -> None:
        for backbone_name in (
            "simplecnn",
            "mobilenet_v2",
            "fbnet_a",
            "shufflenet_v2",
            "efficientnet_b0",
        ):
            with self.subTest(backbone=backbone_name):
                model, metadata = build_backbone(
                    name=backbone_name,
                    num_classes=4,
                    input_channels=1,
                    pretrained=False,
                )
                outputs = model(torch.randn(2, 1, 64, 64))
                self.assertEqual(outputs.shape, (2, 4))
                self.assertFalse(metadata["pretrained_loaded"])

    def test_macro_templates_are_available(self) -> None:
        templates = list_macro_templates()
        self.assertIn("mobilenet_v2_like", templates)
        self.assertIn("fbnet_like", templates)

        fbnet_template = get_macro_template("fbnet_like")
        self.assertEqual(fbnet_template["display_name"], "FBNet-like")
        self.assertEqual(len(fbnet_template["stages"]), 4)

        mobilenet_template = get_macro_template("mobile_anchor")
        self.assertEqual(mobilenet_template["name"], "mobilenet_v2_like")
        self.assertEqual(len(mobilenet_template["stages"]), 4)

    def test_adapt_conv_weight_three_to_two_repeats_channel_average(self) -> None:
        weight = torch.tensor(
            [
                [[[1.0]], [[3.0]], [[5.0]]],
                [[[2.0]], [[4.0]], [[6.0]]],
            ]
        )
        adapted = _adapt_conv_weight(weight, input_channels=2)
        expected = weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        self.assertEqual(tuple(adapted.shape), (2, 2, 1, 1))
        self.assertTrue(torch.allclose(adapted, expected))


if __name__ == "__main__":
    unittest.main()
