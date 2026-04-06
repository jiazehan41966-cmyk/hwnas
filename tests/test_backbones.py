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


class BackboneBuildTests(unittest.TestCase):
    def test_default_candidates_are_declared(self) -> None:
        candidates = default_backbone_candidates()
        self.assertGreaterEqual(len(candidates), 6)
        self.assertEqual(candidates[0].arch_id, "simplecnn")
        self.assertIn("fbnet_like", {candidate.arch_id for candidate in candidates})

    def test_build_all_backbones_forward_single_channel(self) -> None:
        for backbone_name in (
            "simplecnn",
            "mobilenet_v2",
            "fbnet_like",
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


if __name__ == "__main__":
    unittest.main()
