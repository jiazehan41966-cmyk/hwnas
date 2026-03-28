import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.models import build_backbone, default_backbone_candidates


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


if __name__ == "__main__":
    unittest.main()
