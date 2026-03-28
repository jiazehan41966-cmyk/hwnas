import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import BackboneCostEstimator, get_board_profile
from hwnas_fpga.models import build_backbone


class BackboneCostTests(unittest.TestCase):
    def test_estimate_backbone_cost(self) -> None:
        model, _ = build_backbone(
            name="simplecnn",
            num_classes=4,
            input_channels=1,
            pretrained=False,
        )
        estimator = BackboneCostEstimator(
            hardware_spec=get_board_profile("zynq7020"),
            quantization_bits=8,
            cpu_latency_warmup=1,
            cpu_latency_runs=1,
        )
        estimate = estimator.estimate(
            model,
            input_shape=(1, 1, 64, 64),
            cpu_latency_warmup=1,
            cpu_latency_runs=1,
        )
        self.assertGreater(estimate.params, 0)
        self.assertGreater(estimate.macs, 0)
        self.assertGreater(estimate.cpu_latency_ms, 0.0)
        self.assertGreater(len(estimate.per_layer), 0)


if __name__ == "__main__":
    unittest.main()
