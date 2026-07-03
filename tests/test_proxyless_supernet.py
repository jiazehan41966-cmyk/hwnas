import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.models import ProxylessSuperNet
from hwnas_fpga.search import ProxylessSearcher, create_searcher
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


def _build_mobile_anchor_space() -> SearchSpace:
    constraints = SearchConstraints(
        max_latency_ms=50.0,
        max_dsp=840,
        max_bram=445,
        max_lut=50_950,
        max_memory_bandwidth_gbps=3.0,
        max_offchip_mem_mb=128.0,
    )
    config = SearchSpaceConfig.from_dict(
        {
            "family_profile": "mobile_anchor",
            "input_channels": 1,
            "image_size": 224,
            "stem_channels": 24,
            "stem_stride": 2,
            "post_stem_downsample_stride": 2,
            "stage_strides": [2, 2, 2],
            "stage_base_channels": [24, 48, 96],
            "width_multipliers": [0.5, 0.75, 1.0],
            "stage_depth_choices": [[1, 2, 3], [2, 3, 4], [1, 2, 3]],
            "kernel_choices": [3, 5],
            "expand_choices": [1, 2, 4, 6],
            "op_choices": ["dw_pw_conv", "mbconv", "fused_mbconv", "skip"],
            "head_conv_channels": 1024,
            "num_classes": 8,
            "hardware_constraints": constraints,
        }
    )
    return SearchSpace(config)


def _build_estimator(constraints: SearchConstraints) -> FPGACostEstimator:
    return FPGACostEstimator(
        hardware_spec=HardwareSpec(
            name="alinx_av7k325",
            clock_mhz=200,
            max_lut=50_950,
            max_bram=445,
            max_dsp=840,
            max_power_w=12.0,
            memory_bandwidth_gbps=3.0,
            offchip_mem_mb=128.0,
        ),
        constraints=constraints,
    )


class ProxylessSuperNetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = _build_mobile_anchor_space()
        self.estimator = _build_estimator(self.space.config.hardware_constraints)

    def test_supernet_honors_stage_specific_mobile_anchor_choices(self) -> None:
        model = ProxylessSuperNet(
            search_space=self.space,
            num_classes=8,
            cost_estimator=self.estimator,
        )
        self.assertEqual(model.stage_channel_choices, ((12, 18, 24), (24, 36, 48), (48, 72, 96)))
        self.assertEqual(model.stage_depth_choices, ((1, 2, 3), (2, 3, 4), (1, 2, 3)))

        model.sample_single_paths(__import__("random").Random(7))
        with torch.no_grad():
            logits = model(torch.randn(2, 1, 224, 224))
        self.assertEqual(tuple(logits.shape), (2, 8))

        architecture = model.extract_architecture()
        self.assertEqual(architecture.post_stem_downsample_stride, 2)
        self.assertEqual(architecture.head_conv_channels, 1024)
        self.assertTrue(self.space.is_valid(architecture))

    def test_expected_hardware_metrics_scale_with_larger_stage_choices(self) -> None:
        model = ProxylessSuperNet(
            search_space=self.space,
            num_classes=8,
            cost_estimator=self.estimator,
        )

        with torch.no_grad():
            for stage in model.stages:
                for mixed in stage._mixed_ops:
                    mixed.alpha.fill_(-10.0)
                    mixed.alpha[0] = 10.0
                stage.channel_gate.alpha.fill_(-10.0)
                stage.channel_gate.alpha[0] = 10.0
                stage.depth_gate.alpha.fill_(-10.0)
                stage.depth_gate.alpha[0] = 10.0
        small = model.expected_hardware_metrics()

        with torch.no_grad():
            for stage in model.stages:
                stage.channel_gate.alpha.fill_(-10.0)
                stage.channel_gate.alpha[-1] = 10.0
                stage.depth_gate.alpha.fill_(-10.0)
                stage.depth_gate.alpha[-1] = 10.0
        large = model.expected_hardware_metrics()

        self.assertGreater(float(large["latency_ms"]), float(small["latency_ms"]))
        self.assertGreater(float(large["dsp"]), float(small["dsp"]))
        self.assertGreater(float(large["lut"]), float(small["lut"]))

    def test_create_proxyless_searcher_uses_same_stage_specific_space(self) -> None:
        searcher = create_searcher(
            search_space=self.space,
            cost_estimator=self.estimator,
            constraints=self.space.config.hardware_constraints,
            method="proxyless",
            seed=7,
            device="cpu",
            selection_metric="macro_f1",
            proxyless_cfg={
                "warmup_epochs": 1,
                "search_epochs": 1,
                "max_eval_batches": 1,
            },
        )
        self.assertIsInstance(searcher, ProxylessSearcher)
        self.assertEqual(searcher.selection_metric, "macro_f1")
        self.assertEqual(
            searcher.supernet.stage_channel_choices,
            ((12, 18, 24), (24, 36, 48), (48, 72, 96)),
        )
        self.assertEqual(
            searcher.supernet.stage_depth_choices,
            ((1, 2, 3), (2, 3, 4), (1, 2, 3)),
        )


if __name__ == "__main__":
    unittest.main()
