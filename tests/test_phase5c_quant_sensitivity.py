"""Phase 5C quantization-sensitivity bypass tests. Test split remains sealed."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv, H2QuantizedFPN, prepare_h2_ptq
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from phase5a_quantization.common import parse_architecture_id


B1_ARCH = "P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
D_ARCH = "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3"


def _build(arch_id: str) -> nn.Module:
    architecture = parse_architecture_id(arch_id)
    return build_sonar_candidate(SPEC_PATH, architecture).eval()


def _calibration_images():
    return [torch.randn(1, 1, 640, 640)]


def test_fp32_head_bypass_preserves_head_convs_and_wraps_backbone():
    detector = _build(D_ARCH)
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        detector,
        _calibration_images(),
        device="cpu",
        fp32_module_prefixes=("head",),
    )

    head_convs = [(name, module) for name, module in quantized.named_modules() if name.startswith("head") and isinstance(module, nn.Conv2d)]
    assert head_convs, "head FP32 bypass should leave raw Conv2d modules under head"
    assert not any(isinstance(module, H2FakeQuantConv) for name, module in quantized.named_modules() if name.startswith("head"))
    assert any(isinstance(module, H2FakeQuantConv) for name, module in quantized.named_modules() if name.startswith("backbone"))
    assert profile["fp32_bypass"]["module_prefixes"] == ["head"]
    assert all(layer.startswith("head.") for layer in profile["fp32_bypass"]["conv_layers"])
    assert profile["fp32_bypass"]["neck_adds_quantized"] is True
    assert profile["fp32_bypass"]["software_diagnostic_only"] is True


def test_neck_fp32_bypass_disables_quantized_fpn_adds():
    detector = _build(D_ARCH)
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        detector,
        _calibration_images(),
        device="cpu",
        fp32_module_prefixes=("neck",),
        quantize_neck_adds=False,
    )

    assert not isinstance(quantized.neck, H2QuantizedFPN)
    assert any(isinstance(module, nn.Conv2d) for name, module in quantized.named_modules() if name.startswith("neck"))
    assert not any(isinstance(module, H2FakeQuantConv) for name, module in quantized.named_modules() if name.startswith("neck"))
    assert any(isinstance(module, H2FakeQuantConv) for name, module in quantized.named_modules() if name.startswith("head"))
    assert profile["fp32_bypass"]["module_prefixes"] == ["neck"]
    assert profile["fp32_bypass"]["neck_adds_quantized"] is False


def test_b1_p2_path_bypass_is_limited_to_level_zero_modules():
    detector = _build(B1_ARCH)
    prefixes = ("neck.lateral_convs.0", "neck.output_convs.0", "head.towers.0")
    _folded, quantized, _extrema, profile = prepare_h2_ptq(
        detector,
        _calibration_images(),
        device="cpu",
        fp32_module_prefixes=prefixes,
    )

    bypassed = set(profile["fp32_bypass"]["conv_layers"])
    assert bypassed
    assert all(any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes) for name in bypassed)
    assert any(name.startswith("head.towers.0") for name in bypassed)
    assert not any(name.startswith("head.towers.1") for name in bypassed)
    assert any(isinstance(module, H2FakeQuantConv) for name, module in quantized.named_modules() if name.startswith("head.towers.1"))
