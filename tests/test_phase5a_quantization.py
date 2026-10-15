"""Phase 5A static and small-tensor tests. No test-split access."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1, require_h2_aligned
from hwnas_fpga.deploy.h2_ptq import (
    QMAX,
    QMIN,
    prepare_h2_ptq,
    quantize_h2,
    ratio_from_exponents,
)


def load_common():
    spec = importlib.util.spec_from_file_location(
        "phase5a_common", ROOT / "tools/phase5a_quantization/common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h2_contract_hash_is_pinned():
    payload = load_numeric_contract_v1()
    assert payload["range"] == [-128, 127]
    assert payload["rounding"] == "nearest_even"
    assert payload["requant_product_bits"] == 64
    assert payload["zero_point"] == 0
    assert EXPECTED_HASH == "001f461391eeaebd05daa984ff0703f76cc3ddd71cd347b203bf3e4c1ba7d145"


def test_asymmetric_fpga_policy_rejected():
    with pytest.raises(ValueError):
        require_h2_aligned(
            {
                "zero_point": 1,
                "signed_range": [-128, 127],
                "rounding": "nearest_even",
                "activation_bits": 8,
                "weight_bits": 8,
                "accumulator_bits": 32,
                "requant_product_bits": 64,
                "scale": "positive_power_of_two_exponent_int32",
            }
        )


def test_old_symmetric_range_rejected():
    with pytest.raises(ValueError):
        require_h2_aligned(
            {
                "zero_point": 0,
                "signed_range": [-127, 127],
                "rounding": "nearest_even",
                "activation_bits": 8,
                "weight_bits": 8,
                "accumulator_bits": 32,
                "requant_product_bits": 64,
                "scale": "positive_power_of_two_exponent_int32",
            }
        )


def test_only_xstar_admitted():
    common = load_common()
    assert list(common.NETWORKS) == ["B1", "C", "D"]
    assert common.admit_network("B1").startswith("P2_P3_P4_P5")
    with pytest.raises(ValueError):
        common.admit_network("B2")
    with pytest.raises(ValueError):
        common.admit_network("A1")


def test_test_split_is_sealed():
    common = load_common()
    with pytest.raises(ValueError, match="test"):
        common.forbid_split("test")
    common.forbid_split("train")
    common.forbid_split("valid")


def test_requant_multiplier_shift_matches_h2():
    assert ratio_from_exponents(-4, -5) == (2, 0)
    assert ratio_from_exponents(-5, -4) == (1, 1)
    quantized = quantize_h2(torch.tensor([1.0, -1.0, 0.0]), -7)
    assert float(quantized.min()) >= QMIN
    assert float(quantized.max()) <= QMAX


def test_qat_thresholds_are_preregistered():
    common = load_common()
    assert common.QAT_ABSOLUTE_DROP == 0.003
    assert common.QAT_RELATIVE_DROP == 0.20


def test_candidate_d_ptq_wrap_covers_h2_ops():
    from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
    from phase5a_quantization.common import parse_architecture_id
    from phase5a_quantization.export_int8 import operator_coverage

    architecture = parse_architecture_id("P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3")
    detector = build_sonar_candidate(SPEC_PATH, architecture).eval()
    dummy = [torch.randn(1, 1, 640, 640)]
    folded, quantized, _extrema, profile = prepare_h2_ptq(detector, dummy, device="cpu")
    assert profile["h2_aligned"] is True
    assert profile["qat"] is False
    assert profile["bitwidth"]["activation"] == 8
    assert not any(isinstance(module, nn.BatchNorm2d) for module in folded.modules())
    with torch.no_grad():
        fp32 = folded(dummy[0])["predictions"]
        int8 = quantized(dummy[0])["predictions"]
    assert [tuple(item.shape) for item in fp32] == [tuple(item.shape) for item in int8]
    assert any(layer["multiplier"] is not None for layer in profile["layers"])
    coverage = operator_coverage(quantized)
    assert coverage["all_operators_h2_covered"]
    assert "aligned_add" in coverage["operator_counts"]


def test_run_source_keeps_scope_closed():
    source = (ROOT / "tools/phase5a_quantization/run.py").read_text(encoding="utf-8")
    mapping = (ROOT / "tools/phase5a_quantization/common.py").read_text(encoding="utf-8")
    block = mapping.split("NETWORKS = {", 1)[1].split("}", 1)[0]
    assert "B1" in block and "C" in block and "D" in block
    assert "A1" not in block and "B2" not in block
    assert "quantize_all_144" in source
    assert "network_selection_from_ptq" in source
    assert "test_evaluated" in source
