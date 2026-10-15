"""Mixed-head software tests with fixture images; no accuracy/hardware claims."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH
from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv
from tools.fpga_h2 import numeric as body_numeric
from tools.modnas_elastic.cost_graph import canonical_hash
from tools.modnas_elastic.model import StaticElasticNeckDetector
from tools.modnas_comparison import quantization as body_adapter
from tools.modnas_comparison import quantization_mixed as mixed


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


def _fixture(root):
    image, label = root / "train/images/one.png", root / "train/labels/one.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.fromarray((np.arange(64).reshape(8, 8) * 3).astype(np.uint8)).save(image)
    label.write_text("0 0.5 0.5 0.05 0.05\n", encoding="utf-8")
    entries = [dict(image="train/images/one.png", annotation="train/labels/one.txt",
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        annotation_sha256=hashlib.sha256(label.read_bytes()).hexdigest())]
    return mixed.TrainCalibration(root, entries, formal=False, num_images=1)


@pytest.mark.parametrize("pan", [False, True])
def test_all_three_profiles_true_640_dtype_boundaries_contracts_and_all_node_parity(pan, tmp_path):
    calibration = _fixture(tmp_path)
    architecture = dict(pan=pan, td_kernels=[5, 3, 5], pan_kernels=[3, 5, 3] if pan else [])
    model = StaticElasticNeckDetector(architecture, seed=43).eval()
    original = mixed.model_state_hash(model)
    prepared = mixed.prepare_elastic_profiles(model, calibration)
    assert tuple(prepared) == mixed.SUPPORTED_PROFILES
    assert mixed.model_state_hash(model) == original
    assert prepared["W8A8"].profile["adapter_version"] == body_adapter.ADAPTER_VERSION
    assert prepared["W8A8"].profile["numeric_contract_hash"] == EXPECTED_HASH
    images = next(calibration.batches())
    hashes = []
    for profile, candidate in prepared.items():
        witness = mixed.verify_fake_integer_parity(candidate, images)
        assert witness["mismatches"] == 0
        assert witness["integer_vs_HLS_parity"] == witness["board_parity"] == "NOT_RUN"
        integer = candidate.integer_reference
        result = integer.forward(integer.quantize_input(images), retain=True)
        if profile == "W8A8":
            assert all(value.dtype == np.int8 for value in result.values.values())
            continue
        hashes.append(candidate.profile["numeric_contract_hash"])
        assert candidate.profile["numeric_contract_regions"]["body"]["numeric_contract_hash"] == EXPECTED_HASH
        assert candidate.profile["hardware_boundary_nodes_measured"] is False
        nodes = {node["id"]: node for node in integer.graph["nodes"]}
        boundaries = [node for node in nodes.values() if node["op"] == "requantize"]
        assert len(boundaries) == 4
        head_weight_bits = 8 if profile == "HEAD_W8A16" else 16
        for path, value in result.values.items():
            assert value.dtype == (np.int16 if path.startswith("head.") else np.int8)
        for node in boundaries:
            assert node["input_bits"] == 8 and node["output_bits"] == 16
            assert not node["inputs"][0].startswith("head.")
            assert node["hardware_measurement"] == "MISSING"
        package = integer.parameter_package()
        for path, weight in package["weights"].items():
            expected = torch.int16 if path.startswith("head.") and head_weight_bits == 16 else torch.int8
            assert weight.dtype == expected
            assert package["biases"][path].dtype == torch.int32
            if not path.startswith("head."):
                baseline = prepared["W8A8"].integer_reference.parameters[path]
                np.testing.assert_array_equal(weight.numpy(), baseline["weight"])
                np.testing.assert_array_equal(package["biases"][path].numpy(), baseline["bias"])
        for level, path in integer.graph["outputs"].items():
            head = nodes[path]
            assert head["input_bits"] == head["output_bits"] == 16
            assert head["weight_bits"] == head_weight_bits
            assert nodes[head["inputs"][0]]["op"] == "requantize"
        assert len(witness["rows"]) == len(integer.graph["nodes"]) + 1
        changed = deepcopy(integer.graph)
        integer.graph["nodes"][-1]["output_exp"] += 1
        with pytest.raises(ValueError, match="graph drift"):
            mixed.verify_fake_integer_parity(candidate, images)
        integer.graph = changed
    assert len(set(hashes)) == 2 and EXPECTED_HASH not in hashes
    assert canonical_hash(body_numeric.CONTRACT) == EXPECTED_HASH


def test_mixed_signed16_requant_ties_saturation_and_real_INT32_wrap():
    values = np.array([-100000, -3, -1, 1, 3, 100000], dtype=np.int64)
    np.testing.assert_array_equal(mixed.requantize_int32(values, 1, 1, bits=16),
                                  np.array([-32768, -2, 0, 0, 2, 32767], dtype=np.int16))
    body_edge = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    np.testing.assert_array_equal(mixed.requantize_int32(body_edge, 256, 0, bits=16),
                                  np.array([-32768, -256, 0, 256, 32512], dtype=np.int16))
    with pytest.raises(ValueError, match="wrapped INT32"):
        mixed.requantize_int32(np.array([2**31], dtype=np.int64), 1, 0, bits=16)
    conv = nn.Conv2d(1, 1, 1, bias=True).double()
    with torch.no_grad():
        conv.weight.fill_(32767)
        conv.bias.fill_(2**31 - 1)
    kwargs = dict(input_exp=0, output_exp=0, weight_exps=[0], activation="linear",
                  weight_granularity="per_tensor", input_bits=16, weight_bits=16, output_bits=16)
    fake = mixed.ExactValuedMixedHeadConv(conv, **kwargs)
    old_qdq = H2FakeQuantConv(conv, **kwargs)
    image = torch.full((1, 1, 1, 1), 32767.0, dtype=torch.float64)
    accumulator = body_numeric.conv_acc(np.full((1, 1, 1), 32767, dtype=np.int16),
                                        fake.int_weight.numpy(), fake.int_bias.numpy())
    expected = mixed.requantize_int32(accumulator, 1, 0, bits=16)
    assert expected.item() == -32768
    assert fake(image).item() == -32768
    assert old_qdq(image.float()).item() == 32767  # Old FP32 mixed QDQ omits INT32 wrap.


def test_unknown_profile_is_rejected_before_calibration_read():
    with pytest.raises(NotImplementedError, match="unsupported software"):
        mixed.prepare_elastic_ptq(None, None, profile="HEAD_W16A8")
