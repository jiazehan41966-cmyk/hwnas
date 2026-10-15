"""Source-bound new graph PTQ tests; all calibration images here are fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_elastic.model import StaticElasticNeckDetector
from tools.modnas_comparison.quantization import (
    TrainCalibration, model_state_hash, prepare_elastic_ptq,
    verify_fake_integer_parity, save_ptq_artifacts, ExactValuedH2FakeQuantConv,
)
from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv
from tools.fpga_h2 import numeric as integer_numeric


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def _train_fixture(root, count):
    images, labels = root / "train/images", root / "train/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rows = []
    for index in range(count):
        image = images / f"fixture_{index:03d}.png"
        annotation = labels / f"fixture_{index:03d}.txt"
        # Small stored files still decode through the actual 640 bilinear path.
        array = ((np.arange(16).reshape(4, 4) * 13 + index * 3) % 256).astype(np.uint8)
        Image.fromarray(array).save(image)
        annotation.write_text("0 0.5 0.5 0.02 0.02\n", encoding="utf-8")
        rows.append(dict(image=image.relative_to(root).as_posix(),
                         image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                         annotation=annotation.relative_to(root).as_posix(),
                         annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest()))
    return rows


def _arch(pan):
    return dict(pan=pan, td_kernels=[5, 3, 5], pan_kernels=[3, 5, 3] if pan else [])


def test_only_train64_formal_source_sha_and_augmentation_contract(tmp_path):
    rows = _train_fixture(tmp_path, 64)
    calibration = TrainCalibration(tmp_path, rows, batch_size=4)
    assert calibration.manifest["split"] == "train"
    assert calibration.manifest["number_of_calibration_images"] == 64
    assert not calibration.manifest["augmentation"] and not calibration.manifest["shuffle"]
    assert len(list(calibration.batches())) == 16
    bad = deepcopy(rows)
    bad[0]["image"] = "valid/images/fixture_000.png"
    # The split guard is evaluated before any valid/test file is accessed.
    with pytest.raises(ValueError, match="train-only"):
        TrainCalibration(tmp_path, bad)
    with pytest.raises(ValueError, match="exactly"):
        TrainCalibration(tmp_path, rows, num_images=8)
    with pytest.raises(ValueError, match="sorted"):
        TrainCalibration(tmp_path, list(reversed(rows)))
    (tmp_path / rows[0]["annotation"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 drift"):
        next(calibration.batches())


def test_real_formal_count64_folded_and_true_new_DAG_witness(tmp_path):
    rows = _train_fixture(tmp_path, 64)
    calibration = TrainCalibration(tmp_path, rows, batch_size=1)
    model = StaticElasticNeckDetector(_arch(False)).eval()
    before = model_state_hash(model)
    prepared = prepare_elastic_ptq(model, calibration)
    assert model_state_hash(model) == before
    assert not any(isinstance(module, nn.BatchNorm2d) for module in prepared.folded_model.modules())
    assert next(prepared.folded_model.parameters()).dtype == torch.float64
    assert next(model.parameters()).dtype == torch.float32
    witness = prepared.profile["bn_folding"]["witness"]
    assert witness["true_new_graph_prediction_max_errors"] == [0.0] * 4
    assert max(witness["bn_folded_prediction_max_errors"]) < 1e-8
    assert witness["mathematical_fp64"]["all_heads_allclose"]
    assert witness["tighter_mathematical_fp64"]["all_heads_allclose"]
    assert witness["bn_fold_atol"] == witness["bn_fold_rtol"] == 1e-5
    assert witness["observed_source_fp32_vs_folded_fp32"]["atol"] == 1e-5
    assert witness["fp32_observation_is_acceptance_gate"] is False
    assert prepared.profile["source_checkpoint_floating_dtypes"] == ["torch.float32"]
    assert prepared.profile["folded_reference_floating_dtypes"] == ["torch.float64"]
    assert prepared.profile["bn_folding"]["original_fp32_tolerance_relaxed"] is False
    for row in witness["observed_source_fp32_vs_folded_fp32"]["heads"]:
        assert row["max_abs_error_in_output_lsb"] >= 0
        assert row["train64_calibrated_output_lsb"] > 0
    assert prepared.profile["calibration"]["number_of_calibration_images"] == 64
    assert prepared.profile["integer_vs_fake_parity"] == "NOT_RUN"
    assert prepared.profile["integer_vs_HLS_parity"] == prepared.profile["board_parity"] == "NOT_RUN"
    assert prepared.profile["hardware_feasibility"] == "UNKNOWN"
    assert prepared.profile["formal_eligible"] is False
    assert len(prepared.profile["adds"]) == 6  # Three TD fusions plus their MBConv residuals.
    assert not any("bottom_up" in row["name"] for row in prepared.profile["layers"])
    assert all("output_convs" not in row["name"] for row in prepared.profile["layers"])


@pytest.mark.parametrize("pan", [False, True])
def test_true_640_integer_runtime_all_nodes_fake_witness_and_PAN_sources(pan, tmp_path):
    rows = _train_fixture(tmp_path, 1)
    calibration = TrainCalibration(tmp_path, rows, num_images=1, formal=False)
    model = StaticElasticNeckDetector(_arch(pan), seed=43).eval()
    prepared = prepare_elastic_ptq(model, calibration)
    images = next(calibration.batches())
    witness = verify_fake_integer_parity(prepared, images)
    assert witness["mismatches"] == 0
    assert witness["status"] == "PASS_SOFTWARE_INTEGER_FAKE_PARITY"
    assert witness["formal_eligible"] is False
    integer = prepared.integer_reference
    result = integer.forward(integer.quantize_input(images), retain=True)
    assert all(value.dtype == np.int8 for value in result.values.values())
    assert len(witness["rows"]) == len(integer.graph["nodes"]) + 1
    assert len(prepared.profile["adds"]) == (12 if pan else 6)
    assert all("left_max_abs" in row["observation"] and "right_max_abs" in row["observation"]
               for row in prepared.profile["adds"])
    predictions = result.predictions["predictions"]
    assert [list(prediction.shape) for prediction in predictions] == [
        [1, 21, 160, 160], [1, 21, 80, 80], [1, 21, 40, 40], [1, 21, 20, 20]]
    graph = {node["id"]: node for node in integer.graph["nodes"]}
    # Top-down must read the previous processed MBConv residual, not a lateral.
    assert graph["neck.top_down.P3.nearest"]["inputs"] == ["neck.top_down_blocks.P4.residual_add"]
    assert graph["neck.top_down.P2.nearest"]["inputs"] == ["neck.top_down_blocks.P3.residual_add"]
    if pan:
        assert graph["neck.bottom_up.P4.maxpool"]["inputs"] == ["neck.bottom_up_blocks.P3.residual_add"]
        assert graph["neck.bottom_up.P5.maxpool"]["inputs"] == ["neck.bottom_up_blocks.P4.residual_add"]
        assert len([node for node in graph.values() if node["op"] == "maxpool"]) == 3
    else:
        assert not any("bottom_up" in name for name in graph)
    with pytest.raises(ValueError, match="already quantized"):
        integer.forward(images)
    artifact_dir = save_ptq_artifacts(tmp_path / "prepared", prepared, witness=witness)
    package = torch.load(artifact_dir / "integer_parameters.pt", weights_only=True)
    assert all(tensor.dtype == torch.int8 for tensor in package["weights"].values())
    assert all(tensor.dtype == torch.int32 for tensor in package["biases"].values())
    for filename, expected in json.loads((artifact_dir / "artifact_manifest.json").read_text())["files"].items():
        assert hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        save_ptq_artifacts(artifact_dir, prepared)
    first_parameter = next(iter(integer.parameters.values()))
    first_parameter["weight"].flat[0] += np.int8(1)
    with pytest.raises(ValueError, match="parameter drift"):
        verify_fake_integer_parity(prepared, images)


def test_unsupported_INT16_or_raw_calibration_fail_closed(tmp_path):
    rows = _train_fixture(tmp_path, 1)
    calibration = TrainCalibration(tmp_path, rows, num_images=1, formal=False)
    model = StaticElasticNeckDetector(_arch(False), num_classes=3).eval()
    for profile in ("HEAD_W8A16", "HEAD_W16A16"):
        with pytest.raises(NotImplementedError, match="INT16"):
            prepare_elastic_ptq(model, calibration, profile=profile)
    with pytest.raises(ValueError, match="source-bound"):
        prepare_elastic_ptq(model, [torch.zeros(1, 1, 640, 640)])
    calibration.manifest["augmentation"] = True
    with pytest.raises(ValueError, match="manifest drift"):
        prepare_elastic_ptq(model, calibration)


def test_add_and_accumulator_extremes_follow_retained_signed_H2_contract():
    # Different source scales plus clipping and tie-to-even are meaningful
    # corner cases for every new external fusion and MBConv residual add.
    a = np.array([-128, -3, -1, 1, 3, 127], dtype=np.int8)
    b = np.array([127, 0, 0, 0, 0, 127], dtype=np.int8)
    result = integer_numeric.aligned_add(a, b, -2, -1, -1)
    np.testing.assert_array_equal(result, np.array([63, -2, 0, 0, 2, 127], dtype=np.int8))
    np.testing.assert_array_equal(integer_numeric.wrap32(np.array([2**31, 2**32 - 1, -2**31 - 1], dtype=np.int64)),
                                  np.array([-2**31, -1, 2**31 - 1], dtype=np.int64))


def test_legacy_FP32_fake_failure_and_explicit_INT32_overflow_correction():
    conv = nn.Conv2d(1, 1, 1, bias=True)
    with torch.no_grad():
        conv.weight.fill_(64.0)
        conv.bias.fill_(2**31)
    kwargs = dict(input_exp=0, output_exp=0, weight_exps=[0], activation="linear",
                  weight_granularity="per_tensor")
    exact = ExactValuedH2FakeQuantConv(conv, **kwargs)
    old = H2FakeQuantConv(conv, **kwargs)
    image = torch.ones(1, 1, 1, 1)
    expected = integer_numeric.conv(np.ones((1, 1, 1), dtype=np.int8),
        exact.int_weight.numpy(), exact.int_bias.numpy(),
        dict(stride=1, padding=0, groups=1, multiplier=1, shift=0, activation="linear", output_exp=0))
    assert int(expected[0, 0, 0]) == -128
    assert exact(image).item() == -128
    assert old(image).item() == 127  # This old floating simulator has no INT32 wrap.
