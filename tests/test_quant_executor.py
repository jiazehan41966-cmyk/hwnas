"""Small source/number tests; no training, dataset PTQ run, HLS or board job."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from quant import Q
from quant.executor import TrainCalibration, prepare_ptq, prepare_qat, save_ptq_artifacts
from quant.executor import model_state_hash, verify_fake_integer_parity
from quant.executor import verify_saved_artifacts
from quant import executor
from quant.runtime import numeric
from quant.runtime.cost_graph import build_cost_graph, canonical_hash
from quant.runtime.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from quant.runtime.quantization import ExactValuedH2FakeQuantConv
from quant.runtime.quantization_mixed import ExactValuedMixedHeadConv, _contracts, requantize_int32
from space import Genotype


def _train_fixture(tmp_path):
    image = tmp_path / "train/images/one.png"
    label = tmp_path / "train/labels/one.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("L", (2, 2), 127).save(image)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    entry = dict(image="train/images/one.png", annotation="train/labels/one.txt",
                 image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                 annotation_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
    return entry, image


def test_train_calibration_rejects_validation_and_changed_source(tmp_path):
    entry, image = _train_fixture(tmp_path)
    invalid = dict(entry, image="valid/images/one.png")
    with pytest.raises(ValueError, match="train-only"):
        TrainCalibration(tmp_path, [invalid], num_images=1, formal=False)
    calibration = TrainCalibration(tmp_path, [entry], num_images=1, formal=False)
    assert calibration.manifest["split"] == "train"
    assert calibration.manifest["test_accessed"] is False
    with pytest.raises(ValueError, match="exactly"):
        TrainCalibration(tmp_path, [entry], num_images=1)
    image.write_bytes(b"source drift")
    with pytest.raises(ValueError, match="SHA256 drift"):
        next(calibration.batches())


def test_contract_hashes_and_signed_rounding_boundaries():
    assert canonical_hash(load_numeric_contract_v1()) == EXPECTED_HASH
    assert canonical_hash(numeric.CONTRACT) == EXPECTED_HASH
    contracts = [_contracts(p) for p in ("HEAD_W8A16", "HEAD_W16A16")]
    assert contracts[0]["candidate"] != contracts[1]["candidate"]
    values = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9], dtype=np.int32)
    expected = np.array([-4, -4, -2, -2, 0, 0, 2, 2, 4, 4])
    assert np.array_equal(numeric.round_shift(values, 1), expected)
    extremes = np.array([-2**31, -32769, -129, 128, 32768, 2**31-1], dtype=np.int64)
    assert requantize_int32(extremes, 1, 0, bits=8).tolist() == [-128, -128, -128, 127, 127, 127]
    assert requantize_int32(extremes, 1, 0, bits=16).tolist() == [-32768, -32768, -129, 128, 32767, 32767]
    with pytest.raises(ValueError, match="INT32"):
        requantize_int32(np.array([2**31]), 1, 0, bits=16)


@pytest.mark.parametrize("input_bits,weight_bits,output_bits", [(8, 8, 8), (16, 8, 16), (16, 16, 16)])
def test_micro_convolution_fake_integer_parity_including_int32_overflow(input_bits, weight_bits, output_bits):
    maximum = 2**(weight_bits-1)-1
    conv = nn.Conv2d(1, 1, 1, bias=True).eval()
    with torch.no_grad():
        conv.weight.fill_(maximum)
        conv.bias.zero_()
    cls = ExactValuedH2FakeQuantConv if output_bits == 8 else ExactValuedMixedHeadConv
    fake = cls(conv, input_exp=0, output_exp=0, weight_exps=[0], activation="linear",
               weight_granularity="per_tensor", input_bits=input_bits,
               weight_bits=weight_bits, output_bits=output_bits)
    fake.int_bias.fill_(2**31-4)
    limit = 2**(input_bits-1)
    inputs = np.array([limit-1, -limit, 1, -1], dtype=np.int16 if input_bits == 16 else np.int8).reshape(1, 2, 2)
    weights = np.array([maximum], dtype=np.int16 if weight_bits == 16 else np.int8).reshape(1, 1, 1, 1)
    accumulator = numeric.conv_acc(inputs, weights, np.array([2**31-4], dtype=np.int32))
    assert np.any(accumulator < 0)  # actual modulo overflow, not a safe-range-only fixture
    expected = requantize_int32(accumulator, 1, 0, bits=output_bits)
    actual = fake(torch.from_numpy(inputs.copy()).float().unsqueeze(0)).detach().numpy()[0]
    assert np.array_equal(actual, expected)


def test_graph_uses_external_model_and_preserves_full_pan():
    genotype = Genotype(True, (3, 5, 3), (5, 3, 5))
    model = SimpleNamespace(candidate_architecture=genotype.legacy_architecture(), num_classes=2)
    graph = build_cost_graph(model)
    assert tuple(graph["outputs"]) == ("P2", "P3", "P4", "P5")
    assert sum(node["op"] == "maxpool" for node in graph["nodes"]) == 3
    assert sum(node["op"] == "nearest" for node in graph["nodes"]) == 3
    assert sum(node["op"] == "residual_add" for node in graph["nodes"]) == 6


def test_executor_binds_plan_and_rejects_identity_drift(tmp_path, monkeypatch):
    genotype = Genotype(False, (3, 3, 3))
    model = nn.Module().eval()
    model.candidate_architecture = genotype.legacy_architecture()
    model.num_classes = 2
    entry, _ = _train_fixture(tmp_path)
    calibration = TrainCalibration(tmp_path, [entry], num_images=1, formal=False)
    with pytest.raises(ValueError, match="different model architecture"):
        prepare_ptq(model, calibration, plan=Q(Genotype(False, (5, 3, 3))))
    with pytest.raises(TypeError, match="source-bound"):
        prepare_ptq(model, [], plan=Q(genotype))
    graph = build_cost_graph(model)
    graph["numeric_graph_sha256"] = canonical_hash(graph)
    prepared = SimpleNamespace(profile=dict(precision_profile="W8A8", numeric_graph_sha256=graph["numeric_graph_sha256"]),
                               fake_model=SimpleNamespace(graph=copy.deepcopy(graph)),
                               integer_reference=SimpleNamespace(graph=copy.deepcopy(graph)))
    monkeypatch.setattr(executor._runtime, "prepare_elastic_ptq", lambda *args, **kwargs: prepared)
    result = prepare_ptq(model, calibration, plan=Q(genotype))
    assert result.profile["arch_hash"] == genotype.arch_hash
    assert result.profile["q_hash"] == Q(genotype).q_hash
    result.fake_model.graph["nodes"][0]["stride"] = 1
    with pytest.raises(ValueError, match="numeric graph drift"):
        save_ptq_artifacts(tmp_path / "ptq", result)
    with pytest.raises(NotImplementedError, match="no QAT"):
        prepare_qat()


def test_migrated_runtime_has_no_legacy_imports_and_manifest_hashes():
    runtime = Path(executor._runtime.__file__).parent
    manifest = json.loads((runtime / "MIGRATION.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest, name
    for path in runtime.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from hwnas_fpga" not in source
        assert "from tools." not in source
        assert "sys.path" not in source


def test_real_e1_mixed_ptq_artifact_interface_on_synthetic_train_fixture(tmp_path):
    """One synthetic image exercises integration, never accuracy or formal PTQ."""
    from models import build_model

    previous_threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        genotype = Genotype(True, (3, 3, 3), (3, 3, 3))
        model = build_model(genotype, seed=42).eval()
        original_state = model_state_hash(model)
        entry, _ = _train_fixture(tmp_path)
        calibration = TrainCalibration(tmp_path, [entry], num_images=1, formal=False)
        plan = Q(genotype, "HEAD_W8A16")
        prepared = prepare_ptq(model, calibration, plan=plan)
        assert model_state_hash(model) == original_state
        assert prepared.profile["arch_hash"] == genotype.arch_hash
        assert prepared.profile["q_hash"] == plan.q_hash
        assert prepared.profile["calibration"]["formal_calibration_count"] is False
        assert prepared.profile["formal_eligible"] is False
        conversions = prepared.profile["head_input_conversions"]
        assert len(conversions) == 4
        assert all(n["input_bits"] == 8 and n["output_bits"] == 16 for n in conversions)
        images = next(calibration.batches())
        witness = verify_fake_integer_parity(prepared, images)
        assert witness["mismatches"] == 0
        assert witness["status"] == "PASS_SOFTWARE_INTEGER_FAKE_PARITY"
        assert witness["board_parity"] == "NOT_RUN"
        output = save_ptq_artifacts(tmp_path / "ptq", prepared, witness=witness)
        profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
        package = torch.load(output / "integer_parameters.pt", map_location="cpu", weights_only=True)
        for payload in (profile, manifest, package):
            assert payload["arch_hash"] == genotype.arch_hash
            assert payload["q_hash"] == plan.q_hash
            assert payload["quantization_plan"] == plan.to_dict()
        for filename, expected_sha in manifest["files"].items():
            assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected_sha
        assert verify_saved_artifacts(output, expected_arch_hash=genotype.arch_hash,
                                      expected_q_hash=plan.q_hash) == manifest
        with pytest.raises(FileExistsError, match="new empty"):
            save_ptq_artifacts(output, prepared, witness=witness)
        with pytest.raises(ValueError, match="different architecture/plan"):
            verify_saved_artifacts(output, expected_arch_hash=genotype.arch_hash,
                                   expected_q_hash=Q(genotype, "W8A8").q_hash)
        # Updating the outer file digest cannot disguise mutated integer tensors.
        parameter_bytes = (output / "integer_parameters.pt").read_bytes()
        parameter_sha = manifest["files"]["integer_parameters.pt"]
        first = next(iter(package["weights"]))
        package["weights"][first].view(-1)[0] ^= 1
        torch.save(package, output / "integer_parameters.pt")
        manifest["files"]["integer_parameters.pt"] = hashlib.sha256(
            (output / "integer_parameters.pt").read_bytes()).hexdigest()
        (output / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="tensor source hash drift"):
            verify_saved_artifacts(output, expected_arch_hash=genotype.arch_hash,
                                   expected_q_hash=plan.q_hash)
        (output / "integer_parameters.pt").write_bytes(parameter_bytes)
        manifest["files"]["integer_parameters.pt"] = parameter_sha
        (output / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    finally:
        torch.set_num_threads(previous_threads)
        print(f"Synthetic E1 HEAD_W8A16 integration seconds: {time.monotonic()-started:.3f}")
