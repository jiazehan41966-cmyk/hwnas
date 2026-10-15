"""Isolated synthetic fixtures; no dataset or historical metrics are touched."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hwnas_fpga.deploy.h2_ptq import prepare_h2_ptq
from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from tools.fpga_h2 import numeric
from tools.fpga_h2.interface import pack, unpack
from tools.sonar_deploy.ptq_export import (
    IntegerGraphModel, exact_conv_torch, export_ptq_package,
    requant_torch, restore_archived_ptq, wrap32_torch,
)


@pytest.fixture(scope="module")
def calibrated():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    arch = {"output_combination": "P3_P4_P5", "kernels": {f"stage{i}": 3 for i in (1, 2, 4, 5, 6, 7)}}
    model = build_sonar_candidate(SPEC_PATH, arch, image_size=64).eval()
    image = torch.rand(1, 1, 64, 64)
    _, quantized, _, profile = prepare_h2_ptq(model, [image, image * 0.5], device="cpu")
    yield model, quantized, profile, image
    torch.set_num_threads(previous)


def export_fixture(tmp_path, calibrated, name="package"):
    model, _, profile, _ = calibrated
    model = deepcopy(model)
    model.image_size = 640
    provenance = {"checkpoint_sha256": "a" * 64, "profile_sha256": "b" * 64, "test_evaluated": False, "fixture_only": True}
    return export_ptq_package(model, profile, tmp_path / name, provenance)


def test_restore_replays_archived_float_ptq(calibrated):
    model, expected, profile, image = calibrated
    _, restored = restore_archived_ptq(model, profile)
    with torch.no_grad():
        for a, b in zip(expected(image)["predictions"], restored(image)["predictions"]):
            assert torch.equal(a, b)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in restored.modules())


def test_input_model_unchanged(calibrated):
    model, _, profile, _ = calibrated
    original = {k: v.clone() for k, v in model.state_dict().items()}
    restore_archived_ptq(model, profile)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())


def test_deterministic_export_and_descriptors(tmp_path, calibrated):
    graph = export_fixture(tmp_path, calibrated)
    export_fixture(tmp_path, calibrated, "repeat")
    for name in ("graph.json", "export_manifest.json", "weights.bin", "biases.bin", "layers.bin"):
        assert (tmp_path / "package" / name).read_bytes() == (tmp_path / "repeat" / name).read_bytes()
    assert list(graph["outputs"]) == ["P3", "P4", "P5"]
    for descriptor in graph["descriptors"]:
        level = descriptor["name"]
        assert graph["edges"][graph["outputs"][level]]["shape"] == [21, 640 // descriptor["stride"], 640 // descriptor["stride"]]
        arr = np.ones((21, descriptor["height"], descriptor["width"]), dtype=np.int8)
        assert np.array_equal(arr, unpack(pack(arr), arr.shape))


def test_consumer_bridges_and_historical_fpn_indices(tmp_path, calibrated):
    graph = export_fixture(tmp_path, calibrated)
    profile = calibrated[2]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert any(n["op"] == "requant" for n in nodes.values())
    for layer in profile["layers"]:
        n = nodes[layer["name"]]
        assert graph["edges"][n["inputs"][0]]["exp"] == layer["input_exp"]
    for n in graph["nodes"]:
        if n["op"] == "upsample_add":
            index = int(n["id"].rsplit(".", 1)[1])
            s = profile["fpn_adds"][index]
            assert (n["input_exp_a"], n["input_exp_b"], n["output_exp"]) == (s["left_exp"], s["right_exp"], s["output_exp"])
            assert graph["edges"][n["inputs"][0]]["exp"] == s["left_exp"]
            assert graph["edges"][n["inputs"][1]]["exp"] == s["right_exp"]
    assert graph["fpn_history"]["saved_record_count"] == 4
    assert graph["fpn_history"]["used_record_indices"] == [0, 1]


@pytest.mark.parametrize("shift", [0, 1, 2, 7, 31])
def test_signed_rounding_saturation_and_extremes(shift):
    vals = np.array([-(2**31), -257, -255, -7, -5, -3, -1, 0, 1, 3, 5, 7, 255, 257, 2**31 - 1], dtype=np.int64)
    for mult in (1, 3, 2**30):
        expected = numeric.requantize_int32_to_int8(vals, mult, shift)
        assert np.array_equal(requant_torch(torch.from_numpy(vals), mult, shift).numpy(), expected)


def test_modulo_overflow():
    vals = np.array([-(2**32), -(2**31)-1, -(2**31), 2**31-1, 2**31, 2**32+12], dtype=np.int64)
    assert np.array_equal(wrap32_torch(torch.from_numpy(vals)).numpy(), numeric.wrap32(vals))


@pytest.mark.parametrize("kernel,stride,groups", [(1, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 3), (3, 2, 3), (5, 1, 3), (5, 2, 3)])
def test_exact_convolution_against_independent_numpy(kernel, stride, groups):
    rng = np.random.default_rng(42)
    x = rng.integers(-128, 128, (3, 11, 11), dtype=np.int8)
    co = 3 if groups == 3 else 5
    w = rng.integers(-128, 128, (co, 3 // groups, kernel, kernel), dtype=np.int8)
    b = np.array([2**31 - 1 if i % 2 else -(2**31) for i in range(co)], dtype=np.int32)
    for act in ("linear", "relu", "relu6"):
        node = dict(stride=stride, padding=kernel // 2, groups=groups, multiplier=1, shift=25, activation=act, output_exp=-2, q6=24)
        expected = numeric.conv(x, w, b, node)
        actual = exact_conv_torch(torch.from_numpy(x)[None], torch.from_numpy(w), torch.from_numpy(b), node)[0].numpy()
        assert np.array_equal(expected, actual)


def test_full_graph_forward_shapes_finite_and_lifetime(tmp_path, calibrated):
    export_fixture(tmp_path, calibrated)
    model = IntegerGraphModel(tmp_path / "package", calibrated[0]).eval()
    image = torch.zeros(1, 1, 640, 640)
    outputs, retained = model.forward_quantized(image, retain=False)
    assert set(retained) == set(model.graph["outputs"].values())
    assert [tuple(v.shape) for v in outputs.values()] == [(1, 21, 80, 80), (1, 21, 40, 40), (1, 21, 20, 20)]
    assert all(v.dtype == torch.int8 for v in outputs.values())
    assert all(torch.isfinite(v).all() for v in model(image)["predictions"])


def test_corruption_fails_closed(tmp_path, calibrated):
    export_fixture(tmp_path, calibrated)
    path = tmp_path / "package" / "weights.bin"
    data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA mismatch"):
        IntegerGraphModel(tmp_path / "package")


@pytest.mark.parametrize("change", ["scale", "layer", "mixed", "contract", "fpn"])
def test_invalid_profile_rejected(calibrated, change):
    model, _, p, _ = calibrated
    p = deepcopy(p)
    if change == "scale": p["layers"][0]["multiplier"] = 3
    if change == "layer": p["layers"].pop()
    if change == "mixed": p["layers"][0]["weight_bits"] = 16
    if change == "contract": p["numeric_contract_hash"] = "0" * 64
    if change == "fpn": p["fpn_adds"] = []
    with pytest.raises(ValueError): restore_archived_ptq(model, p)


def test_p2_four_output_mixed_kernel_graph(tmp_path):
    torch.manual_seed(4)
    arch = {"output_combination": "P2_P3_P4_P5", "kernels": {f"stage{i}": 3 if i == 1 else 5 for i in (1, 2, 4, 5, 6, 7)}}
    model = build_sonar_candidate(SPEC_PATH, arch, image_size=32).eval()
    _, _, _, profile = prepare_h2_ptq(model, [torch.rand(1, 1, 32, 32)], device="cpu")
    model.image_size = 640
    graph = export_ptq_package(model, profile, tmp_path / "p2", {"checkpoint_sha256": "a"*64, "profile_sha256": "b"*64})
    assert list(graph["outputs"]) == ["P2", "P3", "P4", "P5"]
    assert graph["edges"][graph["outputs"]["P2"]]["shape"] == [21, 160, 160]
    assert len([n for n in graph["nodes"] if n["op"] == "upsample_add"]) == 3
    assert {n["kernel"] for n in graph["nodes"] if n["id"].endswith("dw_conv")} == {3, 5}


def test_historical_trace_covers_all_graph_nodes(tmp_path, calibrated):
    from tools.sonar_deploy.parity import _historical_trace
    graph = export_fixture(tmp_path, calibrated)
    historical = deepcopy(calibrated[1]); historical.image_size = 640
    records = _historical_trace(historical, torch.zeros(1, 1, 640, 640), graph)
    assert set(records) == {node["id"] for node in graph["nodes"]}
    for name, value in records.items():
        assert list(value.shape[1:]) == graph["edges"][name]["shape"]
        assert torch.isfinite(value).all()


def test_trace_rejects_non_train_before_iteration():
    from tools.sonar_deploy.parity import _train_prefix
    manifest = {"split": "valid", "test_evaluated": False, "entries": []}
    with pytest.raises(ValueError, match="train-only64"):
        next(_train_prefix(manifest))


def test_export_rejects_false_preparation_backend(tmp_path, calibrated):
    model, _, profile, _ = calibrated
    provenance = {"checkpoint_sha256": "a" * 64, "profile_sha256": "b" * 64,
                  "parameter_preparation_device": "cuda"}
    with pytest.raises(ValueError, match="preparation device"):
        export_ptq_package(model, profile, tmp_path / "wrong_device", provenance)


def test_train_prefix_uses_frozen_lexical_paths_and_no_augmentation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.sonar_deploy import parity
    from phase5a_quantization import common
    import baseline as frozen
    data = tmp_path / "isolated" / "data"
    entries = [dict(image=f"train/images/{i:03d}.jpg", annotation=f"train/labels/{i:03d}.txt",
                    image_sha256="a"*64, annotation_sha256="a"*64) for i in range(64)]
    calls = []
    class Dataset:
        root = data
        samples = [data / e["image"] for e in entries]
        def __getitem__(self, index):
            calls.append(index)
            return torch.full((1, 640, 640), float(index)), {}
    def make_loader(config, root, split, seed, workers, *, train):
        assert root == data and split == "train" and seed == 42 and workers == 0 and train is False
        return SimpleNamespace(dataset=Dataset())
    monkeypatch.setattr(common, "baseline", lambda: {"dataset": {"data_dir": str(data)}})
    monkeypatch.setattr(frozen, "make_loader", make_loader)
    monkeypatch.setattr(parity, "sha", lambda path: "a"*64)
    manifest = dict(split="train", test_evaluated=False, augmentation=False, shuffle=False,
                    calibration_seed=42, batch_size=4, entries=entries)
    rows = list(parity._train_prefix(manifest))
    assert calls == list(range(8))
    assert all(tuple(image.shape) == (1, 1, 640, 640) for _, _, image in rows)
    manifest["entries"] = list(reversed(entries))
    with pytest.raises(ValueError, match="order drifted"):
        next(parity._train_prefix(manifest))
