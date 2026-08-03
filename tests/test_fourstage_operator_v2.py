from pathlib import Path
import json

import numpy as np
from PIL import Image
import pytest
import torch

from hwnas_fpga.data.dataset import FrozenGeometryTransform, get_sonar_transforms
from hwnas_fpga.dir_int8_reference import (
    deterministic_weights,
    dir_mbconv3_split11_e3_v1_int8,
    round_shift_signed,
)
from hwnas_fpga.experiment_contract import validate_formal_values_against_contract
from hwnas_fpga.fourstage_int8_reference import (
    collect_activation_stats,
    full_network_int8_reference,
)
from hwnas_fpga.fourstage_operator import (
    architecture_sha256,
    build_fourstage_architecture,
    enumerate_base8,
    enumerate_extended,
    parameter_and_mac_count,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import (
    DirMBConv3Split11E3V1Block,
    SplitDW3ControlBlock,
    build_model,
)
from hwnas_fpga.search_space import ArchitectureSpec
from hwnas_fpga.fourstage_selection import (
    group_stress_split,
    infer_hash_groups,
)


def test_base8_is_complete_unique_factorial_and_k5_is_real():
    rows = enumerate_base8()
    assert len(rows) == 8
    assert len({row.arch_id for row in rows}) == 8
    assert len({architecture_sha256(row.architecture) for row in rows}) == 8
    assert {
        (row.kernel, row.expansion, row.stage4) for row in rows
    } == {
        (kernel, expansion, stage4)
        for kernel in (3, 5)
        for expansion in (3, 6)
        for stage4 in ("MBConv", "Skip")
    }
    for row in rows:
        validate_frozen_fourstage(row.architecture)
        assert row.architecture.stages[1].blocks[0].kernel_size == row.kernel


def test_frozen_fourstage_has_expected_shape_gradient_params_and_macs():
    architecture = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="mbconv_k3_e3",
    )
    counts = parameter_and_mac_count(architecture)
    assert counts["input_shape"] == [1, 1, 224, 224]
    assert counts["output_shape"] == [1, 8]
    assert counts["parameter_count"] > 0
    assert counts["macs"] > 0


def test_dir_v1_and_split_control_have_frozen_stage4_graphs():
    dir_arch = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="dir_mbconv3_split11_e3_v1",
    )
    control_arch = build_fourstage_architecture(
        stage2_kernel=3,
        stage2_expansion=3,
        stage4_op="split_dw3_control",
    )
    dir_model = build_model(dir_arch, num_classes=8)
    control_model = build_model(control_arch, num_classes=8)
    dir_block = dir_model.stages[3][0]
    control_block = control_model.stages[3][0]
    assert isinstance(dir_block, DirMBConv3Split11E3V1Block)
    assert isinstance(control_block, SplitDW3ControlBlock)
    assert dir_block.branch_order == ("dw_1x3", "dw_3x1")
    assert dir_block.dw_1x3.kernel_size == (1, 3)
    assert dir_block.dw_3x1.kernel_size == (3, 1)
    assert dir_block.branch_channels == 48
    assert control_block.dw_3x3_first.kernel_size == (3, 3)
    assert control_block.dw_3x3_second.kernel_size == (3, 3)

    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    output = dir_model(x)
    assert output.shape == (2, 8)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dir_v1_serialization_hash_and_matched_control_costs(tmp_path):
    dir_arch = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=3,
        stage4_op="dir_mbconv3_split11_e3_v1",
    )
    restored = ArchitectureSpec.from_dict(dir_arch.to_dict())
    assert restored == dir_arch
    assert architecture_sha256(restored) == architecture_sha256(dir_arch)

    model = build_model(dir_arch, num_classes=8).eval()
    checkpoint = tmp_path / "dir_v1.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_model(restored, num_classes=8).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    probe = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(probe), reloaded(probe))

    dir_counts = parameter_and_mac_count(dir_arch)
    control_counts = parameter_and_mac_count(
        build_fourstage_architecture(
            stage2_kernel=5,
            stage2_expansion=3,
            stage4_op="split_dw3_control",
        )
    )
    k3_counts = parameter_and_mac_count(
        build_fourstage_architecture(
            stage2_kernel=5,
            stage2_expansion=3,
            stage4_op="mbconv_k3_e3",
        )
    )
    assert control_counts["parameter_count"] == k3_counts["parameter_count"]
    assert control_counts["macs"] == k3_counts["macs"]
    assert dir_counts["parameter_count"] < control_counts["parameter_count"]
    assert dir_counts["macs"] < control_counts["macs"]


def test_extended_space_is_12_until_stage4_k5_hardware_gate_opens():
    rows12 = enumerate_extended(include_stage4_k5=False)
    assert len(rows12) == 12
    assert sum(row.stage4 == "Dir-MBConv3-e3" for row in rows12) == 4
    assert all(row.stage4 != "MBConv-k5-e3" for row in rows12)
    rows16 = enumerate_extended(include_stage4_k5=True)
    assert len(rows16) == 16
    assert sum(row.stage4 == "MBConv-k5-e3" for row in rows16) == 4


def test_dir_v1_rejects_non_frozen_placement():
    with pytest.raises(ValueError, match="frozen to 32->32"):
        DirMBConv3Split11E3V1Block(24, 32, stride=2, expand_ratio=3)


def test_dir_v1_integer_reference_is_deterministic_and_saturating():
    values = np.asarray([-1536, -512, 512, 1536], dtype=np.int32)
    assert round_shift_signed(values, 10).tolist() == [-2, -1, 1, 2]
    weights = deterministic_weights()
    for value in (-128, 0, 127):
        inputs = np.full((32, 28, 28), value, dtype=np.int8)
        first = dir_mbconv3_split11_e3_v1_int8(inputs, weights)
        second = dir_mbconv3_split11_e3_v1_int8(inputs, weights)
        assert first.shape == (32, 28, 28)
        assert first.dtype == np.int8
        assert np.array_equal(first, second)


def test_frozen_geometry_contracts_are_deterministic_and_bounded():
    image = Image.fromarray(np.arange(120 * 80, dtype=np.uint8).reshape(80, 120))
    letterbox = FrozenGeometryTransform(
        image_size=224,
        mode="letterbox_224",
        padding_value=7,
    )
    first = np.asarray(letterbox(image))
    second = np.asarray(letterbox(image))
    assert first.shape == (224, 224)
    assert np.array_equal(first, second)
    assert np.all(first[:37] == 7)

    fixed = FrozenGeometryTransform(
        image_size=224,
        mode="fixed_scale_pad_224",
        fixed_scale_factor=224 / 714,
        padding_value=0,
    )
    assert np.asarray(fixed(image)).shape == (224, 224)
    with pytest.raises(ValueError, match="overflows target canvas"):
        fixed(Image.new("L", (716, 714)))


def test_cached_geometry_path_is_tensor_identical_without_augmentation():
    image = Image.fromarray(np.arange(120 * 80, dtype=np.uint8).reshape(80, 120))
    geometry = FrozenGeometryTransform(
        image_size=224,
        mode="letterbox_224",
        padding_value=0,
    )
    full = get_sonar_transforms(
        image_size=224,
        is_training=False,
        geometry_mode="letterbox_224",
        augmentation_profile="none",
    )
    post_geometry = get_sonar_transforms(
        image_size=224,
        is_training=False,
        geometry_mode="letterbox_224",
        augmentation_profile="none",
        geometry_already_applied=True,
    )
    assert torch.equal(full(image), post_geometry(geometry(image)))


def test_inferred_groups_never_cross_stress_split():
    samples = [(f"sample_{index}.png", index // 4) for index in range(8)]
    hashes = {0: 0, 1: 0, 2: 7, 3: 15, 4: 0, 5: 0, 6: 7, 7: 15}
    groups = infer_hash_groups(
        samples,
        list(range(8)),
        hamming_threshold=0,
        precomputed_hashes=hashes,
    )
    train, inner, metadata = group_stress_split(
        samples,
        list(range(8)),
        groups,
        seed=42,
        fraction=0.25,
    )
    assert set(train) | set(inner) == set(range(8))
    assert set(train).isdisjoint(inner)
    assert {groups[index] for index in train}.isdisjoint(
        {groups[index] for index in inner}
    )
    assert "not real acquisition" in metadata["claim_boundary"]


def test_contract_rejects_recipe_or_geometry_drift():
    contract = {
        "dataset": {
            "image_size": 224,
            "input_channels": 1,
            "geometry_mode": "stretch_224",
            "fixed_scale_factor": None,
            "geometry_padding_value": 0,
            "augmentation_profile": "none",
            "inner_val_fraction": 0.15,
        },
        "training_recipe": {
            "epochs": 150,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
            "scheduler": "cosine_with_warmup",
            "warmup_epochs": 0,
            "min_lr_ratio": 0.01,
            "label_smoothing": 0.0,
            "logit_adjust_tau": 0.0,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "amp": True,
        },
    }
    dataset = dict(contract["dataset"])
    recipe = dict(contract["training_recipe"])
    validate_formal_values_against_contract(
        contract=contract,
        observed_dataset=dataset,
        observed_recipe=recipe,
    )
    dataset["geometry_mode"] = "letterbox_224"
    with pytest.raises(ValueError, match="dataset.geometry_mode"):
        validate_formal_values_against_contract(
            contract=contract,
            observed_dataset=dataset,
            observed_recipe=recipe,
        )
    dataset = dict(contract["dataset"])
    recipe["weight_decay"] = 0.0001
    with pytest.raises(ValueError, match="training_recipe.weight_decay"):
        validate_formal_values_against_contract(
            contract=contract,
            observed_dataset=dataset,
            observed_recipe=recipe,
        )


def test_deployment_selection_freezes_k5_queue_and_boundaries():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "fourstage_deployment_candidate_selection.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "GENERAL_OP_SELECTED"
    assert payload["stage"] == "FULL_NETWORK_DEPLOYMENT_CLOSURE_QUEUE_FROZEN"
    assert payload["selected_candidate_count"] == 4
    assert payload["frozen_experiment_conclusion"]["evaluated_structure_count"] == 16
    assert (
        payload["frozen_experiment_conclusion"][
            "dir_mbconv3_split11_e3_v1"
        ]
        == "NOT_ADMITTED_ACCURACY_GATE_FAILED"
    )
    assert payload["hardware_claim_boundary"]["complete_network_route"] == "NOT_RUN"
    assert payload["hardware_claim_boundary"]["power"] == "NOT_MEASURED"
    selected = {row["role"]: row["arch_id"] for row in payload["selected_candidates"]}
    assert selected["original_baseline"] == "fourstage_s2_k3_e3_s4_mbconv_k3_e3"
    assert selected["stage2_k5_representative"].startswith("fourstage_s2_k5_")
    assert selected["stage4_k5_representative"].endswith("s4_mbconv_k5_e3")
    assert selected["low_cost_skip_representative"].endswith("s4_skip")
    for row in payload["selected_candidates"]:
        assert (
            row["deployment_state"]
            == "PENDING_FULL_NETWORK_HARDWARE_CLOSURE"
        )
        assert row["formal_protocol_units"]["expected"] == 15
        assert row["formal_protocol_units"]["checkpoint_count"] == 15
        assert row["formal_protocol_units"]["outer_prediction_count"] == 15


def test_checkpoint_export_gate_does_not_claim_downstream_hardware():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "fourstage_checkpoint_export_summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["gate"] == "real_checkpoint_export"
    assert payload["candidate_count"] == 4
    assert (
        payload["downstream_gates"]["int8_activation_calibration"]
        == "PENDING_REAL_ACTIVATION_CALIBRATION"
    )
    assert payload["downstream_gates"]["bitstream"] == "NOT_GENERATED"
    assert payload["downstream_gates"]["external_meter_power"] == "NOT_MEASURED"
    for row in payload["candidates"]:
        assert row["source_checkpoint"]["path"].endswith("best_fold0_seed42.pt")
        assert row["quantization_contract"]["status"] == "WEIGHT_EXPORT_ONLY"
        assert row["quantization_contract"]["parity_ready"] is False


def test_fourstage_int8_reference_runs_on_supported_mature_candidate():
    architecture = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=3,
        stage4_op="mbconv_k5_e3",
    )
    model = build_model(architecture, num_classes=8).eval()
    inputs = torch.linspace(-1.0, 1.0, steps=1 * 1 * 64 * 64).reshape(1, 1, 64, 64)
    calibration = collect_activation_stats(
        model,
        [(inputs, torch.tensor([0]))],
        max_samples=1,
    )
    result = full_network_int8_reference(model, inputs, calibration)
    assert result["logits_int8"].shape == (1, 8)
    assert result["logits_int8"].dtype == torch.int8
    assert result["argmax_match"].shape == (1,)


def test_int8_reference_gate_keeps_hls_and_board_pending():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "fourstage_int8_reference_summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["gate"] == "activation_calibrated_python_int8_reference"
    assert payload["candidate_count"] == 4
    assert payload["calibration_dataset"]["outer_validation_accessed"] is False
    assert payload["calibration_dataset"]["inner_validation_accessed"] is False
    assert payload["downstream_gates"]["hls_c_sim"] == "PENDING"
    assert payload["downstream_gates"]["bitstream"] == "NOT_GENERATED"
    assert payload["downstream_gates"]["external_meter_power"] == "NOT_MEASURED"
    for row in payload["candidates"]:
        assert row["status"] == "PASS"
        assert row["calibration_samples"] > 0
        assert row["scale_count"] >= 8
