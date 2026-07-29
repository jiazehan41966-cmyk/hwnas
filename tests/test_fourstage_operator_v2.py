from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from hwnas_fpga.data.dataset import FrozenGeometryTransform, get_sonar_transforms
from hwnas_fpga.experiment_contract import validate_formal_values_against_contract
from hwnas_fpga.fourstage_operator import (
    architecture_sha256,
    build_fourstage_architecture,
    enumerate_base8,
    parameter_and_mac_count,
    validate_frozen_fourstage,
)
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
