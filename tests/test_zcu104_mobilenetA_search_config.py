import json
from pathlib import Path

import pytest
import torch

from hwnas_fpga.models import build_model
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/search/zcu104_fourstage_cfem20_v1.yaml",
    ROOT / "configs/search/zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml",
)
CURRENT_MOBILENETA_CONFIG = (
    ROOT / "configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml"
)
FROZEN_RL_ARCH_13 = (
    ROOT
    / "artifacts/zcu104_full_campaign_v1/frozen_candidates/mobilenetA_rl200_20260908_155713/candidates/rl_arch_13.frozen_candidate.json"
)
FROZEN_V3_RL_ARCH_50 = (
    ROOT
    / "artifacts/zcu104_full_campaign_v1/frozen_candidates/mobilenetA_v3gap1280_rl200_20260910_202352/candidates/rl_arch_50.frozen_candidate.json"
)


def _load_space(config_path: Path):
    config = load_config(str(config_path))
    return config, build_search_space(
        config,
        image_size=int(config["dataset"]["image_size"]),
        input_channels=int(config["dataset"]["input_channels"]),
        num_classes=int(config["dataset"]["num_classes"]),
        constraints=build_constraints(config),
    )


def test_zcu104_configs_use_mobilenetA_full_dmdc_cfem_search_architecture() -> None:
    for config_path in CONFIGS:
        config, space = _load_space(config_path)
        raw_space = config["search_space"]

        assert raw_space["search_space_version"] == "mobilenetA_full_dmdc_cfem_v1"
        assert raw_space["macro_type"] == "mobilenetA_full_dmdc_cfem"
        assert space.config.stem_channels == 32
        assert space.config.stem_stride == 2
        assert space.config.stage_strides == (1, 2, 2, 1, 2, 1, 2, 1)
        assert space.config.stage_channel_choices == (
            (16,),
            (24,),
            (32,),
            (32,),
            (64,),
            (96,),
            (160,),
            (320,),
        )
        assert [len(choices) for choices in space.config.stage_block_choices] == [
            1,
            4,
            4,
            5,
            4,
            5,
            4,
            5,
        ]
        assert space.config.head_conv_channels == 1280


def test_zcu104_mobilenetA_baseline_resolves_expected_lut_shapes() -> None:
    _config, space = _load_space(CONFIGS[1])
    architecture = space.baseline_architecture()
    resolved = space.resolve_blocks(architecture)

    assert [
        (
            block.in_channels,
            block.out_channels,
            block.input_resolution,
            block.output_resolution,
            block.stride,
        )
        for block in resolved
    ] == [
        (32, 16, 112, 112, 1),
        (16, 24, 112, 56, 2),
        (24, 32, 56, 28, 2),
        (32, 32, 28, 28, 1),
        (32, 64, 28, 14, 2),
        (64, 96, 14, 14, 1),
        (96, 160, 14, 7, 2),
        (160, 320, 7, 7, 1),
    ]
    assert resolved[3].op == "skip"


def test_zcu104_mobilenetA_stage5_stage7_skip_is_filtered_until_legal() -> None:
    _config, space = _load_space(CONFIGS[1])

    stage5_first_block = space.concrete_block_choices_for_stage(
        stage_index=5,
        in_channels=64,
        out_channels=96,
        stride=1,
    )
    stage7_first_block = space.concrete_block_choices_for_stage(
        stage_index=7,
        in_channels=160,
        out_channels=320,
        stride=1,
    )

    assert {block.op for block in stage5_first_block} == {"mbconv"}
    assert {block.op for block in stage7_first_block} == {"mbconv"}


def test_zcu104_mobilenetA_baseline_model_forward_is_8_class() -> None:
    config, space = _load_space(CONFIGS[1])
    architecture = space.baseline_architecture()
    model = build_model(architecture, num_classes=int(config["dataset"]["num_classes"]))
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 1, 224, 224))

    assert tuple(output.shape) == (1, 8)


def test_current_mobilenetA_proxy_rejects_known_lutram_false_feasible_candidate() -> None:
    if not CURRENT_MOBILENETA_CONFIG.is_file() or not FROZEN_RL_ARCH_13.is_file():
        pytest.skip("requires local ZCU104 MobileNetA formal LUT and frozen candidate artifacts")
    config = load_config(str(CURRENT_MOBILENETA_CONFIG))
    hardware_cfg = config.get("hardware", {}) if isinstance(config.get("hardware"), dict) else {}
    for key in (
        "operator_cost_lut_path",
        "operator_cost_lut_coverage_path",
        "operator_cost_lut_preflight_path",
    ):
        path = ROOT / str(hardware_cfg.get(key, ""))
        if not path.is_file():
            pytest.skip(f"requires local {key} artifact")

    constraints = build_constraints(config)
    dataset_cfg = config["dataset"]
    space = build_search_space(
        config,
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_classes=int(dataset_cfg["num_classes"]),
        constraints=constraints,
    )
    try:
        # The historical 88-row table lacks the classifier's 1280-channel GAP.
        # Reject that evidence before making any candidate feasibility claim.
        lut = json.loads((ROOT / hardware_cfg["operator_cost_lut_path"]).read_text(encoding="utf-8"))
        if not any(row.get("shape_name") == "gap_7_1280" for row in lut["rows"]):
            with pytest.raises(ValueError, match="gap_7_1280|hardware LUT version"):
                build_cost_estimator(config, hardware_spec=build_hardware_spec(config), constraints=constraints)
            return
        estimator = build_cost_estimator(
            config,
            hardware_spec=build_hardware_spec(config),
            constraints=constraints,
        )
    except ValueError as exc:
        assert "global_avg_pool/gap_7_1280" in str(exc) or "hardware LUT version" in str(exc)
        return
    payload = json.loads(FROZEN_RL_ARCH_13.read_text(encoding="utf-8"))
    architecture = ArchitectureSpec.from_dict(payload["candidate"]["encoding"])

    estimate = estimator.estimate(architecture, space)

    assert estimate.physical_metrics["zcu104_operator_lutram_heuristic_pressure"] > 1.0
    assert estimate.physical_metrics["zcu104_predicted_lut_as_memory_pressure"] > 1.0
    assert any("zcu104_predicted_lut_as_memory_pressure" in item for item in estimate.violations)


def test_current_mobilenetA_proxy_rejects_v3_headconv1280_full_route_false_feasible() -> None:
    if not CURRENT_MOBILENETA_CONFIG.is_file() or not FROZEN_V3_RL_ARCH_50.is_file():
        pytest.skip("requires local v3 ZCU104 MobileNetA frozen candidate artifacts")
    config = load_config(str(CURRENT_MOBILENETA_CONFIG))
    constraints = build_constraints(config)
    dataset_cfg = config["dataset"]
    space = build_search_space(
        config,
        image_size=int(dataset_cfg["image_size"]),
        input_channels=int(dataset_cfg["input_channels"]),
        num_classes=int(dataset_cfg["num_classes"]),
        constraints=constraints,
    )
    estimator = build_cost_estimator(
        config,
        hardware_spec=build_hardware_spec(config),
        constraints=constraints,
    )
    payload = json.loads(FROZEN_V3_RL_ARCH_50.read_text(encoding="utf-8"))
    architecture = ArchitectureSpec.from_dict(payload["candidate"]["encoding"])

    estimate = estimator.estimate(architecture, space)

    assert estimate.physical_metrics["zcu104_lut_as_memory_scale"] == pytest.approx(2.01)
    assert estimate.physical_metrics["zcu104_predicted_lut_as_memory_pressure"] > 1.0
    assert estimate.physical_metrics["zcu104_predicted_bram18_pressure"] > 1.0
    assert any("zcu104_predicted_lut_as_memory_pressure" in item for item in estimate.violations)
    assert any("zcu104_predicted_bram18_pressure" in item for item in estimate.violations)
