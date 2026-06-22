"""Shared runtime helpers for search, retraining, and deployment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from hwnas_fpga.data import (
    NKSIDDataset,
    create_dummy_dataloaders,
    create_nksid_dataloaders,
    download_nksid_dataset,
)
from hwnas_fpga.hardware import (
    FPGACostEstimator,
    LutQueryEngine,
    LutTable,
    create_dummy_fpga_lut,
    resolve_board_profile,
)
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


# Full implementation-level capability set retained for legacy and lightweight
# profiles. The formal MobileNetV2 mainline now excludes fused_mbconv and
# mixconv from the default search space; denoise and edge remain research
# operators that can stay searchable while still being deployment-conditional.
DEFAULT_OP_CHOICES = (
    "conv",
    "dw_pw_conv",
    "mbconv",
    "skip",
    "denoise",
    "edge",
)


def load_config(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return payload


def pick(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def build_constraints(config: dict[str, Any]) -> SearchConstraints:
    constraints_cfg = config.get("constraints", {})
    physical_cfg = constraints_cfg.get("physical")
    if physical_cfg is None:
        physical_cfg = config.get("physical_constraints", {})
    return SearchConstraints(
        max_latency_ms=constraints_cfg.get("max_latency_ms"),
        max_energy_mj=constraints_cfg.get("max_energy_mj"),
        max_model_size_mb=constraints_cfg.get("max_model_size_mb"),
        max_dsp=constraints_cfg.get("max_dsp"),
        max_bram=constraints_cfg.get("max_bram"),
        max_lut=constraints_cfg.get("max_lut"),
        max_power_w=constraints_cfg.get("max_power_w"),
        max_memory_bandwidth_gbps=constraints_cfg.get("max_memory_bandwidth_gbps"),
        max_offchip_mem_mb=constraints_cfg.get("max_offchip_mem_mb"),
        physical=dict(physical_cfg or {}),
    )


def build_hardware_spec(config: dict[str, Any]) -> HardwareSpec:
    hardware_cfg = config.get("hardware", {})
    overrides = {
        "name": hardware_cfg.get("name"),
        "clock_mhz": hardware_cfg.get("clock_mhz"),
        "max_lut": hardware_cfg.get("max_lut"),
        "max_ff": hardware_cfg.get("max_ff"),
        "max_bram": hardware_cfg.get("max_bram"),
        "max_dsp": hardware_cfg.get("max_dsp"),
        "max_power_w": hardware_cfg.get("max_power_w"),
        "memory_bandwidth_gbps": hardware_cfg.get("memory_bandwidth_gbps"),
        "offchip_mem_mb": hardware_cfg.get("offchip_mem_mb"),
    }
    return resolve_board_profile(hardware_cfg.get("board"), overrides=overrides)


def load_anchor_profile_from_pool(
    pool_path: str | Path,
    anchor_role: str = "search_anchor",
) -> Optional[str]:
    """从 selected_backbone_pool.json 中解析指定锚点对应的 family_profile 名。

    pool JSON 结构:
        recommended_profiles:
            mobile_anchor:      {source_role: "search_anchor", ...}
            accuracy_biased:    {source_role: "accuracy_anchor", ...}
            lightweight_sonar:  {source_role: "lightweight_anchor", ...}

    返回 profile 名（如 "mobile_anchor"），找不到时返回 None。
    """
    path = Path(pool_path)
    if not path.is_file():
        raise FileNotFoundError(f"backbone pool not found: {path}")
    pool: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for profile_name, info in pool.get("recommended_profiles", {}).items():
        if info.get("source_role") == anchor_role:
            return str(profile_name)
    return None


def build_search_space(
    config: dict[str, Any],
    *,
    image_size: int,
    input_channels: int,
    num_classes: Optional[int],
    constraints: SearchConstraints,
) -> SearchSpace:
    search_cfg = config.get("search_space", {})
    payload = {
        "family_profile": search_cfg.get("family_profile"),
        "input_channels": input_channels,
        "image_size": image_size,
        "stem_channels": search_cfg.get("stem_channels"),
        "stem_stride": search_cfg.get("stem_stride"),
        "post_stem_downsample_stride": search_cfg.get("post_stem_downsample_stride"),
        "stage_strides": search_cfg.get("stage_strides"),
        "stage_base_channels": search_cfg.get("stage_base_channels"),
        "width_multipliers": search_cfg.get("width_multipliers"),
        "stage_channel_choices": search_cfg.get("stage_channel_choices"),
        "stage_depth_choices": search_cfg.get("stage_depth_choices"),
        "channel_choices": search_cfg.get("channel_choices"),
        "depth_choices": search_cfg.get("depth_choices"),
        "kernel_choices": search_cfg.get("kernel_choices"),
        "expand_choices": search_cfg.get("expand_choices"),
        "op_choices": search_cfg.get("op_choices"),
        "stage_block_choices": search_cfg.get("stage_block_choices"),
        "head_conv_channels": search_cfg.get("head_conv_channels"),
        "head_channels": search_cfg.get("head_channels"),
        "num_classes": num_classes,
        "hardware_constraints": constraints,
    }
    normalized_payload = {key: value for key, value in payload.items() if value is not None}
    for nullable_key in ("head_conv_channels", "head_channels"):
        if nullable_key in search_cfg:
            normalized_payload[nullable_key] = search_cfg.get(nullable_key)
    return SearchSpace(SearchSpaceConfig.from_dict(normalized_payload))


def load_lut_query_engine(config: dict[str, Any]) -> Optional[LutQueryEngine]:
    hardware_cfg = config.get("hardware", {})
    lut_path = hardware_cfg.get("lut_path")
    formal_lut_status_path = hardware_cfg.get("formal_lut_status_path")
    use_dummy_lut = bool(hardware_cfg.get("use_dummy_lut", False))
    enable_interpolation = bool(hardware_cfg.get("lut_enable_interpolation", False))
    allow_shape_only_match = bool(hardware_cfg.get("lut_allow_shape_only_match", True))
    strict_formal_lut = bool(hardware_cfg.get("strict_formal_lut", False))
    if strict_formal_lut and not formal_lut_status_path:
        raise ValueError(
            "hardware.strict_formal_lut requires hardware.formal_lut_status_path"
        )
    if lut_path:
        lut_file = Path(lut_path).expanduser()
        if not lut_file.is_absolute():
            lut_file = (Path.cwd() / lut_file).resolve()
        if not lut_file.exists():
            raise FileNotFoundError(f"LUT table file not found: {lut_file}")
        if lut_file.suffix.lower() == ".json":
            lut_table = LutTable.load_json(
                str(lut_file),
                default_clock_mhz=hardware_cfg.get("clock_mhz"),
            )
        else:
            lut_table = LutTable.load(str(lut_file))
        formal_status_entries = None
        if formal_lut_status_path:
            status_file = Path(formal_lut_status_path).expanduser()
            if not status_file.is_absolute():
                status_file = (Path.cwd() / status_file).resolve()
            if not status_file.exists():
                raise FileNotFoundError(f"Formal LUT status file not found: {status_file}")
            formal_status_entries = LutQueryEngine.load_formal_status_json(str(status_file))
        return LutQueryEngine(
            lut_table,
            enable_interpolation=enable_interpolation,
            allow_shape_only_match=allow_shape_only_match,
            strict_formal_lut=strict_formal_lut,
            formal_status_entries=formal_status_entries,
        )
    if use_dummy_lut:
        return LutQueryEngine(create_dummy_fpga_lut())
    return None


def _canonicalize_operator_policy_key(op: str) -> str:
    normalized = str(op).strip()
    if normalized.startswith("fused_mbconv"):
        return "fused_mbconv"
    if normalized.startswith("mixconv"):
        return "mixconv"
    if normalized.startswith("denoise"):
        return "denoise"
    if normalized.startswith("edge"):
        return "edge"
    return normalized


def load_operator_policies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hardware_cfg = config.get("hardware", {})
    manifest_path = hardware_cfg.get("operator_manifest_path")
    if manifest_path:
        manifest_file = Path(manifest_path).expanduser()
        if not manifest_file.is_absolute():
            manifest_file = (Path.cwd() / manifest_file).resolve()
    else:
        manifest_file = (
            Path(__file__).resolve().parents[2]
            / "hls_lut_builder"
            / "configs"
            / "operator_manifest.yaml"
        )

    if not manifest_file.exists():
        return {}

    payload = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}

    policies: dict[str, dict[str, Any]] = {}
    search_policy = payload.get("search_policy", {})
    if isinstance(search_policy, dict):
        for op in search_policy.get("default_mainline_search_ops", []):
            key = _canonicalize_operator_policy_key(op)
            policies.setdefault(key, {})
            policies[key].setdefault("search_enabled", True)
            policies[key].setdefault("deployment_status", "ready")
        for op in search_policy.get("conditional_deployment_ops", []):
            key = _canonicalize_operator_policy_key(op)
            policies.setdefault(key, {})
            policies[key]["search_enabled"] = True
            policies[key]["deployment_status"] = "conditional"
        for op in search_policy.get("paused_ops", []):
            key = _canonicalize_operator_policy_key(op)
            policies.setdefault(key, {})
            policies[key]["search_enabled"] = False
            policies[key]["deployment_status"] = "paused"
        for op in search_policy.get("dropped_current_target_ops", []):
            key = _canonicalize_operator_policy_key(op)
            policies.setdefault(key, {})
            policies[key]["search_enabled"] = False
            policies[key]["deployment_status"] = "dropped_current_target"

    operator_status = payload.get("operator_status", {})
    if isinstance(operator_status, dict):
        for op, policy in operator_status.items():
            if not isinstance(policy, dict):
                continue
            key = _canonicalize_operator_policy_key(op)
            policies.setdefault(key, {})
            policies[key].update(policy)

    return policies


def apply_operator_policies_to_search_space(
    search_space: SearchSpace,
    operator_policies: dict[str, dict[str, Any]],
) -> tuple[SearchSpace, dict[str, Any]]:
    original_ops = tuple(search_space.config.op_choices)
    kept_ops: list[str] = []
    removed_ops: list[dict[str, Any]] = []
    conditional_ops: list[dict[str, Any]] = []

    for op in original_ops:
        policy = dict(operator_policies.get(str(op).strip(), {}))
        if policy and not bool(policy.get("search_enabled", True)):
            removed_ops.append(
                {
                    "op": op,
                    "deployment_status": policy.get("deployment_status"),
                    "reason": policy.get("reason"),
                }
            )
            continue

        kept_ops.append(op)
        if str(policy.get("deployment_status", "")).strip() == "conditional":
            conditional_ops.append(
                {
                    "op": op,
                    "achieved_post_route_fmax_mhz": policy.get("achieved_post_route_fmax_mhz"),
                    "latency_rule": (
                        policy.get("hardware_cost_policy", {}) or {}
                    ).get("latency_rule"),
                    "deployable_at_200mhz": (
                        policy.get("hardware_cost_policy", {}) or {}
                    ).get("deployable_at_200mhz"),
                }
            )

    summary = {
        "original_ops": list(original_ops),
        "kept_ops": list(kept_ops),
        "removed_ops": removed_ops,
        "conditional_ops": conditional_ops,
    }

    if tuple(kept_ops) == original_ops:
        return search_space, summary

    payload = dict(vars(search_space.config))
    payload["op_choices"] = tuple(kept_ops)
    filtered_space = SearchSpace(SearchSpaceConfig.from_dict(payload))
    return filtered_space, summary


def build_cost_estimator(
    config: dict[str, Any],
    *,
    hardware_spec: HardwareSpec,
    constraints: SearchConstraints,
) -> FPGACostEstimator:
    hardware_cfg = config.get("hardware", {})
    return FPGACostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=hardware_cfg.get("quantization_bits", 8),
        lut_query_engine=load_lut_query_engine(config),
        operator_policies=load_operator_policies(config),
        require_deployable_operators=bool(
            hardware_cfg.get("require_deployable_operators", False)
        ),
        strict_formal_lut=bool(hardware_cfg.get("strict_formal_lut", False)),
        strict_lut_label=str(hardware_cfg.get("strict_lut_label", "formal LUT")),
        use_lut_for_head_layers=bool(hardware_cfg.get("use_lut_for_head_layers", False)),
    )


def create_data_pipeline(
    *,
    dataset_name: str,
    data_dir: Optional[str],
    batch_size: int,
    image_size: int,
    num_classes: Optional[int],
    input_channels: int,
    fold: int,
    num_workers: int,
    device: str,
    use_kfold: bool = True,
    valid_size: Optional[float | int] = None,
    split_seed: int = 42,
) -> tuple[Any, Any, Optional[torch.Tensor], int]:
    pin_memory = device.startswith("cuda")

    if dataset_name == "nksid":
        if not data_dir:
            download_nksid_dataset()
            raise ValueError("NKSID dataset requires --data-dir pointing to the extracted dataset root")
        train_loader, val_loader, class_weights = create_nksid_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            fold=fold,
            num_workers=num_workers,
            pin_memory=pin_memory,
            use_kfold=use_kfold,
            valid_size=valid_size,
            split_seed=split_seed,
        )
        resolved_num_classes = num_classes or NKSIDDataset.NUM_CLASSES
        return train_loader, val_loader, class_weights, resolved_num_classes

    resolved_num_classes = num_classes or 8
    train_loader, val_loader = create_dummy_dataloaders(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=resolved_num_classes,
        input_channels=input_channels,
        num_train=500,
        num_val=100,
    )
    return train_loader, val_loader, None, resolved_num_classes
