"""Shared runtime helpers for search, retraining, and deployment scripts."""

from __future__ import annotations

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


DEFAULT_OP_CHOICES = (
    "conv",
    "dw_pw_conv",
    "mbconv",
    "fused_mbconv",
    "skip",
    "mixconv",
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
    )


def build_hardware_spec(config: dict[str, Any]) -> HardwareSpec:
    hardware_cfg = config.get("hardware", {})
    overrides = {
        "name": hardware_cfg.get("name") or hardware_cfg.get("board"),
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


def build_search_space(
    config: dict[str, Any],
    *,
    image_size: int,
    input_channels: int,
    num_classes: Optional[int],
    constraints: SearchConstraints,
) -> SearchSpace:
    search_cfg = config.get("search_space", {})
    return SearchSpace(
        SearchSpaceConfig(
            input_channels=input_channels,
            image_size=image_size,
            stem_channels=search_cfg.get("stem_channels", 16),
            stem_stride=search_cfg.get("stem_stride", 2),
            stage_strides=tuple(search_cfg.get("stage_strides", (1, 2, 2, 2))),
            channel_choices=tuple(search_cfg.get("channel_choices", (16, 24, 32, 48, 64, 96))),
            depth_choices=tuple(search_cfg.get("depth_choices", (1, 2, 3, 4))),
            kernel_choices=tuple(search_cfg.get("kernel_choices", (3, 5))),
            expand_choices=tuple(search_cfg.get("expand_choices", (1, 2, 4))),
            op_choices=tuple(search_cfg.get("op_choices", DEFAULT_OP_CHOICES)),
            head_channels=search_cfg.get("head_channels"),
            num_classes=num_classes,
            hardware_constraints=constraints,
        )
    )


def load_lut_query_engine(config: dict[str, Any]) -> Optional[LutQueryEngine]:
    hardware_cfg = config.get("hardware", {})
    lut_path = hardware_cfg.get("lut_path")
    use_dummy_lut = bool(hardware_cfg.get("use_dummy_lut", False))
    if lut_path:
        lut_file = Path(lut_path).expanduser()
        if not lut_file.is_absolute():
            lut_file = (Path.cwd() / lut_file).resolve()
        if not lut_file.exists():
            raise FileNotFoundError(f"LUT table file not found: {lut_file}")
        return LutQueryEngine(LutTable.load(str(lut_file)))
    if use_dummy_lut:
        return LutQueryEngine(create_dummy_fpga_lut())
    return None


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
