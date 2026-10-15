"""Analytic cost supplement for the fixed multi-scale detection branch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from hwnas_fpga.hardware.cost import (
    BRAM_BLOCK_BYTES,
    CostEstimate,
    FPGACostEstimator,
    LayerCost,
    _div_up,
)
from hwnas_fpga.search_space import (
    ArchitectureSpec,
    ResolvedBlockSpec,
    SearchSpace,
)


DEFAULT_DETECTION_FEATURE_INDICES = (3, 5, 7)


@dataclass(frozen=True)
class FixedMultiscaleDetectionCostConfig:
    num_classes: int
    feature_stage_indices: tuple[int, ...] = DEFAULT_DETECTION_FEATURE_INDICES
    neck_channels: int = 96
    anchors_per_location: int = 3

    @property
    def prediction_channels(self) -> int:
        return int(self.anchors_per_location) * (int(self.num_classes) + 5)


def _backbone_only_space(search_space: SearchSpace) -> SearchSpace:
    config = replace(
        search_space.config,
        head_conv_channels=None,
        head_channels=None,
        num_classes=None,
    )
    return SearchSpace(config)


def _backbone_only_architecture(architecture: ArchitectureSpec) -> ArchitectureSpec:
    return replace(
        architecture,
        head_conv_channels=None,
        head_channels=None,
        num_classes=None,
    )


def _normalize_feature_indices(
    feature_indices: Sequence[int],
    stage_count: int,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in feature_indices:
        resolved = int(index)
        if resolved < 0:
            resolved += stage_count
        if resolved < 0 or resolved >= stage_count:
            raise ValueError(
                f"detection feature index {index} is outside {stage_count} stages"
            )
        if resolved not in normalized:
            normalized.append(resolved)
    if not normalized:
        raise ValueError("at least one detection feature stage is required")
    return tuple(normalized)


def resolve_detection_feature_blocks(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    feature_stage_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
) -> tuple[ResolvedBlockSpec, ...]:
    backbone_arch = _backbone_only_architecture(architecture)
    backbone_space = _backbone_only_space(search_space)
    indices = _normalize_feature_indices(
        feature_stage_indices,
        len(backbone_arch.stages),
    )
    latest_by_stage: dict[int, ResolvedBlockSpec] = {}
    for block in backbone_space.resolve_blocks(backbone_arch):
        latest_by_stage[block.stage_index] = block
    missing = [index for index in indices if index not in latest_by_stage]
    if missing:
        raise ValueError(f"detection feature stages have no resolved blocks: {missing}")
    return tuple(latest_by_stage[index] for index in indices)


def detection_feature_shape_table(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    feature_stage_indices: Sequence[int] = DEFAULT_DETECTION_FEATURE_INDICES,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for block in resolve_detection_feature_blocks(
        architecture,
        search_space,
        feature_stage_indices,
    ):
        rows.append(
            {
                "stage_index": int(block.stage_index),
                "feature_h": int(block.output_resolution),
                "feature_w": int(block.output_resolution),
                "channels": int(block.out_channels),
                "stride": int(search_space.config.image_size // block.output_resolution),
            }
        )
    return rows


def _conv_layer(
    estimator: FPGACostEstimator,
    *,
    stage_index: int,
    block_index: int,
    op: str,
    resolution: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
) -> LayerCost:
    block = ResolvedBlockSpec(
        stage_index=stage_index,
        block_index=block_index,
        op="conv",
        kernel_size=int(kernel_size),
        expand_ratio=1,
        stride=1,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        input_resolution=int(resolution),
        output_resolution=int(resolution),
    )
    return replace(estimator._estimate_block(block), op=op)


def _upsample_add_layer(
    estimator: FPGACostEstimator,
    *,
    stage_index: int,
    block_index: int,
    resolution: int,
    channels: int,
) -> LayerCost:
    activation_bytes = estimator._tensor_bytes(resolution, channels)
    macs = int(resolution) * int(resolution) * int(channels)
    return LayerCost(
        stage_index=stage_index,
        block_index=block_index,
        op="detection_fpn_upsample_add",
        input_resolution=int(resolution),
        output_resolution=int(resolution),
        in_channels=int(channels),
        out_channels=int(channels),
        params=0,
        macs=macs,
        weight_bytes=0,
        activation_bytes=activation_bytes,
        ideal_dsp=0,
        allocated_dsp=0,
        bram_blocks=_div_up(2 * activation_bytes, BRAM_BLOCK_BYTES),
        lut=96 + int(channels) * 2,
        latency_cycles=max(1, _div_up(macs, max(1, estimator._pack_factor * 8))),
    )


def fixed_detection_head_layers(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    estimator: FPGACostEstimator,
    config: FixedMultiscaleDetectionCostConfig,
) -> tuple[LayerCost, ...]:
    feature_blocks = resolve_detection_feature_blocks(
        architecture,
        search_space,
        config.feature_stage_indices,
    )
    neck_channels = int(config.neck_channels)
    layers: list[LayerCost] = []
    stage_base = 100

    for level, block in enumerate(feature_blocks):
        layers.append(
            _conv_layer(
                estimator,
                stage_index=stage_base,
                block_index=level,
                op="detection_fpn_lateral_1x1",
                resolution=block.output_resolution,
                in_channels=block.out_channels,
                out_channels=neck_channels,
                kernel_size=1,
            )
        )

    for level, block in enumerate(feature_blocks[:-1]):
        layers.append(
            _upsample_add_layer(
                estimator,
                stage_index=stage_base + 1,
                block_index=level,
                resolution=block.output_resolution,
                channels=neck_channels,
            )
        )

    for level, block in enumerate(feature_blocks):
        layers.append(
            _conv_layer(
                estimator,
                stage_index=stage_base + 2,
                block_index=level,
                op="detection_fpn_output_3x3",
                resolution=block.output_resolution,
                in_channels=neck_channels,
                out_channels=neck_channels,
                kernel_size=3,
            )
        )

    for level, block in enumerate(feature_blocks):
        layers.append(
            _conv_layer(
                estimator,
                stage_index=stage_base + 3,
                block_index=level,
                op="detection_head_tower_3x3",
                resolution=block.output_resolution,
                in_channels=neck_channels,
                out_channels=neck_channels,
                kernel_size=3,
            )
        )
        layers.append(
            _conv_layer(
                estimator,
                stage_index=stage_base + 4,
                block_index=level,
                op="detection_head_pred_1x1",
                resolution=block.output_resolution,
                in_channels=neck_channels,
                out_channels=config.prediction_channels,
                kernel_size=1,
            )
        )

    return tuple(layers)


def estimate_multiscale_detection_cost(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    estimator: FPGACostEstimator,
    config: FixedMultiscaleDetectionCostConfig,
) -> CostEstimate:
    backbone_arch = _backbone_only_architecture(architecture)
    backbone_space = _backbone_only_space(search_space)
    backbone = estimator.estimate(backbone_arch, backbone_space)
    detection_layers = fixed_detection_head_layers(
        architecture,
        search_space,
        estimator,
        config,
    )
    per_layer = tuple(backbone.per_layer) + detection_layers

    total_params = sum(layer.params for layer in per_layer)
    total_macs = sum(layer.macs for layer in per_layer)
    peak_activation_bytes = max((layer.activation_bytes for layer in per_layer), default=0)
    peak_weight_bytes = max((layer.weight_bytes for layer in per_layer), default=0)
    peak_buffer_bytes = max(
        (layer.activation_bytes + layer.weight_bytes for layer in per_layer),
        default=0,
    )
    peak_dsp = max((layer.allocated_dsp for layer in per_layer), default=0)
    peak_bram = max((layer.bram_blocks for layer in per_layer), default=0)
    peak_lut = max((layer.lut for layer in per_layer), default=0)
    total_dsp = sum(layer.allocated_dsp for layer in per_layer)
    total_bram = sum(layer.bram_blocks for layer in per_layer)
    total_lut = sum(layer.lut for layer in per_layer)
    latency_cycles = sum(layer.latency_cycles for layer in per_layer)
    latency_ms = sum(
        layer.resolved_latency_ms(estimator.hardware_spec.clock_mhz)
        for layer in per_layer
    )
    model_size_mb = total_params * estimator._bytes_per_scalar / (1024**2)
    power_w = estimator._estimate_power(
        peak_dsp=peak_dsp,
        peak_bram=peak_bram,
        peak_lut=peak_lut,
    )
    energy_mj = power_w * latency_ms
    traffic_bytes = sum(
        layer.activation_bytes + layer.weight_bytes for layer in per_layer
    )
    memory_bandwidth_gbps = estimator._estimate_memory_bandwidth(
        total_memory_traffic_bytes=traffic_bytes,
        latency_ms=latency_ms,
    )
    offchip_mem_mb = estimator._estimate_offchip_memory(peak_buffer_bytes)
    physical_metrics = dict(backbone.physical_metrics)
    physical_metrics.update(
        {
            "detection_fixed_multiscale_feature_count": float(
                len(config.feature_stage_indices)
            ),
            "detection_fixed_neck_channels": float(config.neck_channels),
            "detection_fixed_prediction_channels": float(config.prediction_channels),
        }
    )
    violations = list(backbone.violations)
    violations.extend(
        estimator._check_constraints(
            latency_ms=latency_ms,
            energy_mj=energy_mj,
            model_size_mb=model_size_mb,
            resource_dsp=total_dsp,
            resource_bram=total_bram,
            resource_lut=total_lut,
            power_w=power_w,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            offchip_mem_mb=offchip_mem_mb,
        )
    )
    violations = list(dict.fromkeys(violations))

    return CostEstimate(
        params=total_params,
        macs=total_macs,
        model_size_mb=model_size_mb,
        peak_activation_bytes=peak_activation_bytes,
        peak_weight_bytes=peak_weight_bytes,
        peak_buffer_bytes=peak_buffer_bytes,
        peak_dsp=peak_dsp,
        peak_bram=peak_bram,
        peak_lut=peak_lut,
        total_dsp=total_dsp,
        total_bram=total_bram,
        total_lut=total_lut,
        latency_cycles=latency_cycles,
        latency_ms=latency_ms,
        power_w=power_w,
        energy_mj=energy_mj,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        offchip_mem_mb=offchip_mem_mb,
        violations=tuple(violations),
        per_layer=per_layer,
        physical_metrics=physical_metrics,
    )


def detection_cost_table_payload(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    estimator: FPGACostEstimator,
    config: FixedMultiscaleDetectionCostConfig,
) -> dict[str, Any]:
    estimate = estimate_multiscale_detection_cost(
        architecture,
        search_space,
        estimator,
        config,
    )
    detection_layers = [
        layer for layer in estimate.per_layer if layer.stage_index >= 100
    ]
    return {
        "artifact_type": "detection_multiscale_640_hardware_cost_table",
        "schema_version": 1,
        "input_image_size": int(search_space.config.image_size),
        "feature_stage_indices": list(config.feature_stage_indices),
        "feature_map_shapes": detection_feature_shape_table(
            architecture,
            search_space,
            config.feature_stage_indices,
        ),
        "fixed_neck_head": {
            "neck": "fixed_fpn",
            "head": "fixed_dense_detection_head",
            "neck_channels": int(config.neck_channels),
            "anchors_per_location": int(config.anchors_per_location),
            "prediction_channels": int(config.prediction_channels),
            "searchable": False,
        },
        "cost_scope": "analytic_detection_neck_head_supplement",
        "claim_boundary": {
            "backbone_search_space": "copied_from_classification_branch",
            "neck_head_search": "fixed_not_searched_in_stage_1",
            "hls_route_cost": "not_established_by_this_table",
            "board_detection_accuracy": "not_measured_by_this_table",
            "external_power": "not_measured_by_this_table",
        },
        "summary": {
            "params": estimate.params,
            "macs": estimate.macs,
            "latency_ms": estimate.latency_ms,
            "energy_mj": estimate.energy_mj,
            "lut": estimate.resource_lut,
            "bram": estimate.resource_bram,
            "dsp": estimate.resource_dsp,
            "power_w": estimate.power_w,
            "violations": list(estimate.violations),
        },
        "rows": [
            {
                "row_key": (
                    f"{layer.op}_p{index}_"
                    f"{layer.input_resolution}_{layer.in_channels}_{layer.out_channels}"
                ),
                "op": layer.op,
                "feature_h": layer.input_resolution,
                "feature_w": layer.input_resolution,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": 1 if layer.op.endswith("1x1") else 3,
                "params": layer.params,
                "macs": layer.macs,
                "latency_cycles": layer.latency_cycles,
                "latency_ms": layer.resolved_latency_ms(
                    estimator.hardware_spec.clock_mhz
                ),
                "lut": layer.lut,
                "bram": layer.bram_blocks,
                "dsp": layer.allocated_dsp,
            }
            for index, layer in enumerate(detection_layers)
        ],
    }
