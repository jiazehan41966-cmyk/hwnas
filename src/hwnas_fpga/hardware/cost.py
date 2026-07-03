from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Optional, Union

from hwnas_fpga.interfaces import CandidateMetrics, HardwareSpec, SearchConstraints
from hwnas_fpga.search_space import ArchitectureSpec, ResolvedBlockSpec, SearchSpace
from hwnas_fpga.hardware.lookup_table import LutQueryEngine, OpSpec


BRAM_BLOCK_BYTES = 36 * 1024 // 8


def _div_up(value: Union[int, float], divisor: Union[int, float]) -> int:
    return int(ceil(value / divisor))


@dataclass(frozen=True)
class LayerCost:
    stage_index: int
    block_index: int
    op: str
    input_resolution: int
    output_resolution: int
    in_channels: int
    out_channels: int
    params: int
    macs: int
    weight_bytes: int
    activation_bytes: int
    ideal_dsp: int
    allocated_dsp: int
    bram_blocks: int
    lut: int
    latency_cycles: int
    latency_ms_override: Optional[float] = None
    effective_clock_mhz: Optional[float] = None

    def resolved_latency_ms(self, default_clock_mhz: float) -> float:
        if self.latency_ms_override is not None:
            return float(self.latency_ms_override)
        return float(self.latency_cycles) / (float(default_clock_mhz) * 1_000.0)


@dataclass(frozen=True)
class CostEstimate:
    params: int
    macs: int
    model_size_mb: float
    peak_activation_bytes: int
    peak_weight_bytes: int
    peak_buffer_bytes: int
    peak_dsp: int
    peak_bram: int
    peak_lut: int
    total_dsp: int
    total_bram: int
    total_lut: int
    latency_cycles: int
    latency_ms: float
    power_w: float
    energy_mj: float
    memory_bandwidth_gbps: float
    offchip_mem_mb: float
    violations: tuple[str, ...]
    per_layer: tuple[LayerCost, ...]
    physical_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def resource_dsp(self) -> int:
        return self.total_dsp

    @property
    def resource_bram(self) -> int:
        return self.total_bram

    @property
    def resource_lut(self) -> int:
        return self.total_lut

    def to_candidate_metrics(self) -> CandidateMetrics:
        return CandidateMetrics(
            latency_ms=self.latency_ms,
            energy_mj=self.energy_mj,
            lut=self.resource_lut,
            bram=self.resource_bram,
            dsp=self.resource_dsp,
            power_w=self.power_w,
            memory_bandwidth_gbps=self.memory_bandwidth_gbps,
            offchip_mem_mb=self.offchip_mem_mb,
            early_expand_pressure=self.physical_metrics.get("early_expand_pressure"),
            interconnect_pressure=self.physical_metrics.get("interconnect_pressure"),
            memory_pressure=self.physical_metrics.get("memory_pressure"),
            fanout_pressure=self.physical_metrics.get("fanout_pressure"),
            stream_width=self.physical_metrics.get("stream_width"),
            physical_risk=self.physical_metrics.get("physical_risk"),
        )


class FPGACostEstimator:
    def __init__(
        self,
        hardware_spec: HardwareSpec,
        constraints: Optional[SearchConstraints] = None,
        *,
        quantization_bits: int = 8,
        pipeline_efficiency: float = 0.7,
        default_dsp_budget: int = 256,
        lut_query_engine: Optional[LutQueryEngine] = None,
        operator_policies: Optional[dict[str, dict[str, Any]]] = None,
        require_deployable_operators: bool = False,
        strict_formal_lut: bool = False,
        strict_lut_label: str = "formal LUT",
        use_lut_for_head_layers: bool = False,
    ) -> None:
        if quantization_bits <= 0:
            raise ValueError("quantization_bits must be positive")
        if not 0 < pipeline_efficiency <= 1:
            raise ValueError("pipeline_efficiency must be in (0, 1]")
        self.hardware_spec = hardware_spec
        self.constraints = constraints
        self.quantization_bits = quantization_bits
        self.pipeline_efficiency = pipeline_efficiency
        self.default_dsp_budget = default_dsp_budget
        self.lut_query_engine = lut_query_engine
        self.operator_policies = {
            str(op).strip(): dict(policy)
            for op, policy in (operator_policies or {}).items()
        }
        self.require_deployable_operators = bool(require_deployable_operators)
        self.strict_formal_lut = bool(strict_formal_lut)
        self.strict_lut_label = str(strict_lut_label or "formal LUT").strip() or "formal LUT"
        self.use_lut_for_head_layers = bool(use_lut_for_head_layers)
        self.lut_hits = 0  # LUT hit counter
        self.lut_misses = 0  # LUT miss counter
        self.true_lut_misses = 0
        self.deferred_hits = 0
        self.deferred_hit_records: list[dict[str, Any]] = []
        self.true_miss_records: list[dict[str, Any]] = []
        self._estimate_dynamic_violations: list[str] = []
    def estimate(
        self,
        architecture: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> CostEstimate:
        self._estimate_dynamic_violations = []
        resolved_blocks = search_space.resolve_blocks(architecture)
        stem_layer = self._estimate_stem(architecture, search_space)
        stem_pool_layer = self._estimate_stem_pool(architecture, search_space)
        block_layers = tuple(self._estimate_block(block) for block in resolved_blocks)
        head_layers = self._estimate_head_layers(architecture, resolved_blocks, search_space)
        per_layer = (
            (stem_layer,)
            + ((stem_pool_layer,) if stem_pool_layer is not None else ())
            + block_layers
            + head_layers
        )

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
            layer.resolved_latency_ms(self.hardware_spec.clock_mhz)
            for layer in per_layer
        )
        model_size_mb = total_params * self._bytes_per_scalar / (1024**2)
        power_w = self._estimate_power(
            peak_dsp=peak_dsp,
            peak_bram=peak_bram,
            peak_lut=peak_lut,
        )
        energy_mj = power_w * latency_ms
        total_memory_traffic_bytes = sum(
            layer.activation_bytes + layer.weight_bytes for layer in per_layer
        )
        memory_bandwidth_gbps = self._estimate_memory_bandwidth(
            total_memory_traffic_bytes=total_memory_traffic_bytes,
            latency_ms=latency_ms,
        )
        offchip_mem_mb = self._estimate_offchip_memory(peak_buffer_bytes)
        physical_metrics = self._compute_physical_metrics(
            resolved_blocks=resolved_blocks,
            peak_buffer_bytes=peak_buffer_bytes,
        )
        violations = list(
            self._check_constraints(
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
        violations.extend(
            self._check_physical_constraints(
                resource_dsp=total_dsp,
                resource_bram=total_bram,
                resource_lut=total_lut,
                physical_metrics=physical_metrics,
                resolved_blocks=resolved_blocks,
            )
        )
        violations.extend(self._check_operator_policy(resolved_blocks))
        violations.extend(self._estimate_dynamic_violations)
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

    @property
    def _bytes_per_scalar(self) -> int:
        return max(1, self.quantization_bits // 8)

    @property
    def _dsp_budget(self) -> int:
        return self.hardware_spec.max_dsp or self.default_dsp_budget

    @property
    def _pack_factor(self) -> int:
        return 2 if self.quantization_bits <= 8 else 1

    def _estimate_stem(
        self,
        architecture: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> LayerCost:
        input_resolution = search_space.config.image_size
        output_resolution = max(1, _div_up(input_resolution, architecture.stem_stride))
        stem_block = ResolvedBlockSpec(
            stage_index=-1,
            block_index=-1,
            op="conv",
            kernel_size=3,
            expand_ratio=1,
            stride=architecture.stem_stride,
            in_channels=architecture.input_channels,
            out_channels=architecture.stem_channels,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
        )
        base_cost = self._estimate_block(stem_block)
        return LayerCost(
            stage_index=-1,
            block_index=-1,
            op="stem_conv",
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            in_channels=architecture.input_channels,
            out_channels=architecture.stem_channels,
            params=base_cost.params,
            macs=base_cost.macs,
            weight_bytes=base_cost.weight_bytes,
            activation_bytes=base_cost.activation_bytes,
            ideal_dsp=base_cost.ideal_dsp,
            allocated_dsp=base_cost.allocated_dsp,
            bram_blocks=base_cost.bram_blocks,
            lut=base_cost.lut,
            latency_cycles=base_cost.latency_cycles,
        )

    def _estimate_stem_pool(
        self,
        architecture: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> Optional[LayerCost]:
        if architecture.post_stem_downsample_stride <= 1:
            return None

        input_resolution = max(1, _div_up(search_space.config.image_size, architecture.stem_stride))
        output_resolution = max(
            1,
            _div_up(input_resolution, architecture.post_stem_downsample_stride),
        )
        input_bytes = self._tensor_bytes(input_resolution, architecture.stem_channels)
        output_bytes = self._tensor_bytes(output_resolution, architecture.stem_channels)
        bram_blocks = _div_up(input_bytes + output_bytes, BRAM_BLOCK_BYTES)
        lut = 48 + architecture.stem_channels * 2 + output_resolution
        latency_cycles = max(
            1,
            _div_up(
                output_resolution * output_resolution * architecture.stem_channels,
                max(1, self._pack_factor * 8),
            ),
        )

        return LayerCost(
            stage_index=-1,
            block_index=-2,
            op="stem_pool",
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            in_channels=architecture.stem_channels,
            out_channels=architecture.stem_channels,
            params=0,
            macs=0,
            weight_bytes=0,
            activation_bytes=output_bytes,
            ideal_dsp=0,
            allocated_dsp=0,
            bram_blocks=bram_blocks,
            lut=lut,
            latency_cycles=latency_cycles,
        )

    def _record_lut_non_hit(
        self,
        *,
        logical_op: str,
        op_spec: OpSpec,
        query_result: Any,
        shape: dict[str, Any],
    ) -> None:
        if (
            self.strict_formal_lut
            and query_result.status not in {"measured", "missing"}
        ):
            self.deferred_hits += 1
            status_entry = query_result.status_entry
            record = {
                "op": logical_op,
                "lookup_op": op_spec.op,
                "shape": shape,
                "status": query_result.status,
                "defer_reason": (
                    None if status_entry is None else status_entry.defer_reason
                ),
                "case_name": (
                    None if status_entry is None else status_entry.case_name
                ),
            }
            self.deferred_hit_records.append(record)
            violation_reason = (
                status_entry.defer_reason
                if status_entry is not None and status_entry.defer_reason
                else query_result.status
            )
            self._estimate_dynamic_violations.append(
                f"{self.strict_lut_label} infeasible: {op_spec.op} ({violation_reason})"
            )
            return

        self.lut_misses += 1
        self.true_lut_misses += 1
        if self.strict_formal_lut and query_result.status == "missing":
            self._estimate_dynamic_violations.append(
                f"{self.strict_lut_label} missing: {op_spec.op}"
            )
        self.true_miss_records.append(
            {
                "op": logical_op,
                "lookup_op": op_spec.op,
                "shape": shape,
                "status": query_result.status,
            }
        )

    def _query_lut_entry(
        self,
        *,
        logical_op: str,
        op_spec: OpSpec,
        shape: dict[str, Any],
    ) -> Any | None:
        if not self.lut_query_engine:
            return None

        query_result = self.lut_query_engine.query_with_status(op_spec)
        if query_result.entry:
            self.lut_hits += 1
            return query_result.entry

        self._record_lut_non_hit(
            logical_op=logical_op,
            op_spec=op_spec,
            query_result=query_result,
            shape=shape,
        )
        return None

    def _block_shape_record(self, block: ResolvedBlockSpec) -> dict[str, Any]:
        return {
            "input_resolution": block.input_resolution,
            "output_resolution": block.output_resolution,
            "in_channels": block.in_channels,
            "out_channels": block.out_channels,
            "kernel_size": block.kernel_size,
            "expand_ratio": block.expand_ratio,
            "stride": block.stride,
        }

    def _head_gap_to_op_spec(self, *, final_resolution: int, final_channels: int) -> OpSpec:
        return OpSpec(
            op="global_avg_pool",
            kernel_size=1,
            in_channels=final_channels,
            out_channels=final_channels,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(final_resolution, final_resolution),
        )

    def _head_fc_to_op_spec(self, *, final_channels: int, output_channels: int) -> OpSpec:
        return OpSpec(
            op="fc_layer",
            kernel_size=1,
            in_channels=final_channels,
            out_channels=output_channels,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(1, 1),
        )

    def _estimate_head_layers(
        self,
        architecture: ArchitectureSpec,
        resolved_blocks: tuple[ResolvedBlockSpec, ...],
        search_space: SearchSpace,
    ) -> tuple[LayerCost, ...]:
        if architecture.num_classes is None or architecture.num_classes <= 0:
            return ()

        if resolved_blocks:
            final_channels = resolved_blocks[-1].out_channels
            final_resolution = resolved_blocks[-1].output_resolution
        else:
            final_channels = architecture.stem_channels
            final_resolution = max(1, _div_up(search_space.config.image_size, architecture.stem_stride))
            final_resolution = max(
                1,
                _div_up(final_resolution, architecture.post_stem_downsample_stride),
            )

        head_layers: list[LayerCost] = []

        if architecture.head_conv_channels is not None and architecture.head_conv_channels > 0:
            conv_block = ResolvedBlockSpec(
                stage_index=len(architecture.stages),
                block_index=-2,
                op="conv",
                kernel_size=1,
                expand_ratio=1,
                stride=1,
                in_channels=final_channels,
                out_channels=architecture.head_conv_channels,
                input_resolution=final_resolution,
                output_resolution=final_resolution,
            )
            conv_cost = self._estimate_block(conv_block)
            head_layers.append(
                LayerCost(
                    stage_index=len(architecture.stages),
                    block_index=-2,
                    op="head_conv1x1",
                    input_resolution=final_resolution,
                    output_resolution=final_resolution,
                    in_channels=final_channels,
                    out_channels=architecture.head_conv_channels,
                    params=conv_cost.params,
                    macs=conv_cost.macs,
                    weight_bytes=conv_cost.weight_bytes,
                    activation_bytes=conv_cost.activation_bytes,
                    ideal_dsp=conv_cost.ideal_dsp,
                    allocated_dsp=conv_cost.allocated_dsp,
                    bram_blocks=conv_cost.bram_blocks,
                    lut=conv_cost.lut,
                    latency_cycles=conv_cost.latency_cycles,
                )
            )
            final_channels = architecture.head_conv_channels

        fc_input_resolution = final_resolution
        if self.use_lut_for_head_layers:
            gap_spec = self._head_gap_to_op_spec(
                final_resolution=final_resolution,
                final_channels=final_channels,
            )
            gap_shape = {
                "input_resolution": final_resolution,
                "output_resolution": 1,
                "in_channels": final_channels,
                "out_channels": final_channels,
                "kernel_size": 1,
                "expand_ratio": 1,
                "stride": 1,
            }
            gap_entry = self._query_lut_entry(
                logical_op="head_gap",
                op_spec=gap_spec,
                shape=gap_shape,
            )
            gap_macs = final_resolution * final_resolution * final_channels
            pooled_bytes = self._tensor_bytes(1, final_channels)
            if gap_entry is not None:
                head_layers.append(
                    LayerCost(
                        stage_index=len(architecture.stages),
                        block_index=-3,
                        op="head_gap",
                        input_resolution=final_resolution,
                        output_resolution=1,
                        in_channels=final_channels,
                        out_channels=final_channels,
                        params=0,
                        macs=gap_macs,
                        weight_bytes=0,
                        activation_bytes=pooled_bytes,
                        ideal_dsp=gap_entry.dsp,
                        allocated_dsp=gap_entry.dsp,
                        bram_blocks=gap_entry.bram,
                        lut=gap_entry.lut,
                        latency_cycles=gap_entry.cycles,
                        latency_ms_override=self._latency_override_ms(
                            "global_avg_pool",
                            cycles=gap_entry.cycles,
                            fallback_latency_ms=gap_entry.latency_ms,
                        ),
                        effective_clock_mhz=self._effective_clock_mhz("global_avg_pool"),
                    )
                )
            else:
                input_bytes = self._tensor_bytes(final_resolution, final_channels)
                latency_cycles = max(
                    1,
                    _div_up(gap_macs, max(1, self._pack_factor * 8)),
                )
                head_layers.append(
                    LayerCost(
                        stage_index=len(architecture.stages),
                        block_index=-3,
                        op="head_gap",
                        input_resolution=final_resolution,
                        output_resolution=1,
                        in_channels=final_channels,
                        out_channels=final_channels,
                        params=0,
                        macs=gap_macs,
                        weight_bytes=0,
                        activation_bytes=pooled_bytes,
                        ideal_dsp=0,
                        allocated_dsp=0,
                        bram_blocks=_div_up(input_bytes + pooled_bytes, BRAM_BLOCK_BYTES),
                        lut=64 + final_channels * 2,
                        latency_cycles=latency_cycles,
                    )
                )
            fc_input_resolution = 1

        hidden_channels = architecture.head_channels if architecture.head_channels and architecture.head_channels > 0 else None
        if hidden_channels is not None:
            params = final_channels * hidden_channels + hidden_channels * architecture.num_classes
            macs = params
            output_channels = architecture.num_classes
            lut_bias = 96
            dsp_raw = max(
                self._linear_dsp(final_channels, hidden_channels),
                self._linear_dsp(hidden_channels, architecture.num_classes),
            )
        else:
            params = final_channels * architecture.num_classes
            macs = params
            output_channels = architecture.num_classes
            lut_bias = 64
            dsp_raw = self._linear_dsp(final_channels, architecture.num_classes)

        weight_bytes = params * self._bytes_per_scalar
        activation_bytes = output_channels * self._bytes_per_scalar
        pooled_bytes = self._tensor_bytes(1, final_channels)

        fc_entry = None
        if self.use_lut_for_head_layers and hidden_channels is None:
            fc_spec = self._head_fc_to_op_spec(
                final_channels=final_channels,
                output_channels=output_channels,
            )
            fc_entry = self._query_lut_entry(
                logical_op="head_fc",
                op_spec=fc_spec,
                shape={
                    "input_resolution": 1,
                    "output_resolution": 1,
                    "in_channels": final_channels,
                    "out_channels": output_channels,
                    "kernel_size": 1,
                    "expand_ratio": 1,
                    "stride": 1,
                },
            )

        if fc_entry is not None:
            allocated_dsp = fc_entry.dsp
            latency_cycles = fc_entry.cycles
            bram_blocks = fc_entry.bram
            lut = fc_entry.lut
            latency_ms_override = self._latency_override_ms(
                "fc_layer",
                cycles=fc_entry.cycles,
                fallback_latency_ms=fc_entry.latency_ms,
            )
            effective_clock_mhz = self._effective_clock_mhz("fc_layer")
        else:
            allocated_dsp = min(max(1, dsp_raw), self._dsp_budget)
            throughput = max(1.0, allocated_dsp * self._pack_factor * self.pipeline_efficiency)
            latency_cycles = max(1, _div_up(macs, throughput))
            bram_blocks = _div_up(pooled_bytes + activation_bytes + weight_bytes, BRAM_BLOCK_BYTES)
            lut = lut_bias + allocated_dsp * 10 + output_channels * 4
            latency_ms_override = None
            effective_clock_mhz = None

        head_layers.append(
            LayerCost(
                stage_index=len(architecture.stages),
                block_index=-1,
                op="head_fc",
                input_resolution=fc_input_resolution,
                output_resolution=1,
                in_channels=final_channels,
                out_channels=output_channels,
                params=params,
                macs=macs,
                weight_bytes=weight_bytes,
                activation_bytes=activation_bytes,
                ideal_dsp=allocated_dsp,
                allocated_dsp=allocated_dsp,
                bram_blocks=bram_blocks,
                lut=lut,
                latency_cycles=latency_cycles,
                latency_ms_override=latency_ms_override,
                effective_clock_mhz=effective_clock_mhz,
            )
        )
        return tuple(head_layers)

    def _estimate_block(self, block: ResolvedBlockSpec) -> LayerCost:
        latency_clock_mhz = self._effective_clock_mhz(block.op)
        latency_ms_override: Optional[float] = None
        # 1. 灏濊瘯 LUT 鏌ヨ锛堝鏋滃彲鐢級
        if self.lut_query_engine:
            op_spec = self._block_to_op_spec(block)
            shape_record = self._block_shape_record(block)
            query_result = self.lut_query_engine.query_with_status(op_spec)
            if query_result.entry:
                lut_entry = query_result.entry
                self.lut_hits += 1
                latency_ms_override = self._latency_override_ms(
                    block.op,
                    cycles=lut_entry.cycles,
                    fallback_latency_ms=lut_entry.latency_ms,
                )
                # 璁＄畻鍏朵粬鍒嗘瀽鎸囨爣锛坧arams, macs 绛夛級
                params, macs = self._block_params_macs(block)
                weight_bytes = params * self._bytes_per_scalar
                activation_bytes = self._tensor_bytes(block.output_resolution, block.out_channels)

                return LayerCost(
                    stage_index=block.stage_index,
                    block_index=block.block_index,
                    op=block.op,
                    input_resolution=block.input_resolution,
                    output_resolution=block.output_resolution,
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    params=params,
                    macs=macs,
                    weight_bytes=weight_bytes,
                    activation_bytes=activation_bytes,
                    ideal_dsp=lut_entry.dsp,
                    allocated_dsp=lut_entry.dsp,
                    bram_blocks=lut_entry.bram,
                    lut=lut_entry.lut,
                    latency_cycles=lut_entry.cycles,
                    latency_ms_override=latency_ms_override,
                    effective_clock_mhz=latency_clock_mhz,
                )
            if (
                self.strict_formal_lut
                and query_result.status not in {"measured", "missing"}
            ):
                self.deferred_hits += 1
                status_entry = query_result.status_entry
                record = {
                    "op": block.op,
                    "lookup_op": op_spec.op,
                    "shape": shape_record,
                    "status": query_result.status,
                    "defer_reason": (
                        None if status_entry is None else status_entry.defer_reason
                    ),
                    "case_name": (
                        None if status_entry is None else status_entry.case_name
                    ),
                }
                self.deferred_hit_records.append(record)
                violation_reason = (
                    status_entry.defer_reason
                    if status_entry is not None and status_entry.defer_reason
                    else query_result.status
                )
                self._estimate_dynamic_violations.append(
                    f"{self.strict_lut_label} infeasible: {op_spec.op} ({violation_reason})"
                )
            else:
                self.lut_misses += 1
                self.true_lut_misses += 1
                if self.strict_formal_lut and query_result.status == "missing":
                    self._estimate_dynamic_violations.append(
                        f"{self.strict_lut_label} missing: {op_spec.op}"
                    )
                self.true_miss_records.append(
                    {
                        "op": block.op,
                        "lookup_op": op_spec.op,
                        "shape": shape_record,
                        "status": query_result.status,
                    }
                )

        # 2. 鍥為€€鍒板垎鏋愭ā鍨?
        if block.op == "skip":
            input_bytes = self._tensor_bytes(block.input_resolution, block.in_channels)
            output_bytes = self._tensor_bytes(block.output_resolution, block.out_channels)
            bram_blocks = _div_up(input_bytes + output_bytes, BRAM_BLOCK_BYTES)
            return LayerCost(
                stage_index=block.stage_index,
                block_index=block.block_index,
                op=block.op,
                input_resolution=block.input_resolution,
                output_resolution=block.output_resolution,
                in_channels=block.in_channels,
                out_channels=block.out_channels,
                params=0,
                macs=0,
                weight_bytes=0,
                activation_bytes=output_bytes,
                ideal_dsp=0,
                allocated_dsp=0,
                bram_blocks=bram_blocks,
                lut=32,
                latency_cycles=1,
                latency_ms_override=self._latency_override_ms(block.op, cycles=1),
                effective_clock_mhz=latency_clock_mhz,
            )

        params, macs, raw_dsp = self._block_complexity(block)
        ideal_dsp = min(max(1, raw_dsp), self._dsp_budget)
        allocated_dsp = min(ideal_dsp, self._dsp_budget)
        throughput = max(1.0, allocated_dsp * self._pack_factor * self.pipeline_efficiency)
        latency_cycles = max(1, _div_up(macs, throughput))
        weight_bytes = params * self._bytes_per_scalar
        activation_bytes = self._tensor_bytes(block.output_resolution, block.out_channels)
        input_bytes = self._tensor_bytes(block.input_resolution, block.in_channels)
        tile_weight_bytes = min(weight_bytes, 8 * BRAM_BLOCK_BYTES)
        bram_blocks = _div_up(input_bytes + activation_bytes + tile_weight_bytes, BRAM_BLOCK_BYTES)
        lut = self._estimate_lut(block=block, allocated_dsp=allocated_dsp)

        return LayerCost(
            stage_index=block.stage_index,
            block_index=block.block_index,
            op=block.op,
            input_resolution=block.input_resolution,
            output_resolution=block.output_resolution,
            in_channels=block.in_channels,
            out_channels=block.out_channels,
            params=params,
            macs=macs,
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            ideal_dsp=ideal_dsp,
            allocated_dsp=allocated_dsp,
            bram_blocks=bram_blocks,
            lut=lut,
            latency_cycles=latency_cycles,
            latency_ms_override=self._latency_override_ms(
                block.op,
                cycles=latency_cycles,
            ),
            effective_clock_mhz=latency_clock_mhz,
        )

    def _operator_policy_for(self, op: str) -> dict[str, Any]:
        return dict(self.operator_policies.get(str(op).strip(), {}))

    def _effective_clock_mhz(self, op: str) -> Optional[float]:
        policy = self._operator_policy_for(op)
        cost_policy = policy.get("hardware_cost_policy", {})
        if not isinstance(cost_policy, dict):
            return None
        if cost_policy.get("latency_rule") != "use_actual_post_route_fmax":
            return None
        fmax = policy.get("achieved_post_route_fmax_mhz")
        if fmax is None:
            return None
        return float(fmax)

    def _latency_override_ms(
        self,
        op: str,
        *,
        cycles: int,
        fallback_latency_ms: Optional[float] = None,
    ) -> Optional[float]:
        effective_clock_mhz = self._effective_clock_mhz(op)
        if effective_clock_mhz is not None and cycles > 0:
            return float(cycles) / (effective_clock_mhz * 1_000.0)
        if fallback_latency_ms is not None:
            return float(fallback_latency_ms)
        return None

    def _check_operator_policy(
        self,
        resolved_blocks: tuple[ResolvedBlockSpec, ...],
    ) -> list[str]:
        violations: list[str] = []
        for block in resolved_blocks:
            policy = self._operator_policy_for(block.op)
            if not policy:
                continue

            deployment_status = str(policy.get("deployment_status", "")).strip()
            if deployment_status == "paused":
                violations.append(f"operator {block.op} is paused in current target policy")
                continue
            if deployment_status == "dropped_current_target":
                violations.append(f"operator {block.op} is dropped for current target")
                continue

            if not self.require_deployable_operators:
                continue

            cost_policy = policy.get("hardware_cost_policy", {})
            if not isinstance(cost_policy, dict):
                continue
            if deployment_status == "conditional" and not bool(
                cost_policy.get("deployable_at_200mhz", True)
            ):
                violations.append(
                    f"operator {block.op} is not deployable at 200MHz under current policy"
                )
        return violations

    def _block_complexity(self, block: ResolvedBlockSpec) -> tuple[int, int, int]:
        if block.op == "conv":
            params = self._conv_params(
                in_channels=block.in_channels,
                out_channels=block.out_channels,
                kernel_size=block.kernel_size,
            )
            macs = self._conv_macs(
                resolution=block.output_resolution,
                in_channels=block.in_channels,
                out_channels=block.out_channels,
                kernel_size=block.kernel_size,
            )
            raw_dsp = self._conv_dsp(
                in_channels=block.in_channels,
                out_channels=block.out_channels,
                kernel_size=block.kernel_size,
                depthwise=False,
            )
            return params, macs, raw_dsp

        if block.op == "dw_pw_conv":
            dw_params = block.in_channels * block.kernel_size * block.kernel_size
            pw_params = block.in_channels * block.out_channels
            dw_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * block.kernel_size
                * block.kernel_size
            )
            pw_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * block.out_channels
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.in_channels,
                    kernel_size=block.kernel_size,
                    depthwise=True,
                ),
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return dw_params + pw_params, dw_macs + pw_macs, raw_dsp

        hidden_channels = block.in_channels * block.expand_ratio
        if block.op == "mbconv":
            expand_params = block.in_channels * hidden_channels
            dw_params = hidden_channels * block.kernel_size * block.kernel_size
            project_params = hidden_channels * block.out_channels
            expand_macs = (
                block.input_resolution
                * block.input_resolution
                * block.in_channels
                * hidden_channels
            )
            dw_macs = (
                block.output_resolution
                * block.output_resolution
                * hidden_channels
                * block.kernel_size
                * block.kernel_size
            )
            project_macs = (
                block.output_resolution
                * block.output_resolution
                * hidden_channels
                * block.out_channels
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
                self._conv_dsp(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=block.kernel_size,
                    depthwise=True,
                ),
                self._conv_dsp(
                    in_channels=hidden_channels,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return (
                expand_params + dw_params + project_params,
                expand_macs + dw_macs + project_macs,
                raw_dsp,
            )

        if block.op == "fused_mbconv":
            fused_params = (
                block.kernel_size * block.kernel_size * block.in_channels * hidden_channels
            )
            project_params = hidden_channels * block.out_channels
            fused_macs = (
                block.output_resolution
                * block.output_resolution
                * block.kernel_size
                * block.kernel_size
                * block.in_channels
                * hidden_channels
            )
            project_macs = (
                block.output_resolution
                * block.output_resolution
                * hidden_channels
                * block.out_channels
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=hidden_channels,
                    kernel_size=block.kernel_size,
                    depthwise=False,
                ),
                self._conv_dsp(
                    in_channels=hidden_channels,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return fused_params + project_params, fused_macs + project_macs, raw_dsp

        # 澹板憪涓撶敤澶氬昂搴﹀嵎绉?(MixConv)
        if block.op == "mixconv":
            # MixConv 浣跨敤3,5,7涓夌kernel_size骞惰
            # 鎬诲弬鏁?= sum(姣忎釜kernel鐨刣w鍙傛暟) + pw鍙傛暟
            dw_params = 0
            dw_macs = 0
            kernel_sizes = (3, 5, 7)
            channels_per_kernel = block.in_channels // 3
            for kernel in kernel_sizes:
                dw_params += channels_per_kernel * kernel * kernel
                dw_macs += (
                    block.output_resolution
                    * block.output_resolution
                    * channels_per_kernel
                    * kernel
                    * kernel
                )
            pw_params = block.in_channels * block.out_channels
            pw_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * block.out_channels
            )
            # DSP浼拌锛?涓苟琛孌W + 1涓狿W
            raw_dsp = max(
                max(
                    self._conv_dsp(
                        in_channels=channels_per_kernel,
                        out_channels=channels_per_kernel,
                        kernel_size=kernel,
                        depthwise=True,
                    )
                    for kernel in kernel_sizes
                ),
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return dw_params + pw_params, dw_macs + pw_macs, raw_dsp

        # 澹板憪涓撶敤鍘诲櫔鍧?(DenoiseBlock)
        if block.op == "denoise":
            # DW + PW (绫讳技dw_pw_conv浣嗗甫骞虫粦)
            dw_params = block.in_channels * block.kernel_size * block.kernel_size
            pw_params = block.in_channels * block.out_channels
            dw_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * block.kernel_size
                * block.kernel_size
            )
            pw_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * block.out_channels
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.in_channels,
                    kernel_size=block.kernel_size,
                    depthwise=True,
                ),
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return dw_params + pw_params, dw_macs + pw_macs, raw_dsp

        # 澹板憪涓撶敤杈圭紭鎰熺煡鍧?(EdgeAwareBlock)
        if block.op == "edge":
            # 4涓柟鍚戠殑杈圭紭妫€娴?+ 铻嶅悎PW
            edge_params = block.in_channels * block.kernel_size * block.kernel_size * 4
            fusion_params = block.in_channels * 4 * block.out_channels
            edge_macs = (
                block.output_resolution
                * block.output_resolution
                * edge_params
            )
            fusion_macs = (
                block.output_resolution
                * block.output_resolution
                * block.in_channels
                * 4
                * block.out_channels
            )
            raw_dsp = max(
                self._conv_dsp(
                    in_channels=block.in_channels,
                    out_channels=block.in_channels,
                    kernel_size=block.kernel_size,
                    depthwise=True,
                ),
                self._conv_dsp(
                    in_channels=block.in_channels * 4,
                    out_channels=block.out_channels,
                    kernel_size=1,
                    depthwise=False,
                ),
            )
            return edge_params + fusion_params, edge_macs + fusion_macs, raw_dsp

        raise ValueError(f"unsupported op: {block.op}")

    def _conv_params(self, *, in_channels: int, out_channels: int, kernel_size: int) -> int:
        return in_channels * out_channels * kernel_size * kernel_size

    def _conv_macs(
        self,
        *,
        resolution: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> int:
        return (
            resolution * resolution * in_channels * out_channels * kernel_size * kernel_size
        )

    def _conv_dsp(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depthwise: bool,
    ) -> int:
        channel_parallelism = 1 if depthwise else max(1, min(in_channels, 32) // 4)
        output_parallelism = max(1, min(out_channels, 64))
        kernel_factor = 2 if kernel_size >= 5 else 1
        return _div_up(output_parallelism * channel_parallelism * kernel_factor, self._pack_factor)

    def _estimate_lut(self, *, block: ResolvedBlockSpec, allocated_dsp: int) -> int:
        kernel_factor = 2 if block.kernel_size >= 5 else 1
        op_bias = {
            "conv": 120,
            "dw_pw_conv": 140,
            "mbconv": 180,
            "fused_mbconv": 170,
            "skip": 32,
            "mixconv": 200,
            "denoise": 150,
            "edge": 180,
        }.get(block.op, 120)
        return (
            op_bias
            + allocated_dsp * 18
            + block.out_channels * kernel_factor * 4
            + block.output_resolution * 2
        )

    def _estimate_power(self, *, peak_dsp: int, peak_bram: int, peak_lut: int) -> float:
        return 2.5 + peak_dsp * 0.012 + peak_bram * 0.018 + peak_lut * 0.0002

    def _estimate_memory_bandwidth(
        self,
        *,
        total_memory_traffic_bytes: int,
        latency_ms: float,
    ) -> float:
        if latency_ms <= 0:
            return 0.0
        return total_memory_traffic_bytes / (latency_ms / 1000.0) / 1_000_000_000.0

    def _estimate_offchip_memory(self, peak_buffer_bytes: int) -> float:
        if self.hardware_spec.max_bram is None:
            return peak_buffer_bytes / (1024**2)
        onchip_bytes = self.hardware_spec.max_bram * BRAM_BLOCK_BYTES
        spill_bytes = max(0, peak_buffer_bytes - onchip_bytes)
        return spill_bytes / (1024**2)

    def _tensor_bytes(self, resolution: int, channels: int) -> int:
        return resolution * resolution * channels * self._bytes_per_scalar

    def _linear_dsp(self, in_features: int, out_features: int) -> int:
        return _div_up(max(1, min(in_features, 256)) * max(1, min(out_features, 64)), 256 * self._pack_factor)

    def _block_to_op_spec(self, block: ResolvedBlockSpec) -> OpSpec:
        """灏?ResolvedBlockSpec 杞崲涓?OpSpec锛岀敤浜?LUT 鏌ヨ"""
        # 查询侧继续使用 NAS/search-space 的 block 名称。
        # HLS profiling manifest 里更细粒度的 kernel 名称
        # （例如 conv_bn_relu6 / inverted_residual）会在 OpSpec 导入阶段
        # 归一化到这里的查询口径。
        op_mapping = {
            "conv": "conv",
            "dw_pw_conv": "dw_pw_conv",
            "mbconv": f"mbconv_e{block.expand_ratio}_k{block.kernel_size}",
            "fused_mbconv": "fused_mbconv",
            "skip": "skip",
            "mixconv": "mixconv",
            "denoise": "denoise",
            "edge": "edge",
        }

        # 纭畾鍒嗙粍鏁?
        groups = 1
        if block.op in {"dw_pw_conv", "mixconv", "denoise", "edge"}:
            groups = block.in_channels  # depthwise

        return OpSpec(
            op=op_mapping.get(block.op, block.op),
            kernel_size=block.kernel_size,
            in_channels=block.in_channels,
            out_channels=block.out_channels,
            stride=block.stride,
            groups=groups,
            expand_ratio=block.expand_ratio,
            input_resolution=(block.input_resolution, block.input_resolution),
        )

    def _block_params_macs(self, block: ResolvedBlockSpec) -> tuple[int, int]:
        """璁＄畻 block 鐨勫弬鏁伴噺鍜?MACs"""
        if block.op == "skip":
            return 0, 0

        # 瀵逛簬澹板憪涓撶敤绠楀瓙锛岀洿鎺ヤ娇鐢?_block_complexity
        # 杩欎簺绠楀瓙閮芥湁鏈夋晥鐨勫鏉傚害璁＄畻
        params, macs, _ = self._block_complexity(block)
        return params, macs

    def get_lut_stats(self) -> dict[str, Any]:
        """鑾峰彇 LUT 缁熻淇℃伅"""
        total = self.lut_hits + self.lut_misses + self.deferred_hits
        deferred_by_op = Counter(record["lookup_op"] for record in self.deferred_hit_records)
        deferred_by_case = Counter(
            record["case_name"] or f"{record['lookup_op']}:{record['shape']}"
            for record in self.deferred_hit_records
        )
        deferred_by_reason = Counter(
            record["defer_reason"] or "unknown"
            for record in self.deferred_hit_records
        )
        misses_by_op = Counter(record["lookup_op"] for record in self.true_miss_records)
        misses_by_shape = Counter(
            f"{record['lookup_op']}:{record['shape']}"
            for record in self.true_miss_records
        )
        return {
            "hits": self.lut_hits,
            "misses": self.lut_misses,
            "true_misses": self.true_lut_misses,
            "deferred_hits": self.deferred_hits,
            "total": total,
            "hit_rate": self.lut_hits / total if total > 0 else 0.0,
            "fallback_rate": self.true_lut_misses / total if total > 0 else 0.0,
            "strict_formal_lut": self.strict_formal_lut,
            "strict_lut_label": self.strict_lut_label,
            "use_lut_for_head_layers": self.use_lut_for_head_layers,
            "deferred_hits_by_op": dict(deferred_by_op),
            "deferred_hits_by_case": dict(deferred_by_case),
            "deferred_hits_by_reason": dict(deferred_by_reason),
            "deferred_hit_records": list(self.deferred_hit_records),
            "true_misses_by_op": dict(misses_by_op),
            "true_misses_by_shape": dict(misses_by_shape),
            "true_miss_records": list(self.true_miss_records),
        }

    def reset_lut_stats(self) -> None:
        self.lut_hits = 0
        self.lut_misses = 0
        self.true_lut_misses = 0
        self.deferred_hits = 0
        self.deferred_hit_records = []
        self.true_miss_records = []
        self._estimate_dynamic_violations = []

    def _physical_constraints(self) -> dict[str, Any]:
        constraints = self.constraints
        if constraints is None:
            return {}
        physical = getattr(constraints, "physical", None)
        if not isinstance(physical, dict) or not physical:
            return {}
        if physical.get("enabled") is False:
            return {}
        return dict(physical)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    def _physical_capacity(self, physical: dict[str, Any], resource: str) -> Optional[float]:
        explicit = self._optional_float(physical.get(f"board_max_{resource}"))
        if explicit is not None:
            return explicit
        return self._optional_float(getattr(self.hardware_spec, f"max_{resource}", None))

    def _compute_physical_metrics(
        self,
        *,
        resolved_blocks: tuple[ResolvedBlockSpec, ...],
        peak_buffer_bytes: int,
    ) -> dict[str, float]:
        early_expand_pressure = 0.0
        interconnect_pressure = 0.0
        fanout_pressure = 0.0
        stream_width = 0.0
        early_max_expand = 1.0

        for block in resolved_blocks:
            expand = max(1, int(block.expand_ratio))
            hidden_channels = (
                int(block.in_channels) * expand
                if block.op in {"mbconv", "fused_mbconv"}
                else int(block.out_channels)
            )
            input_area = int(block.input_resolution) * int(block.input_resolution)
            output_area = int(block.output_resolution) * int(block.output_resolution)

            if block.op in {"mbconv", "fused_mbconv"}:
                early_expand_pressure += (
                    input_area * int(block.in_channels) * max(0, expand - 1)
                ) / 1_000_000.0
                if int(block.input_resolution) >= 56:
                    early_max_expand = max(early_max_expand, float(expand))

            interconnect_pressure += (
                output_area
                * (int(block.in_channels) + int(block.out_channels) + hidden_channels)
            ) / 1_000_000.0
            fanout_pressure = max(
                fanout_pressure,
                float(int(block.input_resolution) * hidden_channels) / 1024.0,
            )
            stream_width = max(stream_width, float(hidden_channels))

        memory_pressure = float(peak_buffer_bytes) / float(1024**2)
        physical_risk = (
            early_expand_pressure
            + 0.25 * interconnect_pressure
            + 0.10 * memory_pressure
            + 0.10 * fanout_pressure
        )
        return {
            "early_expand_pressure": early_expand_pressure,
            "interconnect_pressure": interconnect_pressure,
            "memory_pressure": memory_pressure,
            "fanout_pressure": fanout_pressure,
            "stream_width": stream_width,
            "early_max_expand": early_max_expand,
            "physical_risk": physical_risk,
        }

    def _check_physical_constraints(
        self,
        *,
        resource_dsp: int,
        resource_bram: int,
        resource_lut: int,
        physical_metrics: dict[str, float],
        resolved_blocks: tuple[ResolvedBlockSpec, ...],
    ) -> tuple[str, ...]:
        physical = self._physical_constraints()
        if not physical:
            return ()

        violations: list[str] = []
        max_total_dsp = self._optional_float(
            physical.get("max_total_dsp", physical.get("dsp_search_budget"))
        )
        if max_total_dsp is not None and float(resource_dsp) > max_total_dsp:
            violations.append(
                f"physical hard constraint: total DSP {resource_dsp} exceeds {max_total_dsp:g}"
            )

        utilization_specs = (
            ("dsp", float(resource_dsp), "max_dsp_utilization"),
            ("bram", float(resource_bram), "max_bram_utilization"),
            ("lut", float(resource_lut), "max_lut_utilization"),
        )
        for resource_name, used, key in utilization_specs:
            max_utilization = self._optional_float(physical.get(key))
            capacity = self._physical_capacity(physical, resource_name)
            if (
                max_utilization is not None
                and capacity is not None
                and capacity > 0
                and used > capacity * max_utilization
            ):
                violations.append(
                    "physical hard constraint: "
                    f"{resource_name.upper()} utilization {used / capacity:.3f} "
                    f"exceeds {max_utilization:g}"
                )

        early_rules: list[tuple[int, int]] = []
        for rule in physical.get("early_expand_limits", ()) or ():
            if not isinstance(rule, dict):
                continue
            min_resolution = self._optional_int(
                rule.get("min_input_resolution", rule.get("min_resolution"))
            )
            max_expand_ratio = self._optional_int(rule.get("max_expand_ratio"))
            if min_resolution is not None and max_expand_ratio is not None:
                early_rules.append((min_resolution, max_expand_ratio))
        early_rules.sort(reverse=True)
        for block in resolved_blocks:
            if block.op not in {"mbconv", "fused_mbconv"}:
                continue
            for min_resolution, max_expand_ratio in early_rules:
                if int(block.input_resolution) < min_resolution:
                    continue
                if int(block.expand_ratio) > max_expand_ratio:
                    violations.append(
                        "physical hard constraint: "
                        f"stage {block.stage_index} block {block.block_index} "
                        f"input_resolution={block.input_resolution} expand_ratio={block.expand_ratio} "
                        f"exceeds {max_expand_ratio} for resolution>={min_resolution}"
                    )
                break

        threshold_keys = (
            ("early_expand_pressure", "max_early_expand_pressure"),
            ("interconnect_pressure", "max_interconnect_pressure"),
            ("memory_pressure", "max_memory_pressure"),
            ("fanout_pressure", "max_fanout_pressure"),
            ("stream_width", "max_stream_width"),
            ("physical_risk", "max_physical_risk"),
        )
        for metric_name, config_key in threshold_keys:
            limit = self._optional_float(physical.get(config_key))
            value = physical_metrics.get(metric_name)
            if limit is not None and value is not None and float(value) > limit:
                violations.append(
                    "physical hard constraint: "
                    f"{metric_name} {float(value):.6g} exceeds {limit:g}"
                )

        return tuple(violations)

    def _check_constraints(
        self,
        *,
        latency_ms: float,
        energy_mj: float,
        model_size_mb: float,
        resource_dsp: int,
        resource_bram: int,
        resource_lut: int,
        power_w: float,
        memory_bandwidth_gbps: float,
        offchip_mem_mb: float,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.constraints:
            if (
                self.constraints.max_latency_ms is not None
                and latency_ms > self.constraints.max_latency_ms
            ):
                violations.append("latency exceeds max_latency_ms")
            if (
                self.constraints.max_energy_mj is not None
                and energy_mj > self.constraints.max_energy_mj
            ):
                violations.append("energy exceeds max_energy_mj")
            if (
                self.constraints.max_model_size_mb is not None
                and model_size_mb > self.constraints.max_model_size_mb
            ):
                violations.append("model size exceeds max_model_size_mb")
            if self.constraints.max_dsp is not None and resource_dsp > self.constraints.max_dsp:
                violations.append("DSP usage exceeds max_dsp")
            if self.constraints.max_bram is not None and resource_bram > self.constraints.max_bram:
                violations.append("BRAM usage exceeds max_bram")
            if self.constraints.max_lut is not None and resource_lut > self.constraints.max_lut:
                violations.append("LUT usage exceeds max_lut")
            if self.constraints.max_power_w is not None and power_w > self.constraints.max_power_w:
                violations.append("power exceeds max_power_w")
            if (
                self.constraints.max_memory_bandwidth_gbps is not None
                and memory_bandwidth_gbps > self.constraints.max_memory_bandwidth_gbps
            ):
                violations.append("memory bandwidth exceeds max_memory_bandwidth_gbps")
            if (
                self.constraints.max_offchip_mem_mb is not None
                and offchip_mem_mb > self.constraints.max_offchip_mem_mb
            ):
                violations.append("offchip memory exceeds max_offchip_mem_mb")

        if self.hardware_spec.max_power_w is not None and power_w > self.hardware_spec.max_power_w:
            violations.append("power exceeds hardware max_power_w")
        if self.hardware_spec.max_bram is not None and resource_bram > self.hardware_spec.max_bram:
            violations.append("BRAM usage exceeds hardware max_bram")
        if self.hardware_spec.max_lut is not None and resource_lut > self.hardware_spec.max_lut:
            violations.append("LUT usage exceeds hardware max_lut")
        if self.hardware_spec.max_dsp is not None and resource_dsp > self.hardware_spec.max_dsp:
            violations.append("DSP usage exceeds hardware max_dsp")
        if (
            self.hardware_spec.memory_bandwidth_gbps is not None
            and memory_bandwidth_gbps > self.hardware_spec.memory_bandwidth_gbps
        ):
            violations.append("memory bandwidth exceeds hardware memory_bandwidth_gbps")
        if (
            self.hardware_spec.offchip_mem_mb is not None
            and offchip_mem_mb > self.hardware_spec.offchip_mem_mb
        ):
            violations.append("offchip memory exceeds hardware offchip_mem_mb")
        return tuple(violations)

