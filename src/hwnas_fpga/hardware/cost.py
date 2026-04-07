from __future__ import annotations

from dataclasses import dataclass
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
        self.lut_hits = 0  # LUT 鍛戒腑璁℃暟
        self.lut_misses = 0  # LUT 鏈懡涓鏁?
    def estimate(
        self,
        architecture: ArchitectureSpec,
        search_space: SearchSpace,
    ) -> CostEstimate:
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
        latency_ms = latency_cycles / (self.hardware_spec.clock_mhz * 1_000)
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
        violations = self._check_constraints(
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
            violations=violations,
            per_layer=per_layer,
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

        allocated_dsp = min(max(1, dsp_raw), self._dsp_budget)
        throughput = max(1.0, allocated_dsp * self._pack_factor * self.pipeline_efficiency)
        latency_cycles = max(1, _div_up(macs, throughput))
        weight_bytes = params * self._bytes_per_scalar
        activation_bytes = output_channels * self._bytes_per_scalar
        pooled_bytes = self._tensor_bytes(1, final_channels)
        bram_blocks = _div_up(pooled_bytes + activation_bytes + weight_bytes, BRAM_BLOCK_BYTES)
        lut = lut_bias + allocated_dsp * 10 + output_channels * 4

        head_layers.append(
            LayerCost(
                stage_index=len(architecture.stages),
                block_index=-1,
                op="head_fc",
                input_resolution=final_resolution,
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
            )
        )
        return tuple(head_layers)

    def _estimate_block(self, block: ResolvedBlockSpec) -> LayerCost:
        # 1. 灏濊瘯 LUT 鏌ヨ锛堝鏋滃彲鐢級
        if self.lut_query_engine:
            op_spec = self._block_to_op_spec(block)
            lut_entry = self.lut_query_engine.query(op_spec)
            if lut_entry:
                self.lut_hits += 1
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
                )
            self.lut_misses += 1

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
        )

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
        }[block.op]
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
        # 鏄犲皠绠楀瓙鍚嶇О
        op_mapping = {
            "conv": "conv",
            "dw_pw_conv": "dw_pw_conv",
            "mbconv": "mbconv",
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
        total = self.lut_hits + self.lut_misses
        return {
            "hits": self.lut_hits,
            "misses": self.lut_misses,
            "total": total,
            "hit_rate": self.lut_hits / total if total > 0 else 0.0,
        }

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

