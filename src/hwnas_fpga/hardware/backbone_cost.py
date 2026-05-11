from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from time import perf_counter
from typing import Optional

import torch
import torch.nn as nn

from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints


BRAM_BLOCK_BYTES = 36 * 1024 // 8


def _div_up(value: float | int, divisor: float | int) -> int:
    return int(ceil(value / divisor))


@dataclass(frozen=True)
class BackboneLayerCost:
    name: str
    op: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
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
class BackboneCostEstimate:
    params: int
    macs: int
    model_size_mb: float
    cpu_latency_ms: float
    peak_activation_bytes: int
    peak_weight_bytes: int
    peak_buffer_bytes: int
    peak_dsp: int
    peak_bram: int
    peak_lut: int
    latency_cycles: int
    latency_ms: float
    power_w: float
    energy_mj: float
    memory_bandwidth_gbps: float
    offchip_mem_mb: float
    feasible: bool
    violations: tuple[str, ...]
    per_layer: tuple[BackboneLayerCost, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BackboneCostEstimator:
    """Approximate whole-backbone FPGA cost from conv/linear layers."""

    def __init__(
        self,
        hardware_spec: HardwareSpec,
        constraints: Optional[SearchConstraints] = None,
        *,
        quantization_bits: int = 8,
        pipeline_efficiency: float = 0.7,
        default_dsp_budget: int = 256,
        cpu_latency_warmup: int = 10,
        cpu_latency_runs: int = 30,
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
        self.cpu_latency_warmup = max(0, cpu_latency_warmup)
        self.cpu_latency_runs = max(1, cpu_latency_runs)

    @property
    def _bytes_per_scalar(self) -> int:
        return max(1, self.quantization_bits // 8)

    @property
    def _dsp_budget(self) -> int:
        return self.hardware_spec.max_dsp or self.default_dsp_budget

    @property
    def _pack_factor(self) -> int:
        return 2 if self.quantization_bits <= 8 else 1

    def measure_cpu_latency(
        self,
        model: nn.Module,
        *,
        input_shape: tuple[int, int, int, int],
        warmup_runs: Optional[int] = None,
        runs: Optional[int] = None,
    ) -> float:
        warmup = self.cpu_latency_warmup if warmup_runs is None else max(0, warmup_runs)
        iterations = self.cpu_latency_runs if runs is None else max(1, runs)
        model = model.to("cpu").eval()
        dummy = torch.randn(*input_shape)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)

            start = perf_counter()
            for _ in range(iterations):
                _ = model(dummy)
            elapsed = perf_counter() - start

        return elapsed * 1000.0 / iterations

    def estimate(
        self,
        model: nn.Module,
        *,
        input_shape: tuple[int, int, int, int],
        measure_cpu_latency: bool = True,
        cpu_latency_runs: Optional[int] = None,
        cpu_latency_warmup: Optional[int] = None,
    ) -> BackboneCostEstimate:
        profiled_layers: list[BackboneLayerCost] = []
        hooks = []

        def register_hook(layer_name: str, layer: nn.Module) -> None:
            def _hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                input_tensor = inputs[0]
                if isinstance(layer, nn.Conv2d):
                    profiled_layers.append(
                        self._profile_conv(layer_name, layer, input_tensor, output)
                    )
                elif isinstance(layer, nn.Linear):
                    profiled_layers.append(
                        self._profile_linear(layer_name, layer, input_tensor, output)
                    )

            hooks.append(layer.register_forward_hook(_hook))

        for layer_name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                register_hook(layer_name, layer)

        model = model.to("cpu").eval()
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            _ = model(dummy)

        for hook in hooks:
            hook.remove()

        params = sum(param.numel() for param in model.parameters())
        total_macs = sum(layer.macs for layer in profiled_layers)
        peak_activation_bytes = max((layer.activation_bytes for layer in profiled_layers), default=0)
        peak_weight_bytes = max((layer.weight_bytes for layer in profiled_layers), default=0)
        peak_buffer_bytes = max(
            (layer.activation_bytes + layer.weight_bytes for layer in profiled_layers),
            default=0,
        )
        peak_dsp = max((layer.allocated_dsp for layer in profiled_layers), default=0)
        peak_bram = max((layer.bram_blocks for layer in profiled_layers), default=0)
        peak_lut = max((layer.lut for layer in profiled_layers), default=0)
        latency_cycles = sum(layer.latency_cycles for layer in profiled_layers)
        latency_ms = latency_cycles / (self.hardware_spec.clock_mhz * 1_000)
        model_size_mb = params * self._bytes_per_scalar / (1024**2)
        power_w = self._estimate_power(peak_dsp=peak_dsp, peak_bram=peak_bram, peak_lut=peak_lut)
        energy_mj = power_w * latency_ms
        total_memory_traffic_bytes = sum(
            layer.activation_bytes + layer.weight_bytes for layer in profiled_layers
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
            peak_dsp=peak_dsp,
            peak_bram=peak_bram,
            peak_lut=peak_lut,
            power_w=power_w,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            offchip_mem_mb=offchip_mem_mb,
        )
        cpu_latency_ms = 0.0
        if measure_cpu_latency:
            cpu_latency_ms = self.measure_cpu_latency(
                model,
                input_shape=input_shape,
                warmup_runs=cpu_latency_warmup,
                runs=cpu_latency_runs,
            )

        return BackboneCostEstimate(
            params=params,
            macs=total_macs,
            model_size_mb=model_size_mb,
            cpu_latency_ms=cpu_latency_ms,
            peak_activation_bytes=peak_activation_bytes,
            peak_weight_bytes=peak_weight_bytes,
            peak_buffer_bytes=peak_buffer_bytes,
            peak_dsp=peak_dsp,
            peak_bram=peak_bram,
            peak_lut=peak_lut,
            latency_cycles=latency_cycles,
            latency_ms=latency_ms,
            power_w=power_w,
            energy_mj=energy_mj,
            memory_bandwidth_gbps=memory_bandwidth_gbps,
            offchip_mem_mb=offchip_mem_mb,
            feasible=len(violations) == 0,
            violations=violations,
            per_layer=tuple(profiled_layers),
        )

    def _profile_conv(
        self,
        name: str,
        layer: nn.Conv2d,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> BackboneLayerCost:
        kernel_h, kernel_w = layer.kernel_size
        _, in_w = input_tensor.shape[-2:]
        out_h, out_w = output_tensor.shape[-2:]
        input_channels_per_group = layer.in_channels // layer.groups
        params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        macs = (
            out_h
            * out_w
            * layer.out_channels
            * input_channels_per_group
            * kernel_h
            * kernel_w
        )
        weight_bytes = params * self._bytes_per_scalar
        activation_bytes = output_tensor.numel() * self._bytes_per_scalar
        input_bytes = input_tensor.numel() * self._bytes_per_scalar
        depthwise = layer.groups == layer.in_channels and layer.out_channels == layer.in_channels
        ideal_dsp = self._conv_dsp(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=max(kernel_h, kernel_w),
            depthwise=depthwise,
        )
        allocated_dsp = min(max(1, ideal_dsp), self._dsp_budget)
        throughput = max(1.0, allocated_dsp * self._pack_factor * self.pipeline_efficiency)
        latency_cycles = max(1, _div_up(macs, throughput))
        # HAO/Tiling 机制的 BRAM 估算
        # 我们不把整张特征图存入 BRAM，而是使用行缓存 (Line Buffer) 和通道分块 (Channel Tiling)
        # Line Buffer 需要存储 kernel_h - 1 行完整的输入特征图用于窗口滑动
        line_buffer_bytes = (kernel_h - 1) * in_w * layer.in_channels * self._bytes_per_scalar
        
        # Tile Buffer 存储当前计算窗口所需的数据 (受限于 DSP 并行度)
        # 假设我们缓存输入通道块(Tin)和输出通道块(Tout)
        Tin = min(layer.in_channels, 16)
        Tout = min(layer.out_channels, 32)
        tile_activation_bytes = kernel_h * kernel_w * Tin * self._bytes_per_scalar
        if depthwise:
            tile_weight_bytes = kernel_h * kernel_w * Tin * self._bytes_per_scalar
        else:
            tile_weight_bytes = kernel_h * kernel_w * Tin * Tout * self._bytes_per_scalar
        
        # 实际 BRAM 消耗是行缓存、分块缓存和部分权重缓存的总和
        total_bram_bytes = line_buffer_bytes + tile_activation_bytes + tile_weight_bytes
        
        bram_blocks = _div_up(total_bram_bytes, BRAM_BLOCK_BYTES)
        lut = self._estimate_lut(
            op="dw_conv" if depthwise else "conv",
            out_channels=layer.out_channels,
            kernel_size=max(kernel_h, kernel_w),
            output_resolution=max(out_h, out_w),
            allocated_dsp=allocated_dsp,
        )
        return BackboneLayerCost(
            name=name,
            op="dw_conv" if depthwise else "conv",
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(output_tensor.shape),
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

    def _profile_linear(
        self,
        name: str,
        layer: nn.Linear,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> BackboneLayerCost:
        params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        macs = layer.in_features * layer.out_features
        weight_bytes = params * self._bytes_per_scalar
        activation_bytes = output_tensor.numel() * self._bytes_per_scalar
        input_bytes = input_tensor.numel() * self._bytes_per_scalar
        ideal_dsp = self._linear_dsp(layer.in_features, layer.out_features)
        allocated_dsp = min(max(1, ideal_dsp), self._dsp_budget)
        throughput = max(1.0, allocated_dsp * self._pack_factor * self.pipeline_efficiency)
        latency_cycles = max(1, _div_up(macs, throughput))
        bram_blocks = _div_up(input_bytes + activation_bytes + weight_bytes, BRAM_BLOCK_BYTES)
        lut = 64 + allocated_dsp * 10 + layer.out_features * 4
        return BackboneLayerCost(
            name=name,
            op="linear",
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(output_tensor.shape),
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

    def _linear_dsp(self, in_features: int, out_features: int) -> int:
        return _div_up(
            max(1, min(in_features, 256)) * max(1, min(out_features, 64)),
            256 * self._pack_factor,
        )

    def _estimate_lut(
        self,
        *,
        op: str,
        out_channels: int,
        kernel_size: int,
        output_resolution: int,
        allocated_dsp: int,
    ) -> int:
        kernel_factor = 2 if kernel_size >= 5 else 1
        op_bias = {
            "conv": 120,
            "dw_conv": 96,
            "linear": 64,
        }.get(op, 100)
        return op_bias + allocated_dsp * 18 + out_channels * kernel_factor * 4 + output_resolution * 2

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

    def _check_constraints(
        self,
        *,
        latency_ms: float,
        energy_mj: float,
        model_size_mb: float,
        peak_dsp: int,
        peak_bram: int,
        peak_lut: int,
        power_w: float,
        memory_bandwidth_gbps: float,
        offchip_mem_mb: float,
    ) -> tuple[str, ...]:
        violations: list[str] = []
        if self.constraints:
            if self.constraints.max_latency_ms is not None and latency_ms > self.constraints.max_latency_ms:
                violations.append("latency exceeds max_latency_ms")
            if self.constraints.max_energy_mj is not None and energy_mj > self.constraints.max_energy_mj:
                violations.append("energy exceeds max_energy_mj")
            if self.constraints.max_model_size_mb is not None and model_size_mb > self.constraints.max_model_size_mb:
                violations.append("model size exceeds max_model_size_mb")
            if self.constraints.max_dsp is not None and peak_dsp > self.constraints.max_dsp:
                violations.append("DSP usage exceeds max_dsp")
            if self.constraints.max_bram is not None and peak_bram > self.constraints.max_bram:
                violations.append("BRAM usage exceeds max_bram")
            if self.constraints.max_lut is not None and peak_lut > self.constraints.max_lut:
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
        if self.hardware_spec.max_bram is not None and peak_bram > self.hardware_spec.max_bram:
            violations.append("BRAM usage exceeds hardware max_bram")
        if self.hardware_spec.max_lut is not None and peak_lut > self.hardware_spec.max_lut:
            violations.append("LUT usage exceeds hardware max_lut")
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
