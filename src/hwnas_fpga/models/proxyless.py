from __future__ import annotations

import math
from random import Random
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.search_space import (
    ArchitectureSpec,
    BlockSpec,
    ResolvedBlockSpec,
    SearchSpace,
    StageSpec,
)

from .builder import HeadBlock, StemBlock, build_block


class MixedOp(nn.Module):
    """Proxyless-style mixed operation with binary-path sampling."""

    METRIC_KEYS = (
        "latency_ms",
        "dsp",
        "bram",
        "lut",
    )

    def __init__(
        self,
        *,
        candidates: Sequence[nn.Module],
        candidate_specs: Sequence[BlockSpec],
        hardware_metrics: dict[str, Sequence[float]],
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not candidates:
            raise ValueError("MixedOp requires at least one candidate op")
        if len(candidates) != len(candidate_specs):
            raise ValueError("candidates and candidate_specs size mismatch")
        self.candidates = nn.ModuleList(list(candidates))
        self.candidate_specs = tuple(candidate_specs)
        self.temperature = max(1e-4, float(temperature))

        self.alpha = nn.Parameter(torch.zeros(len(self.candidates)))

        for key in self.METRIC_KEYS:
            values = list(hardware_metrics.get(key, []))
            if len(values) != len(self.candidates):
                values = [0.0 for _ in range(len(self.candidates))]
            self.register_buffer(
                f"metric_{key}",
                torch.tensor(values, dtype=torch.float32),
                persistent=False,
            )

        self._mode = "single"
        self._active_indices: tuple[int, ...] = (0,)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    def set_mode(self, mode: str) -> None:
        if mode not in {"single", "pair", "full"}:
            raise ValueError(f"Unsupported mixed-op mode: {mode}")
        self._mode = mode

    def probabilities(self) -> torch.Tensor:
        return F.softmax(self.alpha / self.temperature, dim=0)

    def set_active_index(self, index: int) -> None:
        idx = int(index)
        if idx < 0 or idx >= self.num_candidates:
            raise IndexError(f"active index out of range: {idx}")
        self._active_indices = (idx,)

    def set_argmax_active(self) -> None:
        idx = int(torch.argmax(self.alpha).item())
        self._active_indices = (idx,)

    def sample_single(self, rng: Random) -> int:
        probs = self.probabilities().detach().cpu().tolist()
        choices = list(range(self.num_candidates))
        idx = int(rng.choices(choices, weights=probs, k=1)[0])
        self._active_indices = (idx,)
        return idx

    def sample_pair(self, rng: Random) -> tuple[int, int]:
        if self.num_candidates == 1:
            self._active_indices = (0, 0)
            return (0, 0)

        probs = self.probabilities().detach().cpu().tolist()
        choices = list(range(self.num_candidates))
        first = int(rng.choices(choices, weights=probs, k=1)[0])

        second_weights = [0.0 if idx == first else weight for idx, weight in enumerate(probs)]
        if sum(second_weights) <= 1e-12:
            remaining = [idx for idx in choices if idx != first]
            second = int(rng.choice(remaining))
        else:
            second = int(rng.choices(choices, weights=second_weights, k=1)[0])

        self._active_indices = (first, second)
        return (first, second)

    def expected_metrics(self) -> dict[str, torch.Tensor]:
        probs = self.probabilities()
        metrics: dict[str, torch.Tensor] = {}
        for key in self.METRIC_KEYS:
            table = getattr(self, f"metric_{key}")
            metrics[key] = torch.sum(probs * table)
        return metrics

    def selected_candidate_spec(self) -> BlockSpec:
        idx = int(torch.argmax(self.alpha).item())
        return self.candidate_specs[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == "full":
            probs = self.probabilities()
            out = None
            for idx, op in enumerate(self.candidates):
                cur = op(x) * probs[idx]
                out = cur if out is None else (out + cur)
            if out is None:
                raise RuntimeError("No candidate output produced in full mode")
            return out

        if self._mode == "single":
            idx = int(self._active_indices[0])
            return self.candidates[idx](x)

        # pair mode
        if len(self._active_indices) < 2:
            idx = int(self._active_indices[0])
            return self.candidates[idx](x)

        i, j = int(self._active_indices[0]), int(self._active_indices[1])
        if i == j:
            return self.candidates[i](x)

        out_i = self.candidates[i](x)
        out_j = self.candidates[j](x)
        pair_logits = torch.stack([self.alpha[i], self.alpha[j]], dim=0) / self.temperature
        pair_probs = F.softmax(pair_logits, dim=0)
        return pair_probs[0] * out_i + pair_probs[1] * out_j


class ProxylessSuperNet(nn.Module):
    """ProxylessNAS-style supernet with binary-gated mixed operations."""

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        num_classes: int,
        cost_estimator: Optional[FPGACostEstimator] = None,
        stage_channels: Optional[Sequence[int]] = None,
        stage_depth: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.search_space = search_space
        self.config = search_space.config
        self.num_classes = int(num_classes)
        self.cost_estimator = cost_estimator

        self.stage_count = self.config.stage_count
        raw_stage_channels = list(
            stage_channels
            or self._infer_stage_channels(
                self.config.channel_choices,
                self.stage_count,
            )
        )
        valid_channels = tuple(int(channel) for channel in self.config.channel_choices)
        normalized_channels: list[int] = []
        for channel in raw_stage_channels:
            channel = int(channel)
            if channel in valid_channels:
                normalized_channels.append(channel)
                continue
            nearest = min(valid_channels, key=lambda candidate: abs(candidate - channel))
            normalized_channels.append(int(nearest))
        self.stage_channels = tuple(normalized_channels)
        if len(self.stage_channels) != self.stage_count:
            raise ValueError(
                f"stage_channels length {len(self.stage_channels)} != stage_count {self.stage_count}"
            )
        self.stage_depth = int(stage_depth or max(self.config.depth_choices))
        self.temperature = max(1e-4, float(temperature))

        self.stem = StemBlock(
            in_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            stride=self.config.stem_stride,
        )

        self.stages: nn.ModuleList = nn.ModuleList()
        self._mixed_ops: list[MixedOp] = []
        self._build_stages()

        final_channels = self.stage_channels[-1] if self.stage_channels else self.config.stem_channels
        self.head = HeadBlock(
            in_channels=final_channels,
            num_classes=self.num_classes,
            head_channels=self.config.head_channels,
        )

    @staticmethod
    def _infer_stage_channels(channel_choices: Sequence[int], stage_count: int) -> list[int]:
        choices = sorted(int(choice) for choice in channel_choices)
        if not choices:
            raise ValueError("channel_choices must not be empty")
        if stage_count <= 1:
            return [choices[0]]
        max_idx = len(choices) - 1
        result: list[int] = []
        for stage_idx in range(stage_count):
            ratio = stage_idx / max(1, stage_count - 1)
            idx = int(round(ratio * max_idx))
            result.append(choices[idx])
        return result

    def _build_stages(self) -> None:
        stage_strides = self.config.stage_strides
        kernel_choices = self.config.kernel_choices
        expand_choices = self.config.expand_choices

        current_channels = self.config.stem_channels
        current_resolution = max(1, math.ceil(self.config.image_size / self.config.stem_stride))

        for stage_idx in range(self.stage_count):
            stage_stride = int(stage_strides[stage_idx])
            out_channels = int(self.stage_channels[stage_idx])
            stage_blocks = nn.ModuleList()

            for block_idx in range(self.stage_depth):
                stride = stage_stride if block_idx == 0 else 1
                in_channels = current_channels if block_idx == 0 else out_channels
                output_resolution = max(1, math.ceil(current_resolution / stride))

                valid_ops = self.search_space.available_ops(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                )
                candidate_specs = self._build_candidate_specs(
                    valid_ops=valid_ops,
                    stride=stride,
                    kernel_choices=kernel_choices,
                    expand_choices=expand_choices,
                )
                candidates = [build_block(spec, in_channels, out_channels) for spec in candidate_specs]
                hardware_metrics = self._build_hardware_metrics(
                    stage_idx=stage_idx,
                    block_idx=block_idx,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    input_resolution=current_resolution,
                    output_resolution=output_resolution,
                    candidate_specs=candidate_specs,
                )

                mixed_op = MixedOp(
                    candidates=candidates,
                    candidate_specs=candidate_specs,
                    hardware_metrics=hardware_metrics,
                    temperature=self.temperature,
                )
                stage_blocks.append(mixed_op)
                self._mixed_ops.append(mixed_op)
                current_resolution = output_resolution

            self.stages.append(stage_blocks)
            current_channels = out_channels

    @staticmethod
    def _build_candidate_specs(
        *,
        valid_ops: Sequence[str],
        stride: int,
        kernel_choices: Sequence[int],
        expand_choices: Sequence[int],
    ) -> list[BlockSpec]:
        specs: list[BlockSpec] = []
        for op in valid_ops:
            if op == "skip":
                specs.append(BlockSpec(op="skip", kernel_size=1, expand_ratio=1, stride=stride))
                continue

            if op in {"mbconv", "fused_mbconv"}:
                for kernel in kernel_choices:
                    for expand in expand_choices:
                        specs.append(
                            BlockSpec(
                                op=op,
                                kernel_size=int(kernel),
                                expand_ratio=int(expand),
                                stride=stride,
                            )
                        )
                continue

            for kernel in kernel_choices:
                specs.append(
                    BlockSpec(
                        op=op,
                        kernel_size=int(kernel),
                        expand_ratio=1,
                        stride=stride,
                    )
                )

        # De-duplicate while preserving order
        unique_specs: list[BlockSpec] = []
        seen: set[tuple[str, int, int, int]] = set()
        for spec in specs:
            key = (spec.op, int(spec.kernel_size), int(spec.expand_ratio), int(spec.stride))
            if key in seen:
                continue
            unique_specs.append(spec)
            seen.add(key)
        return unique_specs

    def _build_hardware_metrics(
        self,
        *,
        stage_idx: int,
        block_idx: int,
        in_channels: int,
        out_channels: int,
        input_resolution: int,
        output_resolution: int,
        candidate_specs: Sequence[BlockSpec],
    ) -> dict[str, list[float]]:
        metrics = {key: [] for key in MixedOp.METRIC_KEYS}
        if self.cost_estimator is None:
            for _ in candidate_specs:
                for key in MixedOp.METRIC_KEYS:
                    metrics[key].append(0.0)
            return metrics

        clock_mhz = max(1.0, float(self.cost_estimator.hardware_spec.clock_mhz))
        for spec in candidate_specs:
            resolved = ResolvedBlockSpec(
                stage_index=stage_idx,
                block_index=block_idx,
                op=spec.op,
                kernel_size=int(spec.kernel_size),
                expand_ratio=int(spec.expand_ratio),
                stride=int(spec.stride),
                in_channels=int(in_channels),
                out_channels=int(out_channels),
                input_resolution=int(input_resolution),
                output_resolution=int(output_resolution),
            )

            try:
                layer_cost = self.cost_estimator._estimate_block(resolved)  # noqa: SLF001
                latency_ms = float(layer_cost.latency_cycles) / (clock_mhz * 1000.0)
                metrics["latency_ms"].append(latency_ms)
                metrics["dsp"].append(float(layer_cost.allocated_dsp))
                metrics["bram"].append(float(layer_cost.bram_blocks))
                metrics["lut"].append(float(layer_cost.lut))
            except Exception:
                metrics["latency_ms"].append(0.0)
                metrics["dsp"].append(0.0)
                metrics["bram"].append(0.0)
                metrics["lut"].append(0.0)

        return metrics

    @property
    def mixed_ops(self) -> list[MixedOp]:
        return self._mixed_ops

    def set_forward_mode(self, mode: str) -> None:
        for mixed in self._mixed_ops:
            mixed.set_mode(mode)

    def sample_single_paths(self, rng: Random) -> None:
        self.set_forward_mode("single")
        for mixed in self._mixed_ops:
            mixed.sample_single(rng)

    def sample_pair_paths(self, rng: Random) -> None:
        self.set_forward_mode("pair")
        for mixed in self._mixed_ops:
            mixed.sample_pair(rng)

    def activate_argmax_paths(self) -> None:
        self.set_forward_mode("single")
        for mixed in self._mixed_ops:
            mixed.set_argmax_active()

    def expected_hardware_metrics(self) -> dict[str, torch.Tensor]:
        if not self._mixed_ops:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {
                "latency_ms": zero,
                "dsp": zero,
                "bram": zero,
                "lut": zero,
                "energy_proxy_mj": zero,
            }

        device = self._mixed_ops[0].alpha.device
        total = {
            "latency_ms": torch.zeros((), device=device),
            "dsp": torch.zeros((), device=device),
            "bram": torch.zeros((), device=device),
            "lut": torch.zeros((), device=device),
        }
        for mixed in self._mixed_ops:
            expected = mixed.expected_metrics()
            for key in total:
                total[key] = total[key] + expected[key]

        # Smooth differentiable proxy for energy.
        power_proxy = 2.5 + 0.012 * total["dsp"] + 0.018 * total["bram"] + 0.0002 * total["lut"]
        total["energy_proxy_mj"] = total["latency_ms"] * power_proxy
        return total

    def arch_parameters(self) -> list[nn.Parameter]:
        return [mixed.alpha for mixed in self._mixed_ops]

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if name.endswith("alpha"):
                continue
            yield param

    def extract_architecture(self) -> ArchitectureSpec:
        stage_stride = self.config.stage_strides
        depth_choices = tuple(sorted(int(value) for value in self.config.depth_choices))
        min_depth = depth_choices[0]
        max_depth = depth_choices[-1]

        stages: list[StageSpec] = []
        for stage_idx, stage_blocks in enumerate(self.stages):
            selected = [mixed.selected_candidate_spec() for mixed in stage_blocks]
            non_skip = [idx for idx, spec in enumerate(selected) if spec.op != "skip"]
            if non_skip:
                inferred_depth = non_skip[-1] + 1
            else:
                inferred_depth = min_depth

            inferred_depth = max(min_depth, min(max_depth, inferred_depth))
            if inferred_depth not in depth_choices:
                inferred_depth = min(depth_choices, key=lambda d: abs(d - inferred_depth))

            blocks = tuple(selected[:inferred_depth])
            stages.append(
                StageSpec(
                    channels=int(self.stage_channels[stage_idx]),
                    depth=int(inferred_depth),
                    stride=int(stage_stride[stage_idx]),
                    blocks=blocks,
                )
            )

        return ArchitectureSpec(
            input_channels=int(self.config.input_channels),
            stem_channels=int(self.config.stem_channels),
            stem_stride=int(self.config.stem_stride),
            stages=tuple(stages),
            head_channels=self.config.head_channels,
            num_classes=self.num_classes,
        )

    def choice_probabilities(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for mixed in self._mixed_ops:
            rows.append(mixed.probabilities().detach().cpu().tolist())
        return rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            for mixed in stage:
                x = mixed(x)
        x = self.head(x)
        return x


__all__ = [
    "MixedOp",
    "ProxylessSuperNet",
]
