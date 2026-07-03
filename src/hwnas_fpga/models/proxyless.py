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

    unique_specs: list[BlockSpec] = []
    seen: set[tuple[str, int, int, int]] = set()
    for spec in specs:
        key = (spec.op, int(spec.kernel_size), int(spec.expand_ratio), int(spec.stride))
        if key in seen:
            continue
        unique_specs.append(spec)
        seen.add(key)
    return unique_specs


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


class ChoiceGate(nn.Module):
    """Discrete architecture choice with Proxyless-style sampling modes."""

    def __init__(self, choices: Sequence[int], *, temperature: float = 1.0) -> None:
        super().__init__()
        normalized = tuple(int(choice) for choice in choices)
        if not normalized:
            raise ValueError("ChoiceGate requires at least one choice")
        self.choices = normalized
        self.temperature = max(1e-4, float(temperature))
        self.alpha = nn.Parameter(torch.zeros(len(self.choices)))
        self._mode = "single"
        self._active_indices: tuple[int, ...] = (0,)

    @property
    def num_choices(self) -> int:
        return len(self.choices)

    def set_mode(self, mode: str) -> None:
        if mode not in {"single", "pair", "full"}:
            raise ValueError(f"Unsupported choice-gate mode: {mode}")
        self._mode = mode

    def probabilities(self) -> torch.Tensor:
        return F.softmax(self.alpha / self.temperature, dim=0)

    def set_argmax_active(self) -> None:
        self._active_indices = (int(torch.argmax(self.alpha).item()),)

    def sample_single(self, rng: Random) -> int:
        probs = self.probabilities().detach().cpu().tolist()
        choices = list(range(self.num_choices))
        idx = int(rng.choices(choices, weights=probs, k=1)[0])
        self._active_indices = (idx,)
        return idx

    def sample_pair(self, rng: Random) -> tuple[int, int]:
        if self.num_choices == 1:
            self._active_indices = (0, 0)
            return (0, 0)

        probs = self.probabilities().detach().cpu().tolist()
        choices = list(range(self.num_choices))
        first = int(rng.choices(choices, weights=probs, k=1)[0])
        second_weights = [0.0 if idx == first else weight for idx, weight in enumerate(probs)]
        if sum(second_weights) <= 1e-12:
            remaining = [idx for idx in choices if idx != first]
            second = int(rng.choice(remaining))
        else:
            second = int(rng.choices(choices, weights=second_weights, k=1)[0])
        self._active_indices = (first, second)
        return (first, second)

    def active_distribution(self) -> tuple[tuple[int, ...], torch.Tensor]:
        device = self.alpha.device
        dtype = self.alpha.dtype
        if self._mode == "full":
            indices = tuple(range(self.num_choices))
            return indices, self.probabilities()

        if self._mode == "single" or len(self._active_indices) < 2:
            index = int(self._active_indices[0])
            weights = torch.ones(1, device=device, dtype=dtype)
            return (index,), weights

        i, j = int(self._active_indices[0]), int(self._active_indices[1])
        if i == j:
            weights = torch.ones(1, device=device, dtype=dtype)
            return (i,), weights

        pair_logits = torch.stack([self.alpha[i], self.alpha[j]], dim=0) / self.temperature
        pair_weights = F.softmax(pair_logits, dim=0)
        return (i, j), pair_weights

    def selected_choice(self) -> int:
        return int(self.choices[int(torch.argmax(self.alpha).item())])


class SearchableStage(nn.Module):
    """A stage with searchable width, depth, and block operators."""

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        stage_idx: int,
        input_channel_choices: Sequence[int],
        output_channel_choices: Sequence[int],
        depth_choices: Sequence[int],
        input_resolution: int,
        cost_estimator: Optional[FPGACostEstimator],
        temperature: float,
    ) -> None:
        super().__init__()
        self.search_space = search_space
        self.config = search_space.config
        self.stage_idx = int(stage_idx)
        self.stage_stride = int(self.config.stage_strides[self.stage_idx])
        self.temperature = max(1e-4, float(temperature))
        self.cost_estimator = cost_estimator

        self.input_channel_choices = tuple(int(choice) for choice in input_channel_choices)
        self.channel_choices = tuple(int(choice) for choice in output_channel_choices)
        self.depth_choices = tuple(int(choice) for choice in depth_choices)
        if not self.input_channel_choices:
            raise ValueError("input_channel_choices must not be empty")
        if not self.channel_choices:
            raise ValueError("channel_choices must not be empty")
        if not self.depth_choices:
            raise ValueError("depth_choices must not be empty")

        self.max_input_channels = max(self.input_channel_choices)
        self.max_channels = max(self.channel_choices)
        self.max_depth = max(self.depth_choices)
        self.input_resolution = int(input_resolution)

        self.channel_gate = ChoiceGate(self.channel_choices, temperature=self.temperature)
        self.depth_gate = ChoiceGate(self.depth_choices, temperature=self.temperature)

        channel_masks = torch.zeros(len(self.channel_choices), self.max_channels, dtype=torch.float32)
        for index, channels in enumerate(self.channel_choices):
            channel_masks[index, : int(channels)] = 1.0
        self.register_buffer("channel_masks", channel_masks, persistent=False)

        self.blocks = nn.ModuleList()
        self._mixed_ops: list[MixedOp] = []
        self._block_cost_tables: list[dict[str, torch.Tensor]] = []
        self.output_resolution = self.input_resolution
        self._build_blocks()

    def _build_blocks(self) -> None:
        current_resolution = self.input_resolution

        for block_idx in range(self.max_depth):
            stride = self.stage_stride if block_idx == 0 else 1
            in_channels = self.max_input_channels if block_idx == 0 else self.max_channels
            output_resolution = max(1, math.ceil(current_resolution / stride))

            stage_block_choices = self.search_space.concrete_block_choices_for_stage(
                stage_index=self.stage_idx,
                in_channels=in_channels,
                out_channels=self.max_channels,
                stride=stride,
            )
            if stage_block_choices is not None:
                candidate_specs = list(stage_block_choices)
            else:
                valid_ops = self.search_space.available_ops(
                    in_channels=in_channels,
                    out_channels=self.max_channels,
                    stride=stride,
                )
                candidate_specs = _build_candidate_specs(
                    valid_ops=valid_ops,
                    stride=stride,
                    kernel_choices=self.config.kernel_choices,
                    expand_choices=self.config.expand_choices,
                )
            candidates = [build_block(spec, in_channels, self.max_channels) for spec in candidate_specs]
            hardware_metrics = self._build_mixed_op_metrics(
                candidate_specs=candidate_specs,
                in_channels=in_channels,
                out_channels=self.max_channels,
                block_idx=block_idx,
                input_resolution=current_resolution,
                output_resolution=output_resolution,
            )
            mixed_op = MixedOp(
                candidates=candidates,
                candidate_specs=candidate_specs,
                hardware_metrics=hardware_metrics,
                temperature=self.temperature,
            )
            self.blocks.append(mixed_op)
            self._mixed_ops.append(mixed_op)
            self._block_cost_tables.append(
                self._build_block_cost_table(
                    candidate_specs=candidate_specs,
                    block_idx=block_idx,
                    input_resolution=current_resolution,
                    output_resolution=output_resolution,
                )
            )
            current_resolution = output_resolution

        self.output_resolution = current_resolution

    def _estimate_block_cost(
        self,
        *,
        block_idx: int,
        spec: BlockSpec,
        in_channels: int,
        out_channels: int,
        input_resolution: int,
        output_resolution: int,
    ) -> dict[str, float]:
        if self.cost_estimator is None:
            return {key: 0.0 for key in MixedOp.METRIC_KEYS}

        resolved = ResolvedBlockSpec(
            stage_index=self.stage_idx,
            block_index=int(block_idx),
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
            return {
                "latency_ms": layer_cost.resolved_latency_ms(
                    self.cost_estimator.hardware_spec.clock_mhz
                ),
                "dsp": float(layer_cost.allocated_dsp),
                "bram": float(layer_cost.bram_blocks),
                "lut": float(layer_cost.lut),
            }
        except Exception:
            return {key: 0.0 for key in MixedOp.METRIC_KEYS}

    def _build_mixed_op_metrics(
        self,
        *,
        candidate_specs: Sequence[BlockSpec],
        in_channels: int,
        out_channels: int,
        block_idx: int,
        input_resolution: int,
        output_resolution: int,
    ) -> dict[str, list[float]]:
        metrics = {key: [] for key in MixedOp.METRIC_KEYS}
        for spec in candidate_specs:
            cost = self._estimate_block_cost(
                block_idx=block_idx,
                spec=spec,
                in_channels=in_channels,
                out_channels=out_channels,
                input_resolution=input_resolution,
                output_resolution=output_resolution,
            )
            for key in MixedOp.METRIC_KEYS:
                metrics[key].append(float(cost[key]))
        return metrics

    def _build_block_cost_table(
        self,
        *,
        candidate_specs: Sequence[BlockSpec],
        block_idx: int,
        input_resolution: int,
        output_resolution: int,
    ) -> dict[str, torch.Tensor]:
        if block_idx == 0:
            costs = {key: [] for key in MixedOp.METRIC_KEYS}
            for in_channels in self.input_channel_choices:
                row = {key: [] for key in MixedOp.METRIC_KEYS}
                for out_channels in self.channel_choices:
                    column = {key: [] for key in MixedOp.METRIC_KEYS}
                    for spec in candidate_specs:
                        cost = self._estimate_block_cost(
                            block_idx=block_idx,
                            spec=spec,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            input_resolution=input_resolution,
                            output_resolution=output_resolution,
                        )
                        for key in MixedOp.METRIC_KEYS:
                            column[key].append(float(cost[key]))
                    for key in MixedOp.METRIC_KEYS:
                        row[key].append(column[key])
                for key in MixedOp.METRIC_KEYS:
                    costs[key].append(row[key])
            return {
                key: torch.tensor(values, dtype=torch.float32)
                for key, values in costs.items()
            }

        costs = {key: [] for key in MixedOp.METRIC_KEYS}
        for channels in self.channel_choices:
            row = {key: [] for key in MixedOp.METRIC_KEYS}
            for spec in candidate_specs:
                cost = self._estimate_block_cost(
                    block_idx=block_idx,
                    spec=spec,
                    in_channels=channels,
                    out_channels=channels,
                    input_resolution=input_resolution,
                    output_resolution=output_resolution,
                )
                for key in MixedOp.METRIC_KEYS:
                    row[key].append(float(cost[key]))
            for key in MixedOp.METRIC_KEYS:
                costs[key].append(row[key])
        return {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in costs.items()
        }

    def _channel_mask(self, x: torch.Tensor) -> torch.Tensor:
        indices, weights = self.channel_gate.active_distribution()
        weight_tensor = weights.to(device=x.device, dtype=x.dtype)
        mask = self.channel_masks[list(indices)].to(device=x.device, dtype=x.dtype)
        combined = torch.sum(weight_tensor.unsqueeze(1) * mask, dim=0)
        return combined.view(1, self.max_channels, 1, 1)

    def _depth_output(self, block_outputs: list[torch.Tensor]) -> torch.Tensor:
        indices, weights = self.depth_gate.active_distribution()
        weight_tensor = weights.to(device=block_outputs[0].device, dtype=block_outputs[0].dtype)
        outputs = []
        for gate_index in indices:
            depth = int(self.depth_choices[gate_index])
            outputs.append(block_outputs[depth - 1])
        combined = outputs[0] * weight_tensor[0]
        for output, weight in zip(outputs[1:], weight_tensor[1:]):
            combined = combined + output * weight
        return combined

    def set_mode(self, mode: str) -> None:
        self.channel_gate.set_mode(mode)
        self.depth_gate.set_mode(mode)
        for mixed in self._mixed_ops:
            mixed.set_mode(mode)

    def sample_single(self, rng: Random) -> None:
        self.set_mode("single")
        self.channel_gate.sample_single(rng)
        self.depth_gate.sample_single(rng)
        for mixed in self._mixed_ops:
            mixed.sample_single(rng)

    def sample_pair(self, rng: Random) -> None:
        self.set_mode("pair")
        self.channel_gate.sample_pair(rng)
        self.depth_gate.sample_pair(rng)
        for mixed in self._mixed_ops:
            mixed.sample_pair(rng)

    def activate_argmax(self) -> None:
        self.set_mode("single")
        self.channel_gate.set_argmax_active()
        self.depth_gate.set_argmax_active()
        for mixed in self._mixed_ops:
            mixed.set_argmax_active()

    def expected_hardware_metrics(self, prev_channel_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        device = prev_channel_probs.device
        channel_probs = self.channel_gate.probabilities()
        depth_probs = self.depth_gate.probabilities()

        totals = {
            "latency_ms": torch.zeros((), device=device),
            "dsp": torch.zeros((), device=device),
            "bram": torch.zeros((), device=device),
            "lut": torch.zeros((), device=device),
        }

        active_probabilities = []
        for block_idx in range(self.max_depth):
            is_active = torch.tensor(
                [1.0 if depth > block_idx else 0.0 for depth in self.depth_choices],
                device=device,
                dtype=channel_probs.dtype,
            )
            active_probabilities.append(torch.sum(depth_probs * is_active))

        for block_idx, mixed in enumerate(self._mixed_ops):
            op_probs = mixed.probabilities()
            active_prob = active_probabilities[block_idx]
            cost_tables = self._block_cost_tables[block_idx]
            for key, table in cost_tables.items():
                cost_tensor = table.to(device=device, dtype=channel_probs.dtype)
                if block_idx == 0:
                    expected = torch.einsum("p,c,k,pck->", prev_channel_probs, channel_probs, op_probs, cost_tensor)
                else:
                    expected = torch.einsum("c,k,ck->", channel_probs, op_probs, cost_tensor)
                totals[key] = totals[key] + active_prob * expected
        return totals

    def extract_stage_spec(self) -> StageSpec:
        channels = self.channel_gate.selected_choice()
        depth = self.depth_gate.selected_choice()
        blocks = tuple(mixed.selected_candidate_spec() for mixed in self._mixed_ops[:depth])
        return StageSpec(
            channels=int(channels),
            depth=int(depth),
            stride=self.stage_stride,
            blocks=blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_outputs: list[torch.Tensor] = []
        channel_mask = self._channel_mask(x)
        for mixed in self._mixed_ops:
            x = mixed(x)
            x = x * channel_mask
            block_outputs.append(x)
        return self._depth_output(block_outputs)


class ProxylessSuperNet(nn.Module):
    """ProxylessNAS-style supernet over the repo's stage-based search space."""

    def __init__(
        self,
        *,
        search_space: SearchSpace,
        num_classes: int,
        cost_estimator: Optional[FPGACostEstimator] = None,
        stage_channels: Optional[Sequence[int]] = None,
        stage_depth: Optional[int | Sequence[int]] = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.search_space = search_space
        self.config = search_space.config
        self.num_classes = int(num_classes)
        self.cost_estimator = cost_estimator
        self.stage_count = self.config.stage_count
        self.temperature = max(1e-4, float(temperature))

        self.stage_channel_choices = self._resolve_stage_channel_choices(stage_channels)
        self.stage_depth_choices = self._resolve_stage_depth_choices(stage_depth)

        self.stem = StemBlock(
            in_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            stride=self.config.stem_stride,
        )
        self.post_stem_downsample = (
            nn.MaxPool2d(
                kernel_size=3,
                stride=self.config.post_stem_downsample_stride,
                padding=1,
            )
            if self.config.post_stem_downsample_stride > 1
            else None
        )

        self.stages = nn.ModuleList()
        self._mixed_ops: list[MixedOp] = []
        self._build_stages()

        final_channels = max(self.stage_channel_choices[-1]) if self.stage_channel_choices else self.config.stem_channels
        self.head = HeadBlock(
            in_channels=final_channels,
            num_classes=self.num_classes,
            head_channels=self.config.head_channels,
            conv_head_channels=self.config.head_conv_channels,
        )

    def _resolve_stage_channel_choices(
        self,
        stage_channels: Optional[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        resolved: list[tuple[int, ...]] = []
        if stage_channels is None:
            for stage_idx in range(self.stage_count):
                resolved.append(tuple(int(choice) for choice in self.config.channel_choices_for_stage(stage_idx)))
            return tuple(resolved)

        requested = [int(channel) for channel in stage_channels]
        if len(requested) != self.stage_count:
            raise ValueError(
                f"stage_channels length {len(requested)} != stage_count {self.stage_count}"
            )
        for stage_idx, channel in enumerate(requested):
            allowed = tuple(int(choice) for choice in self.config.channel_choices_for_stage(stage_idx))
            nearest = min(allowed, key=lambda candidate: abs(candidate - channel))
            resolved.append((int(nearest),))
        return tuple(resolved)

    def _resolve_stage_depth_choices(
        self,
        stage_depth: Optional[int | Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        resolved: list[tuple[int, ...]] = []
        if stage_depth is None:
            for stage_idx in range(self.stage_count):
                resolved.append(tuple(int(choice) for choice in self.config.depth_choices_for_stage(stage_idx)))
            return tuple(resolved)

        if isinstance(stage_depth, Sequence) and not isinstance(stage_depth, (str, bytes)):
            requested = [int(depth) for depth in stage_depth]
            if len(requested) != self.stage_count:
                raise ValueError(
                    f"stage_depth length {len(requested)} != stage_count {self.stage_count}"
                )
        else:
            requested = [int(stage_depth)] * self.stage_count

        for stage_idx, depth in enumerate(requested):
            allowed = tuple(int(choice) for choice in self.config.depth_choices_for_stage(stage_idx))
            nearest = min(allowed, key=lambda candidate: abs(candidate - depth))
            resolved.append((int(nearest),))
        return tuple(resolved)

    def _build_stages(self) -> None:
        current_resolution = max(1, math.ceil(self.config.image_size / self.config.stem_stride))
        current_resolution = max(
            1,
            math.ceil(current_resolution / self.config.post_stem_downsample_stride),
        )
        prev_channel_choices: tuple[int, ...] = (int(self.config.stem_channels),)

        for stage_idx in range(self.stage_count):
            stage = SearchableStage(
                search_space=self.search_space,
                stage_idx=stage_idx,
                input_channel_choices=prev_channel_choices,
                output_channel_choices=self.stage_channel_choices[stage_idx],
                depth_choices=self.stage_depth_choices[stage_idx],
                input_resolution=current_resolution,
                cost_estimator=self.cost_estimator,
                temperature=self.temperature,
            )
            self.stages.append(stage)
            self._mixed_ops.extend(stage._mixed_ops)
            prev_channel_choices = self.stage_channel_choices[stage_idx]
            current_resolution = stage.output_resolution

    @property
    def mixed_ops(self) -> list[MixedOp]:
        return self._mixed_ops

    def set_forward_mode(self, mode: str) -> None:
        for stage in self.stages:
            stage.set_mode(mode)

    def sample_single_paths(self, rng: Random) -> None:
        for stage in self.stages:
            stage.sample_single(rng)

    def sample_pair_paths(self, rng: Random) -> None:
        for stage in self.stages:
            stage.sample_pair(rng)

    def activate_argmax_paths(self) -> None:
        for stage in self.stages:
            stage.activate_argmax()

    def expected_hardware_metrics(self) -> dict[str, torch.Tensor]:
        if not self.stages:
            device = next(self.parameters()).device
            zero = torch.zeros((), device=device)
            return {
                "latency_ms": zero,
                "dsp": zero,
                "bram": zero,
                "lut": zero,
                "energy_proxy_mj": zero,
            }

        device = next(self.parameters()).device
        totals = {
            "latency_ms": torch.zeros((), device=device),
            "dsp": torch.zeros((), device=device),
            "bram": torch.zeros((), device=device),
            "lut": torch.zeros((), device=device),
        }

        prev_channel_probs = torch.ones(1, device=device)
        for stage in self.stages:
            stage_totals = stage.expected_hardware_metrics(prev_channel_probs)
            for key in totals:
                totals[key] = totals[key] + stage_totals[key]
            prev_channel_probs = stage.channel_gate.probabilities()

        power_proxy = 2.5 + 0.012 * totals["dsp"] + 0.018 * totals["bram"] + 0.0002 * totals["lut"]
        totals["energy_proxy_mj"] = totals["latency_ms"] * power_proxy
        return totals

    def arch_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for stage in self.stages:
            parameters.append(stage.channel_gate.alpha)
            parameters.append(stage.depth_gate.alpha)
            parameters.extend(mixed.alpha for mixed in stage._mixed_ops)
        return parameters

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if name.endswith("alpha"):
                continue
            yield param

    def extract_architecture(self) -> ArchitectureSpec:
        stages = tuple(stage.extract_stage_spec() for stage in self.stages)
        return ArchitectureSpec(
            input_channels=int(self.config.input_channels),
            stem_channels=int(self.config.stem_channels),
            stem_stride=int(self.config.stem_stride),
            post_stem_downsample_stride=int(self.config.post_stem_downsample_stride),
            stages=stages,
            head_conv_channels=self.config.head_conv_channels,
            head_channels=self.config.head_channels,
            num_classes=self.num_classes,
        )

    def choice_probabilities(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for stage in self.stages:
            rows.append(stage.channel_gate.probabilities().detach().cpu().tolist())
            rows.append(stage.depth_gate.probabilities().detach().cpu().tolist())
            for mixed in stage._mixed_ops:
                rows.append(mixed.probabilities().detach().cpu().tolist())
        return rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.post_stem_downsample is not None:
            x = self.post_stem_downsample(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x


__all__ = [
    "ChoiceGate",
    "MixedOp",
    "ProxylessSuperNet",
    "SearchableStage",
]
