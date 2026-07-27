from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from random import Random
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from hwnas_fpga.interfaces import SearchConstraints

if TYPE_CHECKING:
    from hwnas_fpga.hardware import FPGACostEstimator


DEFAULT_STAGE_STRIDES = (1, 2, 2, 2)
# Full capability set retained for legacy and auxiliary profiles. Canonical
# MobileNetV2-family search profiles below define the current formal mainline,
# which now excludes fused_mbconv and mixconv from the default operator set.
DEFAULT_OP_CHOICES = (
    "conv",
    "mbconv",
    "skip",
)

SONAR_OPS = {"mixconv", "mixconv_v2", "denoise", "edge"}

# v1 mainline search keeps mbconv as the primary block operator and
# preferentially removes irregular sonar-heavy operators first.
HIGH_LUT_OPS = {"mixconv", "mixconv_v2", "edge"}
HIGH_DSP_OPS = {"conv"}
# dw_pw_conv remains available for historical lightweight profiles, but is no
# longer part of the formal MobileNetV2 mainline after operator screening.
LIGHTWEIGHT_OPS = {"skip", "dw_pw_conv", "denoise"}

FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    "small": {
        "stem_channels": 16,
        "stage_strides": (1, 2, 2, 2),
        "channel_choices": (16, 24, 32),
        "depth_choices": (1, 2),
        "kernel_choices": (3,),
        "expand_choices": (1, 2),
        "op_choices": ("dw_pw_conv", "skip"),
    },
    "mobile_anchor": {
        "stem_channels": 24,
        "post_stem_downsample_stride": 1,
        "stage_strides": (1, 2, 2, 2, 1, 2, 1),
        "stage_base_channels": (8, 12, 16, 16, 20, 24, 24),
        "width_multipliers": (0.75, 1.0),
        "stage_depth_choices": ((1,), (1, 2), (1, 2), (2,), (1, 2), (1, 2), (1,)),
        "kernel_choices": (3, 5),
        "expand_choices": (1, 2),
        "op_choices": ("conv", "mbconv", "skip"),
        "head_conv_channels": 320,
    },
    "accuracy_biased": {
        "stem_channels": 32,
        "post_stem_downsample_stride": 1,
        "stage_strides": (1, 2, 2, 2, 1, 2, 1),
        "stage_base_channels": (16, 24, 32, 64, 96, 160, 320),
        "width_multipliers": (1.0, 1.25),
        "stage_depth_choices": ((1, 2), (2, 3, 4), (3, 4), (4, 5), (3, 4), (3, 4), (1, 2)),
        "kernel_choices": (3, 5),
        "expand_choices": (3, 6),
        "op_choices": ("conv", "mbconv", "skip"),
        "head_conv_channels": 1280,
    },
    "lightweight_sonar": {
        "stem_channels": 24,
        "post_stem_downsample_stride": 2,
        "stage_strides": (2, 2, 2),
        "stage_base_channels": (24, 48, 96),
        "width_multipliers": (0.5, 0.75, 1.0),
        "stage_depth_choices": ((1, 2, 3), (2, 3, 4), (1, 2, 3)),
        "kernel_choices": (3,),
        "expand_choices": (1, 2, 4),
        "op_choices": ("dw_pw_conv", "skip"),
        "head_conv_channels": 1024,
    },
}


def _as_int_tuple(values: Union[Sequence[int], Sequence[str]]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _as_float_tuple(values: Sequence[Union[int, float, str]]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_nested_int_tuple(values: Sequence[Sequence[Union[int, str]]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in group) for group in values)


def _as_str_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _as_stage_block_choices(
    values: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[tuple["BlockSpec", ...], ...]:
    return tuple(
        tuple(BlockSpec.from_dict(item) for item in group)
        for group in values
    )


def _resolve_family_profile(name: Optional[str]) -> dict[str, Any]:
    if name is None:
        return {}
    normalized = str(name).strip()
    if not normalized:
        return {}
    if normalized not in FAMILY_PROFILES:
        raise ValueError(f"unsupported family profile: {normalized}")
    return dict(FAMILY_PROFILES[normalized])


def list_family_profiles() -> dict[str, dict[str, Any]]:
    return {name: dict(values) for name, values in FAMILY_PROFILES.items()}


def _weighted_choice(random: Random, values: Sequence[Any], weights: Sequence[float]) -> Any:
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0:
        return random.choice(tuple(values))

    threshold = random.random() * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            return value
    return values[-1]


@dataclass(frozen=True)
class SearchSpaceConfig:
    family_profile: Optional[str] = None
    input_channels: int = 1
    image_size: int = 224
    stem_channels: int = 16
    stem_stride: int = 2
    post_stem_downsample_stride: int = 1
    stage_strides: tuple[int, ...] = DEFAULT_STAGE_STRIDES
    stage_base_channels: Optional[tuple[int, ...]] = None
    width_multipliers: Optional[tuple[float, ...]] = None
    stage_channel_choices: Optional[tuple[tuple[int, ...], ...]] = None
    stage_depth_choices: Optional[tuple[tuple[int, ...], ...]] = None
    channel_choices: tuple[int, ...] = (16, 24, 32, 48, 64, 96)
    depth_choices: tuple[int, ...] = (1, 2, 3, 4)
    kernel_choices: tuple[int, ...] = (3, 5)
    expand_choices: tuple[int, ...] = (1, 2, 4)
    op_choices: tuple[str, ...] = DEFAULT_OP_CHOICES
    stage_block_choices: Optional[tuple[tuple["BlockSpec", ...], ...]] = None
    head_conv_channels: Optional[int] = None
    head_channels: Optional[int] = None
    num_classes: Optional[int] = None
    hardware_constraints: Optional[SearchConstraints] = None  # 硬件约束参数

    def __post_init__(self) -> None:
        if self.family_profile is not None and self.family_profile not in FAMILY_PROFILES:
            raise ValueError(f"unsupported family profile: {self.family_profile}")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.stem_channels <= 0:
            raise ValueError("stem_channels must be positive")
        if self.stem_stride <= 0:
            raise ValueError("stem_stride must be positive")
        if self.post_stem_downsample_stride <= 0:
            raise ValueError("post_stem_downsample_stride must be positive")
        if not self.stage_strides:
            raise ValueError("stage_strides must not be empty")
        if any(stride <= 0 for stride in self.stage_strides):
            raise ValueError("stage_strides must be positive")

        if self.stage_base_channels is not None:
            if len(self.stage_base_channels) != self.stage_count:
                raise ValueError("stage_base_channels length must match stage_strides")
            if any(channel <= 0 for channel in self.stage_base_channels):
                raise ValueError("stage_base_channels must be positive")

        if self.width_multipliers is not None:
            if not self.width_multipliers:
                raise ValueError("width_multipliers must not be empty")
            if any(multiplier <= 0 for multiplier in self.width_multipliers):
                raise ValueError("width_multipliers must be positive")

        derived_stage_channel_choices = self.stage_channel_choices
        if (
            derived_stage_channel_choices is None
            and self.stage_base_channels is not None
            and self.width_multipliers is not None
        ):
            derived_stage_channel_choices = tuple(
                tuple(
                    sorted(
                        {
                            max(1, int(round(base_channel * multiplier)))
                            for multiplier in self.width_multipliers
                        }
                    )
                )
                for base_channel in self.stage_base_channels
            )

        if derived_stage_channel_choices is not None:
            if len(derived_stage_channel_choices) != self.stage_count:
                raise ValueError("stage_channel_choices length must match stage_strides")
            normalized_stage_channel_choices = tuple(
                tuple(sorted({int(channel) for channel in choices}))
                for choices in derived_stage_channel_choices
            )
            if any(not choices for choices in normalized_stage_channel_choices):
                raise ValueError("stage_channel_choices must not contain empty groups")
            if any(channel <= 0 for choices in normalized_stage_channel_choices for channel in choices):
                raise ValueError("stage_channel_choices must be positive")
            object.__setattr__(self, "stage_channel_choices", normalized_stage_channel_choices)
            object.__setattr__(
                self,
                "channel_choices",
                tuple(sorted({channel for choices in normalized_stage_channel_choices for channel in choices})),
            )

        if self.stage_depth_choices is not None:
            if len(self.stage_depth_choices) != self.stage_count:
                raise ValueError("stage_depth_choices length must match stage_strides")
            normalized_stage_depth_choices = tuple(
                tuple(sorted({int(depth) for depth in choices}))
                for choices in self.stage_depth_choices
            )
            if any(not choices for choices in normalized_stage_depth_choices):
                raise ValueError("stage_depth_choices must not contain empty groups")
            if any(depth <= 0 for choices in normalized_stage_depth_choices for depth in choices):
                raise ValueError("stage_depth_choices must be positive")
            object.__setattr__(self, "stage_depth_choices", normalized_stage_depth_choices)
            object.__setattr__(
                self,
                "depth_choices",
                tuple(sorted({depth for choices in normalized_stage_depth_choices for depth in choices})),
            )

        if any(channel <= 0 for channel in self.channel_choices):
            raise ValueError("channel_choices must be positive")
        if any(depth <= 0 for depth in self.depth_choices):
            raise ValueError("depth_choices must be positive")
        if any(kernel <= 0 for kernel in self.kernel_choices):
            raise ValueError("kernel_choices must be positive")
        if any(expand <= 0 for expand in self.expand_choices):
            raise ValueError("expand_choices must be positive")
        if self.stage_block_choices is not None:
            if len(self.stage_block_choices) != self.stage_count:
                raise ValueError("stage_block_choices length must match stage_strides")
            normalized_stage_block_choices = tuple(
                tuple(
                    BlockSpec(
                        op=str(block.op),
                        kernel_size=int(block.kernel_size),
                        expand_ratio=int(block.expand_ratio),
                        stride=int(block.stride),
                    )
                    for block in choices
                )
                for choices in self.stage_block_choices
            )
            if any(not choices for choices in normalized_stage_block_choices):
                raise ValueError("stage_block_choices must not contain empty groups")
            object.__setattr__(self, "stage_block_choices", normalized_stage_block_choices)
        if self.head_conv_channels is not None and self.head_conv_channels <= 0:
            raise ValueError("head_conv_channels must be positive")
        # 移除op_choices限制，允许扩展新算子
        # unsupported = set(self.op_choices) - set(DEFAULT_OP_CHOICES)
        # if unsupported:
        #     raise ValueError(f"unsupported op choices: {sorted(unsupported)}")

    @property
    def stage_count(self) -> int:
        return len(self.stage_strides)

    def channel_choices_for_stage(self, stage_index: int) -> tuple[int, ...]:
        if self.stage_channel_choices is not None:
            return self.stage_channel_choices[stage_index]
        return self.channel_choices

    def depth_choices_for_stage(self, stage_index: int) -> tuple[int, ...]:
        if self.stage_depth_choices is not None:
            return self.stage_depth_choices[stage_index]
        return self.depth_choices

    def block_choices_for_stage(self, stage_index: int) -> Optional[tuple["BlockSpec", ...]]:
        if self.stage_block_choices is None:
            return None
        return self.stage_block_choices[stage_index]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpaceConfig":
        data = dict(payload)
        family_profile = data.get("family_profile")
        profile_defaults = _resolve_family_profile(family_profile)
        if family_profile is not None:
            profile_defaults["family_profile"] = str(family_profile)
        data = {**profile_defaults, **data}
        if "stages" in data and "stage_strides" not in data:
            stage_count = int(data["stages"])
            data["stage_strides"] = tuple([1] + [2] * (stage_count - 1))
        if "stem_stride" in data:
            data["stem_stride"] = int(data["stem_stride"])
        if "post_stem_downsample_stride" in data:
            data["post_stem_downsample_stride"] = int(data["post_stem_downsample_stride"])
        if "stage_strides" in data:
            data["stage_strides"] = _as_int_tuple(data["stage_strides"])
        if "stage_base_channels" in data:
            data["stage_base_channels"] = _as_int_tuple(data["stage_base_channels"])
        if "width_multipliers" in data:
            data["width_multipliers"] = _as_float_tuple(data["width_multipliers"])
        if "stage_channel_choices" in data:
            data["stage_channel_choices"] = _as_nested_int_tuple(data["stage_channel_choices"])
        if "stage_depth_choices" in data:
            data["stage_depth_choices"] = _as_nested_int_tuple(data["stage_depth_choices"])
        if "channel_choices" in data:
            data["channel_choices"] = _as_int_tuple(data["channel_choices"])
        if "depth_choices" in data:
            data["depth_choices"] = _as_int_tuple(data["depth_choices"])
        if "kernel_choices" in data:
            data["kernel_choices"] = _as_int_tuple(data["kernel_choices"])
        if "expand_choices" in data:
            data["expand_choices"] = _as_int_tuple(data["expand_choices"])
        if "op_choices" in data:
            data["op_choices"] = _as_str_tuple(data["op_choices"])
        if "stage_block_choices" in data and data["stage_block_choices"] is not None:
            data["stage_block_choices"] = _as_stage_block_choices(data["stage_block_choices"])
        if "head_conv_channels" in data and data["head_conv_channels"] is not None:
            data["head_conv_channels"] = int(data["head_conv_channels"])
        return cls(**data)


@dataclass(frozen=True)
class BlockSpec:
    op: str
    kernel_size: int = 3
    expand_ratio: int = 1
    stride: int = 1

    def to_dict(self) -> dict[str, Union[int, str]]:
        return {
            "op": self.op,
            "kernel_size": self.kernel_size,
            "expand_ratio": self.expand_ratio,
            "stride": self.stride,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BlockSpec":
        return cls(
            op=str(payload["op"]),
            kernel_size=int(payload.get("kernel_size", 3)),
            expand_ratio=int(payload.get("expand_ratio", 1)),
            stride=int(payload.get("stride", 1)),
        )


@dataclass(frozen=True)
class StageSpec:
    channels: int
    depth: int
    stride: int
    blocks: tuple[BlockSpec, ...]

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if len(self.blocks) != self.depth:
            raise ValueError("depth must match the number of blocks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "depth": self.depth,
            "stride": self.stride,
            "blocks": [block.to_dict() for block in self.blocks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageSpec":
        blocks = tuple(BlockSpec.from_dict(block) for block in payload["blocks"])
        return cls(
            channels=int(payload["channels"]),
            depth=int(payload.get("depth", len(blocks))),
            stride=int(payload["stride"]),
            blocks=blocks,
        )


@dataclass(frozen=True)
class ArchitectureSpec:
    input_channels: int
    stem_channels: int
    stages: tuple[StageSpec, ...]
    stem_stride: int = 2
    post_stem_downsample_stride: int = 1
    head_conv_channels: Optional[int] = None
    head_channels: Optional[int] = None
    num_classes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_channels": self.input_channels,
            "stem_channels": self.stem_channels,
            "stem_stride": self.stem_stride,
            "post_stem_downsample_stride": self.post_stem_downsample_stride,
            "head_conv_channels": self.head_conv_channels,
            "head_channels": self.head_channels,
            "num_classes": self.num_classes,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureSpec":
        return cls(
            input_channels=int(payload["input_channels"]),
            stem_channels=int(payload["stem_channels"]),
            stem_stride=int(payload.get("stem_stride", 2)),
            post_stem_downsample_stride=int(payload.get("post_stem_downsample_stride", 1)),
            stages=tuple(StageSpec.from_dict(stage) for stage in payload["stages"]),
            head_conv_channels=(
                None
                if payload.get("head_conv_channels") is None
                else int(payload["head_conv_channels"])
            ),
            head_channels=(
                None
                if payload.get("head_channels") is None
                else int(payload["head_channels"])
            ),
            num_classes=(
                None if payload.get("num_classes") is None else int(payload["num_classes"])
            ),
        )


@dataclass(frozen=True)
class ResolvedBlockSpec:
    stage_index: int
    block_index: int
    op: str
    kernel_size: int
    expand_ratio: int
    stride: int
    in_channels: int
    out_channels: int
    input_resolution: int
    output_resolution: int


class SearchSpace:
    def __init__(self, config: SearchSpaceConfig):
        self.config = config
        self._pruned = False  # 标记是否已进行预剪枝

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpace":
        return cls(SearchSpaceConfig.from_dict(payload))

    @staticmethod
    def _physical_constraints_config(
        constraints: Optional[SearchConstraints],
    ) -> dict[str, Any]:
        if constraints is None:
            return {}
        physical = getattr(constraints, "physical", None)
        if not isinstance(physical, dict) or not physical:
            return {}
        if physical.get("enabled") is False:
            return {}
        return dict(physical)

    @staticmethod
    def _block_choice_allowed_by_physical(
        block: BlockSpec,
        *,
        input_resolution: int,
        physical: Mapping[str, Any],
    ) -> bool:
        if block.op not in {"mbconv", "fused_mbconv"}:
            return True
        for rule in physical.get("early_expand_limits", ()) or ():
            if not isinstance(rule, Mapping):
                continue
            min_resolution = rule.get("min_input_resolution", rule.get("min_resolution"))
            max_expand_ratio = rule.get("max_expand_ratio")
            if min_resolution is None or max_expand_ratio is None:
                continue
            if (
                int(input_resolution) >= int(min_resolution)
                and int(block.expand_ratio) > int(max_expand_ratio)
            ):
                return False
        return True

    def pre_prune(self, cost_estimator: "FPGACostEstimator") -> "SearchSpace":
        """根据FPGA资源约束预判缩减搜索空间。

        策略:
        1. 用 baseline 架构估计最低资源消耗
        2. 若 baseline 已违反约束 → 剪掉大通道/深度/高消耗算子
        3. 返回新的剪枝后搜索空间
        """
        constraints = self.config.hardware_constraints
        if not constraints:
            pruned_space = SearchSpace(self.config)
            pruned_space._pruned = True
            return pruned_space

        print("[硬件驱动空间剪枝] 开始预剪枝...")
        print(f"  初始: channels={self.config.channel_choices}, "
              f"depths={self.config.depth_choices}, ops={self.config.op_choices}")

        # 1. 评估 baseline
        baseline = self.baseline_architecture()
        try:
            estimate = cost_estimator.estimate(baseline, self)
        except Exception as e:
            print(f"  [警告] Baseline 估计失败: {e}，跳过剪枝")
            pruned_space = SearchSpace(self.config)
            pruned_space._pruned = True
            return pruned_space

        effective_budgets = self._resolve_effective_budgets(cost_estimator)
        new_channels = self.config.channel_choices
        new_depths = self.config.depth_choices
        new_stage_channel_choices = self.config.stage_channel_choices
        new_stage_depth_choices = self.config.stage_depth_choices
        new_ops = self.config.op_choices
        new_kernels = self.config.kernel_choices
        new_expands = self.config.expand_choices
        new_stage_block_choices = self.config.stage_block_choices

        if estimate.violations:
            print(f"  Baseline 违反约束: {estimate.violations}")
        else:
            print("  Baseline 满足所有约束，执行板卡感知收缩")

        max_dsp = effective_budgets["max_dsp"]
        max_lut = effective_budgets["max_lut"]
        max_bram = effective_budgets["max_bram"]
        max_latency_ms = effective_budgets["max_latency_ms"]
        max_bw = effective_budgets["max_memory_bandwidth_gbps"]

        tight_board = any(
            (
                max_dsp is not None and max_dsp <= 256,
                max_lut is not None and max_lut <= 60_000,
                max_bram is not None and max_bram <= 200,
            )
        )
        ultra_tight_board = any(
            (
                max_dsp is not None and max_dsp <= 160,
                max_lut is not None and max_lut <= 40_000,
                max_bram is not None and max_bram <= 100,
            )
        )

        if max_dsp is not None:
            max_ch = 32 if tight_board else max(16, max_dsp // 3)
            filtered = tuple(c for c in new_channels if c <= max_ch)
            new_channels = filtered or (min(new_channels),)
            if tight_board:
                new_ops = tuple(op for op in new_ops if op not in HIGH_DSP_OPS)

        if tight_board or (max_lut is not None and max_lut < 80_000):
            filtered = tuple(op for op in new_ops if op not in HIGH_LUT_OPS)
            new_ops = filtered or tuple(sorted(LIGHTWEIGHT_OPS & set(new_ops)))

        if tight_board or (max_bw is not None and max_bw <= 4.0):
            filtered_kernels = tuple(kernel for kernel in new_kernels if kernel <= 3)
            new_kernels = filtered_kernels or (min(new_kernels),)
            filtered_expands = tuple(expand for expand in new_expands if expand <= 2)
            new_expands = filtered_expands or (min(new_expands),)

        if max_latency_ms is not None:
            max_d = 2 if max_latency_ms <= 50 else 3 if max_latency_ms < 100 else 4
            filtered = tuple(d for d in new_depths if d <= max_d)
            new_depths = filtered or (min(new_depths),)

        if tight_board:
            filtered = tuple(d for d in new_depths if d <= 2)
            new_depths = filtered or (min(new_depths),)

        if ultra_tight_board:
            new_channels = tuple(c for c in new_channels if c <= 24) or (min(new_channels),)
            new_depths = tuple(d for d in new_depths if d <= 2) or (min(new_depths),)
            lightweight_ops = tuple(op for op in new_ops if op in LIGHTWEIGHT_OPS)
            if any(op != "skip" for op in lightweight_ops):
                new_ops = lightweight_ops
            else:
                # Do not manufacture an operator the caller did not authorize.
                # A skip-only result is illegal at stride/channel transitions,
                # so retain the first original compute family as a legal
                # fallback even if its hardware estimate remains infeasible.
                compute_fallback = tuple(
                    op for op in self.config.op_choices if op != "skip"
                )
                new_ops = compute_fallback[:1] + (
                    ("skip",) if "skip" in self.config.op_choices else ()
                )
            new_kernels = tuple(kernel for kernel in new_kernels if kernel == 3) or (min(new_kernels),)
            new_expands = tuple(expand for expand in new_expands if expand == 1) or (1,)

        # 3. 兜底保证非空
        if not new_ops:
            new_ops = self.config.op_choices
        if not new_channels:
            new_channels = (min(self.config.channel_choices),)
        if not new_depths:
            new_depths = (min(self.config.depth_choices),)
        if not new_kernels:
            new_kernels = (min(self.config.kernel_choices),)
        if not new_expands:
            new_expands = (min(self.config.expand_choices),)

        if self.config.stage_channel_choices is not None:
            allowed_channels = set(new_channels)
            new_stage_channel_choices = tuple(
                tuple(channel for channel in stage_choices if channel in allowed_channels) or (min(stage_choices),)
                for stage_choices in self.config.stage_channel_choices
            )
        if self.config.stage_depth_choices is not None:
            allowed_depths = set(new_depths)
            new_stage_depth_choices = tuple(
                tuple(depth for depth in stage_choices if depth in allowed_depths) or (min(stage_choices),)
                for stage_choices in self.config.stage_depth_choices
            )

        physical = self._physical_constraints_config(constraints)
        if self.config.stage_block_choices is not None and physical:
            filtered_stage_block_choices: list[tuple[BlockSpec, ...]] = []
            current_resolution = max(1, ceil(self.config.image_size / self.config.stem_stride))
            current_resolution = max(
                1,
                ceil(current_resolution / self.config.post_stem_downsample_stride),
            )
            for stage_index, choices in enumerate(self.config.stage_block_choices):
                filtered = tuple(
                    block
                    for block in choices
                    if self._block_choice_allowed_by_physical(
                        block,
                        input_resolution=current_resolution,
                        physical=physical,
                    )
                )
                filtered_stage_block_choices.append(filtered or choices)
                current_resolution = max(
                    1,
                    ceil(current_resolution / self.config.stage_strides[stage_index]),
                )
            new_stage_block_choices = tuple(filtered_stage_block_choices)

        if new_stage_block_choices is not None:
            block_ops = {
                str(block.op)
                for choices in new_stage_block_choices
                for block in choices
            }
            block_kernels = {
                int(block.kernel_size)
                for choices in new_stage_block_choices
                for block in choices
            }
            block_expands = {
                int(block.expand_ratio)
                for choices in new_stage_block_choices
                for block in choices
            }
            new_ops = tuple(sorted(set(new_ops) | block_ops))
            new_kernels = tuple(sorted(set(new_kernels) | block_kernels))
            new_expands = tuple(sorted(set(new_expands) | block_expands))

        new_config = replace(
            self.config,
            stage_channel_choices=new_stage_channel_choices,
            stage_depth_choices=new_stage_depth_choices,
            channel_choices=new_channels,
            depth_choices=new_depths,
            kernel_choices=new_kernels,
            expand_choices=new_expands,
            op_choices=new_ops,
            stage_block_choices=new_stage_block_choices,
        )

        reduction = self._calculate_reduction_ratio(self.config, new_config)
        print(f"  剪枝后: channels={new_config.channel_choices}, "
              f"depths={new_config.depth_choices}, kernels={new_config.kernel_choices}, "
              f"expands={new_config.expand_choices}, ops={new_config.op_choices}")
        print(f"  搜索空间减少了约 {reduction:.1f}%")

        pruned_space = SearchSpace(new_config)
        pruned_space._pruned = True
        return pruned_space
    
    def _calculate_reduction_ratio(self, old_config: SearchSpaceConfig, new_config: SearchSpaceConfig) -> float:
        """计算搜索空间缩减比例"""
        def size(config: SearchSpaceConfig) -> int:
            channel_factor = 1
            if config.stage_channel_choices is not None:
                for choices in config.stage_channel_choices:
                    channel_factor *= max(1, len(choices))
            else:
                channel_factor = max(1, len(config.channel_choices)) ** config.stage_count

            depth_factor = 1
            if config.stage_depth_choices is not None:
                for choices in config.stage_depth_choices:
                    depth_factor *= max(1, len(choices))
            else:
                depth_factor = max(1, len(config.depth_choices)) ** config.stage_count

            if config.stage_block_choices is not None:
                block_factor = 1
                for choices in config.stage_block_choices:
                    block_factor *= max(1, len(choices))
            else:
                block_factor = (
                    max(1, len(config.op_choices))
                    * max(1, len(config.kernel_choices))
                    * max(1, len(config.expand_choices))
                ) ** config.stage_count
            return channel_factor * depth_factor * block_factor

        old_size = size(old_config)
        new_size = size(new_config)

        if old_size == 0:
            return 0.0
        return ((old_size - new_size) / old_size) * 100
    
    def is_pruned(self) -> bool:
        """检查搜索空间是否已进行过预剪枝"""
        return self._pruned

    def concrete_block_choices_for_stage(
        self,
        *,
        stage_index: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> Optional[tuple[BlockSpec, ...]]:
        choices = self.config.block_choices_for_stage(stage_index)
        if choices is None:
            return None

        concrete: list[BlockSpec] = []
        for choice in choices:
            if choice.op == "skip" and (stride != 1 or in_channels != out_channels):
                continue
            concrete.append(
                BlockSpec(
                    op=choice.op,
                    kernel_size=1 if choice.op == "skip" else int(choice.kernel_size),
                    expand_ratio=1 if choice.op == "skip" else int(choice.expand_ratio),
                    stride=int(stride),
                )
            )

        if not concrete:
            raise ValueError(
                f"no legal stage_block_choices for stage={stage_index}, "
                f"in_channels={in_channels}, out_channels={out_channels}, stride={stride}"
            )
        return tuple(concrete)

    def baseline_architecture(self) -> ArchitectureSpec:
        stages: list[StageSpec] = []
        current_channels = self.config.stem_channels
        base_kernel = min(self.config.kernel_choices)
        base_expand = min(self.config.expand_choices)

        for stage_index, stride in enumerate(self.config.stage_strides):
            stage_channel_choices = self.config.channel_choices_for_stage(stage_index)
            stage_depth_choices = self.config.depth_choices_for_stage(stage_index)
            if self.config.stage_base_channels is not None:
                preferred_channels = self.config.stage_base_channels[stage_index]
                if preferred_channels in stage_channel_choices:
                    channels = preferred_channels
                else:
                    smaller_or_equal = [value for value in stage_channel_choices if value <= preferred_channels]
                    if smaller_or_equal:
                        channels = max(smaller_or_equal)
                    else:
                        channels = min(stage_channel_choices)
            else:
                channels = min(stage_channel_choices)
            base_depth = min(stage_depth_choices)
            blocks: list[BlockSpec] = []
            for block_index in range(base_depth):
                block_stride = stride if block_index == 0 else 1
                in_channels = current_channels if block_index == 0 else channels
                stage_block_choices = self.concrete_block_choices_for_stage(
                    stage_index=stage_index,
                    in_channels=in_channels,
                    out_channels=channels,
                    stride=block_stride,
                )
                if stage_block_choices is not None:
                    blocks.append(stage_block_choices[0])
                else:
                    valid_ops = self.available_ops(
                        in_channels=in_channels,
                        out_channels=channels,
                        stride=block_stride,
                    )
                    preferred_op = "dw_pw_conv" if "dw_pw_conv" in valid_ops else valid_ops[0]
                    blocks.append(
                        self._build_block(
                            op=preferred_op,
                            stride=block_stride,
                            kernel_size=base_kernel,
                            expand_ratio=base_expand,
                        )
                    )
            stages.append(
                StageSpec(
                    channels=channels,
                    depth=base_depth,
                    stride=stride,
                    blocks=tuple(blocks),
                )
            )
            current_channels = channels

        return ArchitectureSpec(
            input_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            stem_stride=self.config.stem_stride,
            post_stem_downsample_stride=self.config.post_stem_downsample_stride,
            stages=tuple(stages),
            head_conv_channels=self.config.head_conv_channels,
            head_channels=self.config.head_channels,
            num_classes=self.config.num_classes,
        )

    def sample(
        self,
        seed: Optional[int] = None,
        rng: Optional[Random] = None,
        *,
        cost_estimator: Optional["FPGACostEstimator"] = None,
        apply_pruning: bool = True,
        require_feasible: bool = False,
        max_feasible_attempts: int = 32,
        prefer_lightweight: bool = False,
    ) -> ArchitectureSpec:
        """
        从搜索空间采样一个架构。
        
        如果提供了cost_estimator且apply_pruning为True，将自动应用预剪枝。
        
        Args:
            seed: 随机种子
            rng: 随机数生成器
            cost_estimator: FPGA成本估算器（用于预剪枝）
            apply_pruning: 是否应用预剪枝
            
        Returns:
            采样的架构规范
        """
        # 如果需要且提供cost_estimator，先进行预剪枝
        sample_space = self
        if apply_pruning and cost_estimator and not self._pruned:
            sample_space = self.pre_prune(cost_estimator)

        if require_feasible and cost_estimator is not None:
            return sample_space._sample_feasible_impl(
                cost_estimator=cost_estimator,
                seed=seed,
                rng=rng,
                max_attempts=max_feasible_attempts,
                prefer_lightweight=prefer_lightweight,
            )

        return sample_space._sample_impl(seed, rng, prefer_lightweight=prefer_lightweight)
    
    def _sample_impl(
        self,
        seed: Optional[int] = None,
        rng: Optional[Random] = None,
        *,
        prefer_lightweight: bool = False,
    ) -> ArchitectureSpec:
        random = rng or Random(seed)
        stages: list[StageSpec] = []
        current_channels = self.config.stem_channels

        for stage_index, stride in enumerate(self.config.stage_strides):
            channels = self._choose_channel(
                random,
                stage_index=stage_index,
                prefer_lightweight=prefer_lightweight,
            )
            depth = self._choose_depth(
                random,
                stage_index=stage_index,
                prefer_lightweight=prefer_lightweight,
            )
            blocks: list[BlockSpec] = []
            for block_index in range(depth):
                block_stride = stride if block_index == 0 else 1
                in_channels = current_channels if block_index == 0 else channels
                stage_block_choices = self.concrete_block_choices_for_stage(
                    stage_index=stage_index,
                    in_channels=in_channels,
                    out_channels=channels,
                    stride=block_stride,
                )
                if stage_block_choices is not None:
                    blocks.append(random.choice(stage_block_choices))
                    continue
                valid_ops = self.available_ops(
                    in_channels=in_channels,
                    out_channels=channels,
                    stride=block_stride,
                )
                op = self._choose_op(random, valid_ops, prefer_lightweight=prefer_lightweight)
                kernel_size = (
                    1 if op == "skip"
                    else self._choose_kernel(random, prefer_lightweight=prefer_lightweight)
                )
                # 声呐专用算子和基础算子不需要expand_ratio
                if op in {"mbconv", "fused_mbconv"}:
                    expand_ratio = self._choose_expand(random, prefer_lightweight=prefer_lightweight)
                else:
                    expand_ratio = 1
                blocks.append(
                    self._build_block(
                        op=op,
                        stride=block_stride,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                    )
                )
            stages.append(
                StageSpec(
                    channels=channels,
                    depth=depth,
                    stride=stride,
                    blocks=tuple(blocks),
                )
            )
            current_channels = channels

        return ArchitectureSpec(
            input_channels=self.config.input_channels,
            stem_channels=self.config.stem_channels,
            stem_stride=self.config.stem_stride,
            post_stem_downsample_stride=self.config.post_stem_downsample_stride,
            stages=tuple(stages),
            head_conv_channels=self.config.head_conv_channels,
            head_channels=self.config.head_channels,
            num_classes=self.config.num_classes,
        )

    def _sample_feasible_impl(
        self,
        *,
        cost_estimator: "FPGACostEstimator",
        seed: Optional[int] = None,
        rng: Optional[Random] = None,
        max_attempts: int = 32,
        prefer_lightweight: bool = True,
    ) -> ArchitectureSpec:
        random = rng or Random(seed)
        best_candidate: Optional[ArchitectureSpec] = None
        best_violation_score: Optional[tuple[int, float]] = None

        for _ in range(max(1, max_attempts)):
            architecture = self._sample_impl(
                seed=None,
                rng=random,
                prefer_lightweight=prefer_lightweight,
            )
            estimate = cost_estimator.estimate(architecture, self)
            if not estimate.violations:
                return architecture

            violation_score = (
                len(estimate.violations),
                estimate.latency_ms + estimate.energy_mj + estimate.resource_dsp + estimate.resource_lut,
            )
            if best_violation_score is None or violation_score < best_violation_score:
                best_violation_score = violation_score
                best_candidate = architecture

        baseline = self.baseline_architecture()
        baseline_estimate = cost_estimator.estimate(baseline, self)
        if not baseline_estimate.violations:
            return baseline
        baseline_violation_score = (
            len(baseline_estimate.violations),
            baseline_estimate.latency_ms
            + baseline_estimate.energy_mj
            + baseline_estimate.resource_dsp
            + baseline_estimate.resource_lut,
        )
        if best_violation_score is None or baseline_violation_score <= best_violation_score:
            return baseline
        if best_candidate is not None:
            return best_candidate
        return baseline

    def available_ops(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> tuple[str, ...]:
        valid_ops: list[str] = []
        for op in self.config.op_choices:
            if op == "skip" and (stride != 1 or in_channels != out_channels):
                continue
            valid_ops.append(op)
        if not valid_ops:
            raise ValueError("no legal ops available for the requested block placement")
        return tuple(valid_ops)

    def validate(self, architecture: ArchitectureSpec) -> list[str]:
        errors: list[str] = []
        if architecture.input_channels != self.config.input_channels:
            errors.append(
                "architecture input_channels does not match the search space configuration"
            )
        if architecture.stem_channels != self.config.stem_channels:
            errors.append(
                "architecture stem_channels does not match the search space configuration"
            )
        if architecture.stem_stride != self.config.stem_stride:
            errors.append(
                "architecture stem_stride does not match the search space configuration"
            )
        if architecture.post_stem_downsample_stride != self.config.post_stem_downsample_stride:
            errors.append(
                "architecture post_stem_downsample_stride does not match the search space configuration"
            )
        if architecture.head_conv_channels != self.config.head_conv_channels:
            errors.append(
                "architecture head_conv_channels does not match the search space configuration"
            )
        if len(architecture.stages) != self.config.stage_count:
            errors.append("architecture stage count does not match stage_strides")
            return errors

        current_channels = architecture.stem_channels
        current_resolution = max(1, ceil(self.config.image_size / architecture.stem_stride))
        current_resolution = max(
            1,
            ceil(current_resolution / architecture.post_stem_downsample_stride),
        )
        for stage_index, (stage, expected_stride) in enumerate(
            zip(architecture.stages, self.config.stage_strides)
        ):
            if stage.channels not in self.config.channel_choices_for_stage(stage_index):
                errors.append(f"stage {stage_index} uses unsupported channels={stage.channels}")
            if stage.depth not in self.config.depth_choices_for_stage(stage_index):
                errors.append(f"stage {stage_index} uses unsupported depth={stage.depth}")
            if stage.stride != expected_stride:
                errors.append(
                    f"stage {stage_index} stride must be {expected_stride}, got {stage.stride}"
                )
            if len(stage.blocks) != stage.depth:
                errors.append(
                    f"stage {stage_index} depth={stage.depth} does not match block count"
                )
                continue

            for block_index, block in enumerate(stage.blocks):
                block_in_channels = current_channels if block_index == 0 else stage.channels
                block_stride = stage.stride if block_index == 0 else 1
                stage_block_choices = self.concrete_block_choices_for_stage(
                    stage_index=stage_index,
                    in_channels=block_in_channels,
                    out_channels=stage.channels,
                    stride=block_stride,
                )
                if stage_block_choices is not None:
                    allowed_keys = {
                        (choice.op, choice.kernel_size, choice.expand_ratio, choice.stride)
                        for choice in stage_block_choices
                    }
                    if (block.op, block.kernel_size, block.expand_ratio, block.stride) not in allowed_keys:
                        errors.append(
                            f"stage {stage_index} block {block_index} block choice is unsupported"
                        )
                    current_channels = stage.channels
                    current_resolution = max(1, ceil(current_resolution / block.stride))
                    continue
                expected_ops = self.available_ops(
                    in_channels=block_in_channels,
                    out_channels=stage.channels,
                    stride=block_stride,
                )
                if block.op not in expected_ops:
                    errors.append(
                        f"stage {stage_index} block {block_index} op={block.op} is illegal"
                    )
                if block.stride != block_stride:
                    errors.append(
                        f"stage {stage_index} block {block_index} stride must be {block_stride}"
                    )
                if block.op == "skip":
                    if block.kernel_size != 1:
                        errors.append(
                            f"stage {stage_index} block {block_index} skip must use kernel_size=1"
                        )
                    if block.expand_ratio != 1:
                        errors.append(
                            f"stage {stage_index} block {block_index} skip must use expand_ratio=1"
                        )
                else:
                    if block.kernel_size not in self.config.kernel_choices:
                        errors.append(
                            f"stage {stage_index} block {block_index} uses unsupported kernel"
                        )
                    if block.op in {"conv", "dw_pw_conv"} and block.expand_ratio != 1:
                        errors.append(
                            f"stage {stage_index} block {block_index} expand_ratio must be 1"
                        )
                    if block.op in SONAR_OPS and block.expand_ratio != 1:
                        errors.append(
                            f"stage {stage_index} block {block_index} "
                            f"sonar op '{block.op}' must use expand_ratio=1"
                        )
                    if (
                        block.op in {"mbconv", "fused_mbconv"}
                        and block.expand_ratio not in self.config.expand_choices
                    ):
                        errors.append(
                            f"stage {stage_index} block {block_index} uses unsupported expand_ratio"
                        )

                current_channels = stage.channels
                current_resolution = max(1, ceil(current_resolution / block.stride))

        if current_resolution <= 0:
            errors.append("architecture collapses the feature map resolution")
        return errors

    def is_valid(self, architecture: ArchitectureSpec) -> bool:
        return not self.validate(architecture)

    def resolve_blocks(self, architecture: ArchitectureSpec) -> tuple[ResolvedBlockSpec, ...]:
        errors = self.validate(architecture)
        if errors:
            raise ValueError("; ".join(errors))

        resolved: list[ResolvedBlockSpec] = []
        current_channels = architecture.stem_channels
        current_resolution = max(1, ceil(self.config.image_size / architecture.stem_stride))
        current_resolution = max(
            1,
            ceil(current_resolution / architecture.post_stem_downsample_stride),
        )
        for stage_index, stage in enumerate(architecture.stages):
            for block_index, block in enumerate(stage.blocks):
                block_in_channels = current_channels if block_index == 0 else stage.channels
                output_resolution = max(1, ceil(current_resolution / block.stride))
                resolved.append(
                    ResolvedBlockSpec(
                        stage_index=stage_index,
                        block_index=block_index,
                        op=block.op,
                        kernel_size=block.kernel_size,
                        expand_ratio=block.expand_ratio,
                        stride=block.stride,
                        in_channels=block_in_channels,
                        out_channels=stage.channels,
                        input_resolution=current_resolution,
                        output_resolution=output_resolution,
                    )
                )
                current_channels = stage.channels
                current_resolution = output_resolution
        return tuple(resolved)

    def _build_block(
        self,
        *,
        op: str,
        stride: int,
        kernel_size: int,
        expand_ratio: int,
    ) -> BlockSpec:
        if op == "skip":
            return BlockSpec(op=op, kernel_size=1, expand_ratio=1, stride=stride)
        if op in {"conv", "dw_pw_conv"} or op in SONAR_OPS:
            return BlockSpec(op=op, kernel_size=kernel_size, expand_ratio=1, stride=stride)
        return BlockSpec(
            op=op,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            stride=stride,
        )

    def _resolve_effective_budgets(self, cost_estimator: "FPGACostEstimator") -> dict[str, Optional[float]]:
        constraints = self.config.hardware_constraints
        hardware_spec = cost_estimator.hardware_spec

        def _minimum(*values: Optional[float]) -> Optional[float]:
            filtered = [value for value in values if value is not None]
            if not filtered:
                return None
            return min(filtered)

        return {
            "max_latency_ms": getattr(constraints, "max_latency_ms", None),
            "max_dsp": _minimum(getattr(constraints, "max_dsp", None), hardware_spec.max_dsp),
            "max_bram": _minimum(getattr(constraints, "max_bram", None), hardware_spec.max_bram),
            "max_lut": _minimum(getattr(constraints, "max_lut", None), hardware_spec.max_lut),
            "max_memory_bandwidth_gbps": _minimum(
                getattr(constraints, "max_memory_bandwidth_gbps", None),
                hardware_spec.memory_bandwidth_gbps,
            ),
        }

    def _choose_channel(
        self,
        random: Random,
        *,
        stage_index: int,
        prefer_lightweight: bool,
    ) -> int:
        channel_choices = self.config.channel_choices_for_stage(stage_index)
        if not prefer_lightweight:
            return random.choice(channel_choices)
        weights = [1.0 / max(1, channel) for channel in channel_choices]
        return _weighted_choice(random, channel_choices, weights)

    def _choose_depth(
        self,
        random: Random,
        *,
        stage_index: int,
        prefer_lightweight: bool,
    ) -> int:
        depth_choices = self.config.depth_choices_for_stage(stage_index)
        if not prefer_lightweight:
            return random.choice(depth_choices)
        weights = [1.0 / max(1, depth) for depth in depth_choices]
        return _weighted_choice(random, depth_choices, weights)

    def _choose_kernel(self, random: Random, *, prefer_lightweight: bool) -> int:
        if not prefer_lightweight:
            return random.choice(self.config.kernel_choices)
        weights = [1.0 / max(1, kernel) for kernel in self.config.kernel_choices]
        return _weighted_choice(random, self.config.kernel_choices, weights)

    def _choose_expand(self, random: Random, *, prefer_lightweight: bool) -> int:
        if not prefer_lightweight:
            return random.choice(self.config.expand_choices)
        weights = [1.0 / max(1, expand) for expand in self.config.expand_choices]
        return _weighted_choice(random, self.config.expand_choices, weights)

    def _choose_op(
        self,
        random: Random,
        valid_ops: Sequence[str],
        *,
        prefer_lightweight: bool,
    ) -> str:
        if not prefer_lightweight:
            return random.choice(tuple(valid_ops))
        weights = []
        for op in valid_ops:
            if op in LIGHTWEIGHT_OPS:
                weights.append(4.0)
            elif op in SONAR_OPS:
                weights.append(1.5)
            elif op in HIGH_DSP_OPS or op in HIGH_LUT_OPS:
                weights.append(0.75)
            else:
                weights.append(1.0)
        return _weighted_choice(random, tuple(valid_ops), weights)
