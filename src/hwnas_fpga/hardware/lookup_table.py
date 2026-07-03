"""LUT-based hardware cost estimation.

This module implements a lookup table (LUT) based hardware cost estimator,
inspired by FBNet's LUT architecture (reference/FBNet/mobile_cv/lut/lib/).

Key design principles:
- Use operator-level profiling to build LUT tables
- Support pickle serialization for fast loading/saving
- Enable fallback to analytical model when LUT misses
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# HLS profiling kernels may use finer-grained names than the current NAS search
# space. Normalize them so a LUT built from HLS manifests can still satisfy the
# existing query path without requiring every caller to know both vocabularies.
LUT_OP_ALIASES: Dict[str, str] = {
    "stem_conv_k3_s2": "conv",
    "pw_conv": "conv",
    "conv_bn_relu6": "conv",
    "inverted_residual": "mbconv",
}


def canonicalize_lut_op_name(op: str) -> str:
    return LUT_OP_ALIASES.get(str(op).strip(), str(op).strip())


# ============================================================================
# Operator Specifications (类似 FBNet 的 OpBase/OpProperty)
# ============================================================================


@dataclass(frozen=True)
class OpSpec:
    """算子规格定义

    用于唯一标识一个算子实例，作为 LUT 的查询键。

    Attributes:
        op: 算子类型 (conv, dw_conv, mbconv, fused_mbconv, skip, etc.)
            允许更细粒度的 HLS kernel 名称；导入时会归一化到查询口径。
        kernel_size: 卷积核大小
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 步长
        groups: 分组数 (用于 depthwise conv)
        expand_ratio: 扩展比 (用于 MBConv)
        input_resolution: 输入分辨率 (H, W)
    """

    op: str
    kernel_size: int
    in_channels: int
    out_channels: int
    stride: int = 1
    groups: int = 1
    expand_ratio: int = 1
    input_resolution: Tuple[int, int] = (224, 224)
    bitwidth: int = 8
    input_parallelism: int = 1
    output_parallelism: int = 1
    unroll_factor: int = 1
    target_clock_mhz: Optional[float] = None
    pack_ch: Optional[int] = None
    ch_block: Optional[int] = None
    target_ii: Optional[int] = None
    tile_order: Optional[str] = None
    stream_order: Optional[str] = None
    dsp_pack: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "op", canonicalize_lut_op_name(self.op))
        # 确保关键参数是整数（用于哈希）
        if isinstance(self.input_resolution, (list, tuple)):
            object.__setattr__(self, "input_resolution", tuple(self.input_resolution))
        object.__setattr__(self, "bitwidth", int(self.bitwidth))
        object.__setattr__(self, "input_parallelism", int(self.input_parallelism))
        object.__setattr__(self, "output_parallelism", int(self.output_parallelism))
        object.__setattr__(self, "unroll_factor", int(self.unroll_factor))
        if self.target_clock_mhz is not None:
            object.__setattr__(self, "target_clock_mhz", float(self.target_clock_mhz))
        if self.pack_ch is not None:
            object.__setattr__(self, "pack_ch", int(self.pack_ch))
        if self.ch_block is not None:
            object.__setattr__(self, "ch_block", int(self.ch_block))
        if self.target_ii is not None:
            object.__setattr__(self, "target_ii", int(self.target_ii))
        if self.tile_order is not None:
            object.__setattr__(self, "tile_order", str(self.tile_order))
        if self.stream_order is not None:
            object.__setattr__(self, "stream_order", str(self.stream_order))
        if self.dsp_pack is not None:
            object.__setattr__(self, "dsp_pack", str(self.dsp_pack))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        payload = {
            "op": self.op,
            "kernel_size": self.kernel_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "stride": self.stride,
            "groups": self.groups,
            "expand_ratio": self.expand_ratio,
            "input_resolution": self.input_resolution,
            "bitwidth": self.bitwidth,
            "input_parallelism": self.input_parallelism,
            "output_parallelism": self.output_parallelism,
            "unroll_factor": self.unroll_factor,
            "target_clock_mhz": self.target_clock_mhz,
        }
        for key in ("pack_ch", "ch_block", "target_ii", "tile_order", "stream_order", "dsp_pack"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpSpec":
        """从字典创建"""
        return cls(
            op=data["op"],
            kernel_size=data["kernel_size"],
            in_channels=data["in_channels"],
            out_channels=data["out_channels"],
            stride=data.get("stride", 1),
            groups=data.get("groups", 1),
            expand_ratio=data.get("expand_ratio", 1),
            input_resolution=tuple(data.get("input_resolution", (224, 224))),
            bitwidth=data.get("bitwidth", 8),
            input_parallelism=data.get("input_parallelism", 1),
            output_parallelism=data.get("output_parallelism", 1),
            unroll_factor=data.get("unroll_factor", 1),
            target_clock_mhz=data.get("target_clock_mhz"),
            pack_ch=data.get("pack_ch"),
            ch_block=data.get("ch_block"),
            target_ii=data.get("target_ii"),
            tile_order=data.get("tile_order"),
            stream_order=data.get("stream_order"),
            dsp_pack=data.get("dsp_pack"),
        )

    def shape_signature(self) -> Tuple[Any, ...]:
        return (
            self.op,
            self.kernel_size,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.groups,
            self.expand_ratio,
            self.input_resolution,
        )

    def implementation_signature(self) -> Tuple[Any, ...]:
        return (
            self.bitwidth,
            self.input_parallelism,
            self.output_parallelism,
            self.unroll_factor,
            self.target_clock_mhz,
            self.pack_ch,
            self.ch_block,
            self.target_ii,
            self.tile_order,
            self.stream_order,
            self.dsp_pack,
        )

    def __hash__(self) -> int:
        return hash((*self.shape_signature(), *self.implementation_signature()))


# ============================================================================
# LUT Entry Definition (类似 FBNet 的 LutItem)
# ============================================================================


@dataclass(frozen=True)
class LutEntry:
    """LUT 条目

    每个条目包含一个算子的硬件代价信息。

    Attributes:
        op_spec: 算子规格
        latency_ms: 延迟（毫秒）
        cycles: 时钟周期数
        dsp: DSP 使用量
        bram: BRAM 使用量（块数）
        lut: LUT 使用量
        power_w: 功耗（瓦特）
        energy_mj: 能耗（毫焦耳）
    """

    op_spec: OpSpec
    latency_ms: float = 0.0
    cycles: int = 0
    dsp: int = 0
    bram: int = 0
    lut: int = 0
    power_w: Optional[float] = None
    energy_mj: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "op_spec": self.op_spec.to_dict(),
            "latency_ms": self.latency_ms,
            "cycles": self.cycles,
            "dsp": self.dsp,
            "bram": self.bram,
            "lut": self.lut,
            "power_w": self.power_w,
            "energy_mj": self.energy_mj,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LutEntry":
        """从字典创建"""
        return cls(
            op_spec=OpSpec.from_dict(data["op_spec"]),
            latency_ms=data["latency_ms"],
            cycles=data["cycles"],
            dsp=data["dsp"],
            bram=data["bram"],
            lut=data["lut"],
            power_w=data["power_w"],
            energy_mj=data["energy_mj"],
        )


@dataclass(frozen=True)
class FormalLutStatusEntry:
    """Formal LUT status entry used by defer-aware NAS experiments."""

    case_name: str
    status: str
    op_type: str
    lookup_op: str
    defer_reason: Optional[str] = None
    root_cause_bucket: Optional[str] = None
    op_spec: OpSpec | None = None
    board_cycles: Optional[int] = None
    board_latency_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormalLutStatusEntry":
        return cls(
            case_name=str(data["case_name"]),
            status=str(data["status"]),
            op_type=str(data.get("op_type", data.get("lookup_op", ""))),
            lookup_op=str(data.get("lookup_op", data.get("op_type", ""))),
            defer_reason=data.get("defer_reason"),
            root_cause_bucket=data.get("root_cause_bucket"),
            op_spec=(
                OpSpec.from_dict(data["op_spec"])
                if isinstance(data.get("op_spec"), dict)
                else None
            ),
            board_cycles=(
                int(data["board_cycles"])
                if data.get("board_cycles") is not None
                else None
            ),
            board_latency_ms=(
                float(data["board_latency_ms"])
                if data.get("board_latency_ms") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LutQueryResult:
    """Structured LUT query result that distinguishes miss vs infeasible."""

    status: str
    entry: Optional[LutEntry] = None
    status_entry: Optional[FormalLutStatusEntry] = None
    matched_by: str = "none"


# ============================================================================
# LUT Table (类似 FBNet 的 LutTable)
# ============================================================================


class LutTable:
    """LUT 查找表

    基于 OpSpec 到 LutEntry 的映射，支持：
    - 快速查询（O(1)）
    - Pickle 序列化
    - 重复条目处理
    - 统计信息查询

    Example:
        >>> lut_table = LutTable()
        >>> entry = LutEntry(op_spec=..., latency_ms=1.5, dsp=32, bram=2, lut=100)
        >>> lut_table.add(entry)
        >>> result = lut_table.query(op_spec)
    """

    def __init__(self, entries: Optional[List[LutEntry]] = None):
        self._index: Dict[OpSpec, LutEntry] = {}
        self._duplicates: Dict[OpSpec, List[LutEntry]] = {}

        if entries:
            for entry in entries:
                self.add(entry)

    def add(self, entry: LutEntry) -> "LutTable":
        """添加一个 LUT 条目"""
        if entry.op_spec in self._index:
            # 处理重复条目（类似 FBNet 的去重逻辑）
            if entry.op_spec not in self._duplicates:
                self._duplicates[entry.op_spec] = [self._index[entry.op_spec]]
            self._duplicates[entry.op_spec].append(entry)
            # 保留第一个（或可以改为平均）
        else:
            self._index[entry.op_spec] = entry
        return self

    def extend(self, entries: List[LutEntry]) -> "LutTable":
        """批量添加条目"""
        for entry in entries:
            self.add(entry)
        return self

    def query(self, op_spec: OpSpec) -> Optional[LutEntry]:
        """查询 LUT

        Returns:
            LutEntry 如果找到，否则返回 None
        """
        return self._index.get(op_spec)

    def contains(self, op_spec: OpSpec) -> bool:
        """检查 LUT 中是否包含该算子"""
        return op_spec in self._index

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, op_spec: OpSpec) -> bool:
        return self.contains(op_spec)

    def save(self, path: str) -> None:
        """保存 LUT 到文件"""
        data = {
            "entries": [entry.to_dict() for entry in self._index.values()],
            "duplicates": {
                str(k): [e.to_dict() for e in v]
                for k, v in self._duplicates.items()
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_json(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a human-readable JSON representation of the LUT."""
        data = {
            "entries": [entry.to_dict() for entry in self._index.values()],
            "duplicates": {
                str(k): [e.to_dict() for e in v]
                for k, v in self._duplicates.items()
            },
        }
        if metadata:
            data["metadata"] = dict(metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LutTable":
        """从文件加载 LUT"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        lut_table = cls()
        for entry_data in data["entries"]:
            lut_table.add(LutEntry.from_dict(entry_data))

        return lut_table

    @classmethod
    def load_json(
        cls,
        path: str,
        *,
        default_clock_mhz: Optional[int] = None,
    ) -> "LutTable":
        """Load a LUT from JSON.

        Supports both the structured ``{"entries": [...]}`` format and the
        legacy analytical ``{"op_k3_e1_cin16_...": {...}}`` mapping.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            lut_table = cls()
            for entry_data in data["entries"]:
                lut_table.add(LutEntry.from_dict(entry_data))
            return lut_table

        if isinstance(data, dict):
            return cls._load_legacy_json_map(data, default_clock_mhz=default_clock_mhz)

        raise ValueError(f"Unsupported LUT JSON format: {path}")

    @classmethod
    def _load_legacy_json_map(
        cls,
        payload: Dict[str, Any],
        *,
        default_clock_mhz: Optional[int] = None,
    ) -> "LutTable":
        lut_table = cls()
        clock_mhz = float(payload.get("clock_mhz", default_clock_mhz or 200))
        pattern = re.compile(
            r"^(?P<op>.+)_k(?P<kernel>\d+)_e(?P<expand>\d+)_cin(?P<cin>\d+)_cout(?P<cout>\d+)_res(?P<res>\d+)_s(?P<stride>\d+)$"
        )

        for raw_key, raw_metrics in payload.items():
            if raw_key == "clock_mhz" or not isinstance(raw_metrics, dict):
                continue

            match = pattern.match(str(raw_key))
            if match is None:
                continue

            op = match.group("op")
            kernel_size = int(match.group("kernel"))
            expand_ratio = int(match.group("expand"))
            in_channels = int(match.group("cin"))
            out_channels = int(match.group("cout"))
            resolution = int(match.group("res"))
            stride = int(match.group("stride"))

            if op == "skip" and (kernel_size != 1 or expand_ratio != 1):
                continue

            lut_op = canonicalize_lut_op_name(op)
            groups = in_channels if op in {"dw_pw_conv", "mixconv", "denoise", "edge"} else 1

            latency_ms = float(raw_metrics.get("latency_ms", 0.0))
            power_w = float(raw_metrics.get("power_w", 0.0))
            energy_mj = raw_metrics.get("energy_mj")
            if energy_mj is None:
                energy_mj = power_w * latency_ms
            cycles = raw_metrics.get("cycles")
            if cycles is None:
                cycles = int(round(latency_ms * clock_mhz * 1_000.0))

            lut_table.add(
                LutEntry(
                    op_spec=OpSpec(
                        op=lut_op,
                        kernel_size=kernel_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        groups=groups,
                        expand_ratio=expand_ratio,
                        input_resolution=(resolution, resolution),
                    ),
                    latency_ms=latency_ms,
                    cycles=int(cycles),
                    dsp=int(raw_metrics.get("dsp", 0)),
                    bram=int(raw_metrics.get("bram", 0)),
                    lut=int(raw_metrics.get("lut", 0)),
                    power_w=power_w,
                    energy_mj=float(energy_mj),
                )
            )

        return lut_table

    def get_stats(self) -> Dict[str, Any]:
        """获取 LUT 统计信息"""
        return {
            "total_entries": len(self._index),
            "duplicate_specs": len(self._duplicates),
            "duplicate_count": sum(len(v) for v in self._duplicates.values()),
        }


# ============================================================================
# LUT Query Engine (类似 FBNet 的 LutQuery)
# ============================================================================


class LutQueryEngine:
    """LUT 查询引擎

    支持高级查询功能：
    - 插值查询（当精确匹配不存在时）
    - 范围查询
    - 统计汇总

    Attributes:
        lut_table: LUT 查找表
        enable_interpolation: 是否启用插值
    """

    def __init__(
        self,
        lut_table: LutTable,
        enable_interpolation: bool = False,
        *,
        allow_shape_only_match: bool = True,
        strict_formal_lut: bool = False,
        formal_status_entries: Optional[Dict[OpSpec, FormalLutStatusEntry]] = None,
    ):
        self.lut_table = lut_table
        self.enable_interpolation = enable_interpolation
        self.allow_shape_only_match = allow_shape_only_match
        self.strict_formal_lut = strict_formal_lut
        self.formal_status_entries = dict(formal_status_entries or {})

    def query(self, op_spec: OpSpec) -> Optional[LutEntry]:
        """查询算子的硬件代价

        Args:
            op_spec: 算子规格

        Returns:
            LutEntry 如果找到，否则尝试插值或返回 None
        """
        result = self.query_with_status(op_spec)
        return result.entry

    def query_with_status(self, op_spec: OpSpec) -> LutQueryResult:
        """Query LUT and preserve infeasible-vs-miss semantics."""
        if self.strict_formal_lut:
            status_entry = self.formal_status_entries.get(op_spec)
            if status_entry is not None:
                if str(status_entry.status).strip() == "measured":
                    entry = self.lut_table.query(op_spec)
                    if entry is None:
                        raise KeyError(
                            f"Formal LUT status marks {status_entry.case_name} as measured "
                            "but the measured LUT table has no exact entry."
                        )
                    return LutQueryResult(
                        status="measured",
                        entry=entry,
                        status_entry=status_entry,
                        matched_by="formal_exact",
                    )
                return LutQueryResult(
                    status=str(status_entry.status).strip(),
                    entry=None,
                    status_entry=status_entry,
                    matched_by="formal_status",
                )

            return LutQueryResult(status="missing", entry=None, status_entry=None, matched_by="formal_miss")

        entry = self.lut_table.query(op_spec)
        if entry is not None:
            return LutQueryResult(status="measured", entry=entry, matched_by="exact")

        if self.allow_shape_only_match:
            shape_only_match = self._query_unique_shape_match(op_spec)
            if shape_only_match is not None:
                return LutQueryResult(status="measured", entry=shape_only_match, matched_by="shape_only")

        if self.enable_interpolation:
            interpolated = self._interpolate_query(op_spec)
            if interpolated is not None:
                return LutQueryResult(status="measured", entry=interpolated, matched_by="interpolated")

        return LutQueryResult(status="missing", entry=None, matched_by="none")

    def _query_unique_shape_match(self, op_spec: OpSpec) -> Optional[LutEntry]:
        matches = [
            entry
            for spec, entry in self.lut_table._index.items()
            if spec.shape_signature() == op_spec.shape_signature()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def load_formal_status_json(path: str) -> Dict[OpSpec, FormalLutStatusEntry]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError(f"Unsupported formal LUT status JSON format: {path}")

        index: Dict[OpSpec, FormalLutStatusEntry] = {}
        for raw_entry in entries:
            entry = FormalLutStatusEntry.from_dict(raw_entry)
            if entry.op_spec is None:
                raise ValueError(
                    f"Formal LUT status entry for case {entry.case_name} is missing op_spec."
                )
            index[entry.op_spec] = entry
        return index

    def _interpolate_query(self, op_spec: OpSpec) -> Optional[LutEntry]:
        """插值查询（基于邻近条目）

        当精确匹配不存在时，从相近的条目插值估算。
        例如：查询 conv(16, 32) 可以从 conv(16, 24) 和 conv(16, 48) 插值。
        """
        # 找到相同 op、kernel_size、stride、groups 的所有条目
        candidates = [
            (k, v)
            for k, v in self.lut_table._index.items()
            if k.op == op_spec.op
            and k.kernel_size == op_spec.kernel_size
            and k.stride == op_spec.stride
            and k.groups == op_spec.groups
            and k.expand_ratio == op_spec.expand_ratio
            and k.input_resolution == op_spec.input_resolution
            and k.implementation_signature() == op_spec.implementation_signature()
        ]

        if not candidates:
            return None

        # 按通道数排序
        candidates.sort(key=lambda x: x[0].out_channels)

        # 找到插值范围
        lower = None
        upper = None

        for spec, entry in candidates:
            if spec.out_channels <= op_spec.out_channels:
                lower = (spec, entry)
            else:
                upper = (spec, entry)
                break

        # 简单线性插值
        if lower and upper:
            lower_spec, lower_entry = lower
            upper_spec, upper_entry = upper

            # 插值系数
            t = (op_spec.out_channels - lower_spec.out_channels) / (
                upper_spec.out_channels - lower_spec.out_channels
            )

            # 对每个指标进行插值
            def interp_optional_float(lower_value: Optional[float], upper_value: Optional[float]) -> Optional[float]:
                if lower_value is None or upper_value is None:
                    return None
                return lower_value + t * (upper_value - lower_value)

            interpolated = LutEntry(
                op_spec=op_spec,
                latency_ms=lower_entry.latency_ms + t * (upper_entry.latency_ms - lower_entry.latency_ms),
                cycles=int(lower_entry.cycles + t * (upper_entry.cycles - lower_entry.cycles)),
                dsp=int(lower_entry.dsp + t * (upper_entry.dsp - lower_entry.dsp)),
                bram=int(lower_entry.bram + t * (upper_entry.bram - lower_entry.bram)),
                lut=int(lower_entry.lut + t * (upper_entry.lut - lower_entry.lut)),
                power_w=interp_optional_float(lower_entry.power_w, upper_entry.power_w),
                energy_mj=interp_optional_float(lower_entry.energy_mj, upper_entry.energy_mj),
            )
            return interpolated

        # 只有下界或上界，直接返回最近邻
        return (lower or upper)[1] if lower or upper else None


# ============================================================================
# LUT Builder - 用于从 profiling 数据构建 LUT
# ============================================================================


class LutBuilder:
    """LUT 构建器

    用于从硬件 profiling 数据或仿真结果构建 LUT 表。
    可以集成 Vivado HLS、Vitis 等工具的输出。

    Example:
        builder = LutBuilder()
        builder.add_profiling_result(...)
        lut_table = builder.build()
    """

    def __init__(self):
        self._entries: List[LutEntry] = []

    def add_profiling_result(
        self,
        op: str,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        expand_ratio: int = 1,
        input_resolution: Tuple[int, int] = (224, 224),
        latency_ms: float = 0.0,
        cycles: int = 0,
        dsp: int = 0,
        bram: int = 0,
        lut: int = 0,
        power_w: float = 0.0,
        bitwidth: int = 8,
        input_parallelism: int = 1,
        output_parallelism: int = 1,
        unroll_factor: int = 1,
        target_clock_mhz: Optional[float] = None,
    ) -> "LutBuilder":
        """添加 profiling 结果

        Args:
            op: 算子类型
            kernel_size: 卷积核大小
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长
            groups: 分组数
            expand_ratio: 扩展比
            input_resolution: 输入分辨率
            latency_ms: 延迟（毫秒）
            cycles: 时钟周期数
            dsp: DSP 使用量
            bram: BRAM 使用量
            lut: LUT 使用量
            power_w: 功耗（瓦特）
        """
        op_spec = OpSpec(
            op=op,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            groups=groups,
            expand_ratio=expand_ratio,
            input_resolution=input_resolution,
            bitwidth=bitwidth,
            input_parallelism=input_parallelism,
            output_parallelism=output_parallelism,
            unroll_factor=unroll_factor,
            target_clock_mhz=target_clock_mhz,
        )

        energy_mj = latency_ms * power_w if power_w > 0 else 0.0

        entry = LutEntry(
            op_spec=op_spec,
            latency_ms=latency_ms,
            cycles=cycles,
            dsp=dsp,
            bram=bram,
            lut=lut,
            power_w=power_w,
            energy_mj=energy_mj,
        )

        self._entries.append(entry)
        return self

    def build(self) -> LutTable:
        """构建 LUT 表"""
        return LutTable(entries=self._entries)

    def save_to_file(self, path: str) -> None:
        """直接构建并保存到文件"""
        lut_table = self.build()
        lut_table.save(path)


# ============================================================================
# 预定义的 FPGA LUT 表（基于公开数据或分析模型生成）
# ============================================================================


def create_dummy_fpga_lut() -> LutTable:
    """创建一个虚拟的 FPGA LUT 表

    用于快速测试和原型开发。实际使用时应该从：
    1. HW-NAS-Bench 提取真实数据
    2. Vivado HLS profiling
    3. FPGA 板卡实测
    """
    builder = LutBuilder()

    # 基本参数范围
    kernel_sizes = [3, 5]
    channel_choices = [16, 24, 32, 48, 64, 96]
    stride_options = [1, 2]

    # 生成虚拟数据（基于分析模型的近似值）
    for kernel_size in kernel_sizes:
        for in_ch in channel_choices:
            for out_ch in channel_choices:
                for stride in stride_options:
                    # Conv
                    macs = in_ch * out_ch * kernel_size * kernel_size * 224 * 224 // (stride * stride)
                    latency_ms = macs / (200e6) * 0.01  # 假设 200MHz，效率 1%
                    dsp = min(512, (out_ch // 8) * kernel_size)
                    bram = math.ceil((in_ch + out_ch) * kernel_size * kernel_size / 1024)
                    lut = int(macs / 1000 * kernel_size)

                    builder.add_profiling_result(
                        op="conv",
                        kernel_size=kernel_size,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        latency_ms=latency_ms,
                        cycles=int(latency_ms * 200e6),
                        dsp=dsp,
                        bram=bram,
                        lut=lut,
                        power_w=2.0 + dsp * 0.01,
                    )

                    # Depthwise Conv
                    dw_macs = in_ch * kernel_size * kernel_size * 224 * 224 // (stride * stride)
                    dw_latency = dw_macs / (200e6) * 0.01
                    dw_dsp = min(512, in_ch // 4)
                    dw_bram = math.ceil((in_ch + out_ch) * kernel_size * kernel_size / 1024)
                    dw_lut = int(dw_macs / 1000 * kernel_size)

                    builder.add_profiling_result(
                        op="dw_conv",
                        kernel_size=kernel_size,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        groups=in_ch,
                        latency_ms=dw_latency,
                        cycles=int(dw_latency * 200e6),
                        dsp=dw_dsp,
                        bram=dw_bram,
                        lut=dw_lut,
                        power_w=1.5 + dw_dsp * 0.01,
                    )

    # MBConv
    for kernel_size in [3, 5]:
        for in_ch in [16, 24, 32, 48, 64]:
            for out_ch in [16, 24, 32, 48, 64, 96]:
                for expand in [1, 2, 4, 6]:
                    hidden_ch = in_ch * expand
                    # Simplified estimation
                    macs = (
                        in_ch * hidden_ch  # expand
                        + hidden_ch * kernel_size * kernel_size  # dw
                        + hidden_ch * out_ch  # project
                    ) * 224 * 224
                    latency_ms = macs / (200e6) * 0.01
                    dsp = min(512, (hidden_ch // 8) * kernel_size)
                    bram = math.ceil((hidden_ch + out_ch) * kernel_size * kernel_size / 1024)
                    lut = int(macs / 1000 * kernel_size)

                    builder.add_profiling_result(
                        op="mbconv",
                        kernel_size=kernel_size,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=1,
                        expand_ratio=expand,
                        latency_ms=latency_ms,
                        cycles=int(latency_ms * 200e6),
                        dsp=dsp,
                        bram=bram,
                        lut=lut,
                        power_w=2.5 + dsp * 0.01,
                    )

                    # Fused MBConv
                    fused_latency = latency_ms * 0.8  # 稍快
                    builder.add_profiling_result(
                        op="fused_mbconv",
                        kernel_size=kernel_size,
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=1,
                        expand_ratio=expand,
                        latency_ms=fused_latency,
                        cycles=int(fused_latency * 200e6),
                        dsp=dsp,
                        bram=int(bram * 0.9),
                        lut=int(lut * 0.9),
                        power_w=2.3 + dsp * 0.01,
                    )

    # Skip
    for ch in channel_choices:
        builder.add_profiling_result(
            op="skip",
            kernel_size=1,
            in_channels=ch,
            out_channels=ch,
            stride=1,
            latency_ms=0.01,
            cycles=int(0.01 * 200e6),
            dsp=0,
            bram=math.ceil(ch / 1024),
            lut=32,
            power_w=0.1,
        )

    return builder.build()
