from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HardwareSpec:
    name: str
    clock_mhz: int
    max_lut: Optional[int] = None
    max_ff: Optional[int] = None
    max_bram: Optional[int] = None
    max_dsp: Optional[int] = None
    max_power_w: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    offchip_mem_mb: Optional[float] = None


@dataclass
class SearchConstraints:
    max_latency_ms: Optional[float] = None
    max_energy_mj: Optional[float] = None
    max_model_size_mb: Optional[float] = None
    max_lut: Optional[int] = None
    max_bram: Optional[int] = None
    max_dsp: Optional[int] = None
    max_power_w: Optional[float] = None
    max_memory_bandwidth_gbps: Optional[float] = None
    max_offchip_mem_mb: Optional[float] = None
    physical: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateMetrics:
    # Keep ``accuracy`` as the literal top-1 accuracy for new records;
    # readers must retain a legacy fallback for old artifacts.
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    # F_clean is clean inner-validation macro-F1. F_robust is the mean
    # macro-F1 over the frozen deterministic sonar-corruption suite.
    f_clean: Optional[float] = None
    f_robust: Optional[float] = None
    robust_worst_macro_f1: Optional[float] = None
    top1: Optional[float] = None
    top5: Optional[float] = None
    latency_ms: Optional[float] = None
    energy_mj: Optional[float] = None
    lut: Optional[int] = None
    bram: Optional[int] = None
    dsp: Optional[int] = None
    power_w: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    offchip_mem_mb: Optional[float] = None
    early_expand_pressure: Optional[float] = None
    interconnect_pressure: Optional[float] = None
    memory_pressure: Optional[float] = None
    fanout_pressure: Optional[float] = None
    stream_width: Optional[float] = None
    physical_risk: Optional[float] = None
    # The metric actually used by the configured search objective.
    selection_score: Optional[float] = None


@dataclass
class SearchCandidate:
    arch_id: str
    encoding: dict[str, Any]
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
