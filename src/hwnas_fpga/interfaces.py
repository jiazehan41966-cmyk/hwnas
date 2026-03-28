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


@dataclass
class CandidateMetrics:
    accuracy: Optional[float] = None
    latency_ms: Optional[float] = None
    energy_mj: Optional[float] = None
    lut: Optional[int] = None
    bram: Optional[int] = None
    dsp: Optional[int] = None
    power_w: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    offchip_mem_mb: Optional[float] = None


@dataclass
class SearchCandidate:
    arch_id: str
    encoding: dict[str, Any]
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
