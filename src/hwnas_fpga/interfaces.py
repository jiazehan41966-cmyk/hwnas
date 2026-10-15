from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class HardwareSpec:
    name: str
    clock_mhz: int
    max_lut: Optional[int] = None
    max_ff: Optional[int] = None
    # ``max_bram`` is retained as the historical BRAM36-block budget used by
    # the NAS cost model.  The explicit fields prevent HLS BRAM18 reports from
    # being compared to that budget without conversion.
    max_bram: Optional[int] = None
    max_bram18: Optional[int] = None
    max_bram36: Optional[int] = None
    max_uram: Optional[int] = None
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
    # Detection-task metrics. ``map`` is mean average precision over the
    # configured IoU thresholds; map50/map75 are common single-threshold views.
    map: Optional[float] = None
    map50: Optional[float] = None
    map75: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    params: Optional[int] = None
    macs: Optional[int] = None
    flops: Optional[int] = None
    latency_ms: Optional[float] = None
    fps: Optional[float] = None
    gmac_per_s: Optional[float] = None
    gops: Optional[float] = None
    energy_mj: Optional[float] = None
    lut: Optional[int] = None
    bram: Optional[int] = None
    dsp: Optional[int] = None
    power_w: Optional[float] = None
    chip_average_power_w: Optional[float] = None
    dynamic_power_w: Optional[float] = None
    fps_per_w: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None
    offchip_mem_mb: Optional[float] = None
    fp32_macro_f1: Optional[float] = None
    int8_macro_f1: Optional[float] = None
    board_macro_f1: Optional[float] = None
    delta_int8_macro_f1: Optional[float] = None
    delta_board_macro_f1: Optional[float] = None
    fp32_top1: Optional[float] = None
    int8_top1: Optional[float] = None
    board_top1: Optional[float] = None
    delta_int8_top1: Optional[float] = None
    delta_board_top1: Optional[float] = None
    board_sample_coverage: Optional[float] = None
    board_crc_error_count: Optional[int] = None
    board_status_error_count: Optional[int] = None
    board_argmax_mismatch_count: Optional[int] = None
    early_expand_pressure: Optional[float] = None
    interconnect_pressure: Optional[float] = None
    memory_pressure: Optional[float] = None
    fanout_pressure: Optional[float] = None
    stream_width: Optional[float] = None
    physical_risk: Optional[float] = None
    zcu104_operator_lutram_heuristic_pressure: Optional[float] = None
    zcu104_predicted_lut_as_memory: Optional[float] = None
    zcu104_predicted_lut_as_memory_pressure: Optional[float] = None
    zcu104_predicted_bram18: Optional[float] = None
    zcu104_predicted_bram18_pressure: Optional[float] = None
    zcu104_lut_as_memory_scale: Optional[float] = None
    # BRAM allocation is clamped to the device tile count, so the pre-fallback
    # request is the only field that shows a saturated, overflowing design.
    zcu104_requested_bram18_before_fallback: Optional[float] = None
    zcu104_predicted_bram18_overflow: Optional[float] = None
    whole_network_storage_calibration_validated: Optional[float] = None
    full_network_route_established: Optional[float] = None
    full_network_power_established: Optional[float] = None
    # The metric actually used by the configured search objective.
    selection_score: Optional[float] = None


@dataclass
class SearchCandidate:
    arch_id: str
    encoding: dict[str, Any]
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
