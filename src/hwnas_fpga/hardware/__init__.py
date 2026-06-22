"""Hardware cost estimation and FPGA measurement hooks."""

from .backbone_cost import BackboneCostEstimate, BackboneCostEstimator, BackboneLayerCost
from .boards import BOARD_PROFILES, get_board_profile, list_board_profiles, resolve_board_profile
from .cost import CostEstimate, FPGACostEstimator, LayerCost
from .lookup_table import (
    LutBuilder,
    LutEntry,
    LutQueryEngine,
    LutTable,
    OpSpec,
    create_dummy_fpga_lut,
)
from .lut_pipeline import build_lut_from_manifest, load_lut_manifest, save_lut_from_manifest
from .report_parser import (
    approximate_fmax_from_wns,
    lut_entry_from_report,
    parse_hls_report,
    parse_hls_report_text,
    parse_vivado_power_text,
    parse_vivado_timing_summary_text,
    parse_vivado_utilization_text,
)
from .route_gate import (
    classify_route_gate_result,
    dedupe_candidates_by_signature,
    encoding_signature,
    pareto_topk_candidates,
    select_route_clean_best,
)
from .validation_report import (
    build_final_hardware_validation_report,
    normalize_candidate_payload,
)
from .acceptance_audit import (
    GapAcceptanceAuditInputs,
    build_gap_acceptance_audit,
)

__all__ = [
    "BOARD_PROFILES",
    "BackboneCostEstimate",
    "BackboneCostEstimator",
    "BackboneLayerCost",
    "CostEstimate",
    "FPGACostEstimator",
    "GapAcceptanceAuditInputs",
    "LayerCost",
    "LutBuilder",
    "LutEntry",
    "LutQueryEngine",
    "LutTable",
    "OpSpec",
    "build_lut_from_manifest",
    "build_gap_acceptance_audit",
    "build_final_hardware_validation_report",
    "classify_route_gate_result",
    "create_dummy_fpga_lut",
    "dedupe_candidates_by_signature",
    "encoding_signature",
    "get_board_profile",
    "load_lut_manifest",
    "list_board_profiles",
    "lut_entry_from_report",
    "normalize_candidate_payload",
    "pareto_topk_candidates",
    "approximate_fmax_from_wns",
    "parse_hls_report",
    "parse_hls_report_text",
    "parse_vivado_power_text",
    "parse_vivado_timing_summary_text",
    "parse_vivado_utilization_text",
    "resolve_board_profile",
    "save_lut_from_manifest",
    "select_route_clean_best",
]
