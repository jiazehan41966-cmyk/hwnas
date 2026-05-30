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

__all__ = [
    "BOARD_PROFILES",
    "BackboneCostEstimate",
    "BackboneCostEstimator",
    "BackboneLayerCost",
    "CostEstimate",
    "FPGACostEstimator",
    "LayerCost",
    "LutBuilder",
    "LutEntry",
    "LutQueryEngine",
    "LutTable",
    "OpSpec",
    "build_lut_from_manifest",
    "create_dummy_fpga_lut",
    "get_board_profile",
    "load_lut_manifest",
    "list_board_profiles",
    "lut_entry_from_report",
    "approximate_fmax_from_wns",
    "parse_hls_report",
    "parse_hls_report_text",
    "parse_vivado_power_text",
    "parse_vivado_timing_summary_text",
    "parse_vivado_utilization_text",
    "resolve_board_profile",
    "save_lut_from_manifest",
]
