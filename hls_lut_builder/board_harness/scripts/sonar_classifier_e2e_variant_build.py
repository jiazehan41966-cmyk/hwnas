#!/usr/bin/env python3
"""Build isolated e2e implementation variants without measuring board latency.

Variants keep the arch_84 topology intact and alter only harness implementation
knobs such as layer-boundary FIFO depth/style and Vivado implementation
strategy.  They are intentionally kept out of the canonical e2e measurement
ledger until a variant is explicitly promoted and measured as a full-network
harness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BOARD_HARNESS_DIR = ROOT / "hls_lut_builder" / "board_harness"
E2E_DIR = BOARD_HARNESS_DIR / "results" / "sonar_classifier_end_to_end_board"
BASE_TARGET_NAME = "sonar_classifier_top8_03_arch_84_e2e"
BASE_MEASUREMENT_ROOT = E2E_DIR / "real_measurement" / BASE_TARGET_NAME
VARIANTS_ROOT = BASE_MEASUREMENT_ROOT / "implementation_variants"
LOWDSP_RECOVERY_ROOT = ROOT / "hls_lut_builder" / "results" / "v2_arch84_e2e_lowdsp_recovery" / "p"

sys.path.insert(0, str(SCRIPT_DIR))

from batch_measure_ops import DEFAULT_BOARD_CONFIG, DEFAULT_VIVADO  # noqa: E402
import current84_fullcombo_board_validation as fullcombo  # noqa: E402
import sonar_classifier_e2e_board_validation as e2e  # noqa: E402


STATUS_FIELDS = [
    "variant_name",
    "target_name",
    "status",
    "returncode",
    "fifo_depth",
    "fifo_style",
    "timing_variant",
    "impl_profile",
    "argmax_pipeline",
    "completion_mode",
    "bitstream",
    "wns_ns",
    "timing_clean",
    "measurement_status",
    "measurement_ready",
    "measurement_json_path",
    "strict_latency_usable",
    "lut",
    "ff",
    "bram_tile",
    "dsp",
    "global_congestion_level",
    "failure_summary",
    "project_root",
    "stdout_log",
    "build_result_json",
]

MEASUREMENT_READY_FIELDS = [
    "variant_name",
    "target_name",
    "implementation_status",
    "bitstream_exists",
    "wns_ns",
    "timing_clean",
    "measurement_status",
    "measurement_ready",
    "measurement_json_path",
    "strict_latency_usable",
    "readiness_blockers",
    "required_next_action",
    "suggested_measure_command",
    "bitstream",
    "bitstream_sha256",
    "preflight_report_json",
    "preflight_report_md",
    "build_result_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated arch_84 e2e implementation variants.")
    parser.add_argument("mode", choices=("generate", "build", "all", "refresh", "measure"))
    parser.add_argument("--variant-name")
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-device-pattern", default="*xc7k325t*")
    parser.add_argument("--hw-target-index", type=int, default=0)
    parser.add_argument("--program-retries", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--uart-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--post-program-delay-seconds", type=float, default=1.0)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=30)
    parser.add_argument("--fifo-depth", type=int, default=64)
    parser.add_argument("--fifo-style", choices=("lut", "lut_registered", "bram"), default="lut_registered")
    parser.add_argument("--timing-variant", default="place_altspread_route_more_global")
    parser.add_argument("--impl-profile", choices=("default", "fanout_only", "fanout_floorplan"), default="default")
    parser.add_argument("--argmax-pipeline", action="store_true")
    parser.add_argument(
        "--completion-mode",
        choices=("axil_and_sink", "sink_and_argmax"),
        default="axil_and_sink",
        help="End condition for e2e timing. Default preserves the original AXI-Lite-and-output rule.",
    )
    parser.add_argument(
        "--component-override",
        action="append",
        default=[],
        help="Override one flattened component as role=case_name. May be repeated. Variant-only; never canonical.",
    )
    parser.add_argument("--watchdog-cycles", type=int, default=200_000_000)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=240)
    parser.add_argument("--skip-measure", action="store_true")
    parser.add_argument("--allow-timing-fail-measurement", action="store_true")
    parser.add_argument("--expected-bitstream-sha256", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.mode != "refresh" and not args.variant_name:
        parser.error("--variant-name is required unless mode is refresh")
    return args


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    if not value:
        raise ValueError("variant name is empty after sanitization")
    return value


def target_name(variant_name: str) -> str:
    return f"{BASE_TARGET_NAME}__{variant_name}"


def variant_root(variant_name: str) -> Path:
    return VARIANTS_ROOT / variant_name


def project_root(variant_name: str) -> Path:
    return variant_root(variant_name) / "harness_project"


def measurement_json_path(variant_name: str) -> Path:
    return variant_root(variant_name) / "measurements" / f"{target_name(variant_name)}.measurement.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_case_dir(case_name: str) -> Path:
    recovery_candidate = LOWDSP_RECOVERY_ROOT / case_name
    if recovery_candidate.exists():
        return recovery_candidate.resolve()
    return fullcombo.resolve_case_dir(case_name)


def parse_component_overrides(raw_items: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"component override must be role=case_name, got: {raw}")
        role, case_name = raw.split("=", 1)
        role = role.strip()
        case_name = case_name.strip()
        if not role or not case_name:
            raise ValueError(f"component override must be role=case_name, got: {raw}")
        overrides[role] = case_name
    return overrides


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def tcl_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if marker in current:
        return
    path.write_text(current.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def fanout_floorplan_xdc() -> str:
    return r"""
# BEGIN sonar_classifier_e2e_variant_impl_profile fanout_floorplan
# Best-effort synthesis hints for very high-fanout harness control nets.
set_property -quiet MAX_FANOUT 64 [get_nets -quiet rst_n]
set_property -quiet MAX_FANOUT 64 [get_nets -quiet start_pulse]
set_property -quiet MAX_FANOUT 64 [get_nets -quiet boot_started]
# END sonar_classifier_e2e_variant_impl_profile fanout_floorplan
"""


def fanout_only_xdc() -> str:
    return r"""
# BEGIN sonar_classifier_e2e_variant_impl_profile fanout_only
# Best-effort synthesis hints for very high-fanout harness control nets.
set_property -quiet MAX_FANOUT 64 [get_nets -quiet rst_n]
set_property -quiet MAX_FANOUT 64 [get_nets -quiet start_pulse]
set_property -quiet MAX_FANOUT 64 [get_nets -quiet boot_started]
# END sonar_classifier_e2e_variant_impl_profile fanout_only
"""


def fanout_floorplan_place_pre_tcl() -> str:
    return r"""
# BEGIN sonar_classifier_e2e_variant_impl_profile fanout_floorplan
proc codex_soft_pblock {name regions patterns} {
    if {[llength [get_pblocks -quiet $name]] == 0} {
        create_pblock $name
    }
    set cells [list]
    foreach pat $patterns {
        set found [get_cells -quiet -hierarchical -filter "NAME =~ $pat"]
        if {[llength $found] > 0} {
            set cells [concat $cells $found]
        } else {
            puts "WARN: fanout_floorplan pblock $name matched no cells for pattern $pat"
        }
    }
    if {[llength $cells] > 0} {
        if {[catch {add_cells_to_pblock [get_pblocks $name] $cells} msg]} {
            puts "WARN: fanout_floorplan add_cells_to_pblock $name failed: $msg"
        }
        if {[catch {resize_pblock [get_pblocks $name] -add $regions} msg]} {
            puts "WARN: fanout_floorplan resize_pblock $name failed: $msg"
        }
        if {[catch {set_property IS_SOFT true [get_pblocks $name]} msg]} {
            puts "WARN: fanout_floorplan set soft pblock $name failed: $msg"
        }
    }
}

# Pipeline-order soft placement affinity. These are intentionally soft: the
# placer may escape if the fully occupied DSP fabric needs it.
codex_soft_pblock pb_e2e_l1_stem {CLOCKREGION_X0Y5:CLOCKREGION_X1Y6} \
    {u_l1_stem_direct_kernel*}
codex_soft_pblock pb_e2e_l2_l3 {CLOCKREGION_X0Y3:CLOCKREGION_X1Y4} \
    {u_l2_stage0_block0_direct_kernel* u_l3_stage1_block0_pw_expand_kernel* u_l3_stage1_block0_dw_kernel* u_l3_stage1_block0_pw_project_kernel*}
codex_soft_pblock pb_e2e_l4 {CLOCKREGION_X0Y1:CLOCKREGION_X1Y3} \
    {u_l4_stage2_block0_pw_expand_kernel* u_l4_stage2_block0_dw_kernel* u_l4_stage2_block0_pw_project_kernel*}
codex_soft_pblock pb_e2e_l5_l6_l7 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y2} \
    {u_l5_stage3_block0_pw_expand_kernel* u_l5_stage3_block0_dw_kernel* u_l5_stage3_block0_pw_project_kernel* u_l6_global_avg_pool_gap_kernel* u_l7_fc_classifier_fc_kernel*}
# END sonar_classifier_e2e_variant_impl_profile fanout_floorplan
"""


def fanout_only_build_tcl(project: Path) -> str:
    xdc = project / "constraints" / "fanout_only_profile.xdc"
    return f"""
# BEGIN sonar_classifier_e2e_variant_impl_profile fanout_only
if {{[file exists "{tcl_path(xdc)}"]}} {{
    read_xdc "{tcl_path(xdc)}"
}}
if {{[catch {{set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE {{ExploreWithRemap}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE {{AggressiveFanoutOpt}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {{MoreGlobalIterations}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
# END sonar_classifier_e2e_variant_impl_profile fanout_only
"""


def fanout_floorplan_build_tcl(project: Path, place_pre_tcl: Path) -> str:
    xdc = project / "constraints" / "fanout_floorplan_profile.xdc"
    return f"""
# BEGIN sonar_classifier_e2e_variant_impl_profile fanout_floorplan
if {{[file exists "{tcl_path(xdc)}"]}} {{
    read_xdc "{tcl_path(xdc)}"
}}
if {{[catch {{set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE {{ExploreWithRemap}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE {{WLDrivenBlockPlacement}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PLACE_DESIGN.TCL.PRE "{tcl_path(place_pre_tcl)}" [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE {{AggressiveFanoutOpt}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
if {{[catch {{set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE {{MoreGlobalIterations}} [get_runs impl_1]}} msg]}} {{puts "WARN: $msg"}}
# END sonar_classifier_e2e_variant_impl_profile fanout_floorplan
"""


def apply_impl_profile(project: Path, profile: str) -> list[str]:
    if profile == "default":
        return []
    if profile == "fanout_only":
        xdc = project / "constraints" / "fanout_only_profile.xdc"
        xdc.write_text(fanout_only_xdc().strip() + "\n", encoding="utf-8")
        build_tcl = project / "scripts" / "build_bitstream.tcl"
        marker = "sonar_classifier_e2e_variant_impl_profile fanout_only"
        text = build_tcl.read_text(encoding="utf-8", errors="replace")
        insert = fanout_only_build_tcl(project).strip()
        if marker not in text:
            text = text.replace("launch_runs synth_1 -jobs 8", f"{insert}\n\nlaunch_runs synth_1 -jobs 8")
            build_tcl.write_text(text, encoding="utf-8")
        return [str(xdc), str(build_tcl)]
    if profile != "fanout_floorplan":
        raise ValueError(f"unsupported implementation profile: {profile}")
    xdc = project / "constraints" / "fanout_floorplan_profile.xdc"
    place_pre_tcl = project / "scripts" / "fanout_floorplan_place_pre.tcl"
    xdc.write_text(fanout_floorplan_xdc().strip() + "\n", encoding="utf-8")
    place_pre_tcl.write_text(fanout_floorplan_place_pre_tcl().strip() + "\n", encoding="utf-8")
    build_tcl = project / "scripts" / "build_bitstream.tcl"
    marker = "sonar_classifier_e2e_variant_impl_profile fanout_floorplan"
    text = build_tcl.read_text(encoding="utf-8", errors="replace")
    insert = fanout_floorplan_build_tcl(project, place_pre_tcl).strip()
    if marker not in text:
        text = text.replace("launch_runs synth_1 -jobs 8", f"{insert}\n\nlaunch_runs synth_1 -jobs 8")
        build_tcl.write_text(text, encoding="utf-8")
    return [str(xdc), str(place_pre_tcl), str(build_tcl)]


def parse_vivado_project_dir(build_tcl: Path) -> Path:
    text = build_tcl.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'^\s*set\s+vivado_project_dir\s+"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse vivado_project_dir from {build_tcl}")
    return Path(match.group(1))


def parse_wns_from_log(text: str) -> str:
    patterns = [
        r"Post Physical Optimization Timing Summary\s*\|\s*WNS=([-0-9.]+)",
        r"Intermediate Timing Summary\s*\|\s*WNS=([-0-9.]+)",
        r"^\s*([-0-9.]+)\s+[-0-9.]+\s+\d+\s+\d+\s+[-0-9.]+",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            return matches[-1]
    return ""


def parse_global_congestion(text: str) -> str:
    matches = re.findall(r"global congestion level is\s+([0-9]+)", text, flags=re.IGNORECASE)
    return matches[-1] if matches else ""


def summarize_build_failure(stdout: str, limit: int = 8) -> str:
    lines = [
        line.strip()
        for line in stdout.splitlines()
        if "ERROR:" in line
        or "CRITICAL WARNING:" in line
        or "WARNING: [Route" in line
        or "Design is not routable" in line
        or "global congestion level" in line
    ]
    return " | ".join(lines[-limit:])[:1600]


def collect_vivado_evidence(project: Path, out_dir: Path) -> dict[str, str]:
    reports_dir = out_dir / "vivado_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, str] = {}
    build_tcl = project / "scripts" / "build_bitstream.tcl"
    if not build_tcl.exists():
        return evidence
    vivado_project_dir = parse_vivado_project_dir(build_tcl)
    project_name = vivado_project_dir.name
    synth_dir = vivado_project_dir / f"{project_name}.runs" / "synth_1"
    impl_dir = vivado_project_dir / f"{project_name}.runs" / "impl_1"
    copies = {
        "synth_utilization_report": (synth_dir / "harness_top_utilization_synth.rpt", "harness_top_utilization_synth.rpt"),
        "placed_utilization_report": (impl_dir / "harness_top_utilization_placed.rpt", "harness_top_utilization_placed.rpt"),
        "placed_control_sets_report": (impl_dir / "harness_top_control_sets_placed.rpt", "harness_top_control_sets_placed.rpt"),
        "placed_io_report": (impl_dir / "harness_top_io_placed.rpt", "harness_top_io_placed.rpt"),
        "routed_route_status_report": (impl_dir / "harness_top_route_status.rpt", "harness_top_route_status_routed.rpt"),
        "routed_timing_summary_report": (impl_dir / "harness_top_timing_summary_routed.rpt", "harness_top_timing_summary_routed.rpt"),
        "routed_clock_utilization_report": (impl_dir / "harness_top_clock_utilization_routed.rpt", "harness_top_clock_utilization_routed.rpt"),
        "routed_power_report": (impl_dir / "harness_top_power_routed.rpt", "harness_top_power_routed.rpt"),
        "final_timing_summary_report": (project / "reports" / "timing_summary.rpt", "timing_summary.rpt"),
        "final_utilization_report": (project / "reports" / "utilization.rpt", "utilization.rpt"),
        "final_clock_utilization_report": (project / "reports" / "clock_utilization.rpt", "clock_utilization.rpt"),
        "impl_runme_log": (impl_dir / "runme.log", "impl_runme.log"),
        "synth_runme_log": (synth_dir / "runme.log", "synth_runme.log"),
    }
    for key, (source, dest_name) in copies.items():
        if source.exists():
            dest = reports_dir / dest_name
            shutil.copy2(source, dest)
            evidence[key] = str(dest)
    route_error_checkpoint = impl_dir / "harness_top_routed_error.dcp"
    if route_error_checkpoint.exists():
        evidence["route_error_checkpoint"] = str(route_error_checkpoint)
    return evidence


def generate_variant(args: argparse.Namespace) -> dict[str, Any]:
    variant_name = sanitize_name(args.variant_name)
    project = project_root(variant_name)
    if project.exists() and args.force:
        resolved = project.resolve()
        allowed = VARIANTS_ROOT.resolve()
        if allowed not in resolved.parents:
            raise RuntimeError(f"Refusing to remove path outside variants root: {resolved}")
        shutil.rmtree(project)
    (project / "export_rtl").mkdir(parents=True, exist_ok=True)

    spec = e2e.load_spec()
    board_config = Path(args.board_config).expanduser().resolve()
    component_rows = e2e.flattened_components(spec)
    overrides = parse_component_overrides(args.component_override)
    seen_overrides: set[str] = set()
    override_records: list[dict[str, str]] = []
    specs: list[dict[str, Any]] = []
    for row in component_rows:
        effective_row = dict(row)
        if effective_row["role"] in overrides:
            original_case = effective_row["case_name"]
            effective_row["case_name"] = overrides[effective_row["role"]]
            seen_overrides.add(effective_row["role"])
            override_records.append(
                {
                    "role": effective_row["role"],
                    "original_case_name": original_case,
                    "override_case_name": effective_row["case_name"],
                }
            )
        case_dir = resolve_case_dir(effective_row["case_name"])
        item = fullcombo.build_component_spec(effective_row["role"], effective_row["case_name"], case_dir, project / "export_rtl")
        item.update(effective_row)
        specs.append(item)
    missing_overrides = sorted(set(overrides) - seen_overrides)
    if missing_overrides:
        valid_roles = ", ".join(row["role"] for row in component_rows)
        raise ValueError(f"component override role(s) not found: {missing_overrides}. Valid roles: {valid_roles}")

    e2e.write_full_network_top(
        project,
        specs,
        board_config,
        args.fifo_depth,
        args.fifo_style,
        args.watchdog_cycles,
        argmax_pipeline=args.argmax_pipeline,
        completion_mode=args.completion_mode,
    )
    fullcombo.write_constraints(project, board_config)
    artifact = target_name(variant_name)
    fullcombo.write_build_tcl(project, artifact, board_config, recovery_variant=args.timing_variant)
    impl_profile_artifacts = apply_impl_profile(project, args.impl_profile)

    manifest = {
        "case_name": artifact,
        "base_target_name": BASE_TARGET_NAME,
        "variant_name": variant_name,
        "module_name": "sonar_classifier_e2e_harness",
        "board_config": str(board_config),
        "harness_kind": "single_full_network_latency_only_harness",
        "implementation_variant": True,
        "component_overrides": override_records,
        "parameter_mode": "latency_only_deterministic",
        "accuracy_claim": "none",
        "timing_variant": args.timing_variant,
        "impl_profile": args.impl_profile,
        "argmax_pipeline": bool(args.argmax_pipeline),
        "completion_mode": args.completion_mode,
        "harness_fifo_depth": args.fifo_depth,
        "harness_fifo_style": args.fifo_style,
        "impl_profile_artifacts": impl_profile_artifacts,
        "generated": {
            "project_root": str(project),
            "rtl_dir": str(project / "rtl"),
            "export_rtl_dir": str(project / "export_rtl"),
            "constraints": str(project / "constraints" / "harness.xdc"),
            "build_tcl": str(project / "scripts" / "build_bitstream.tcl"),
            "bitstream": str(project / "bitstream" / f"{artifact}.bit"),
        },
        "streaming": {
            "input_axis_word_count": int(specs[0]["case_meta"]["input_word_count"]),
            "output_axis_word_count": int(specs[-1]["case_meta"]["output_word_count"]),
            "classifier_class_count": int(specs[-1]["constants"].get("OUT_CH", 8)),
            "uart_checksum_field": "packed {argmax[7:0], checksum24[23:0]} for e2e only",
        },
        "layers": spec.get("topology", []),
        "components": [
            {
                "role": item["role"],
                "case_name": item["case_name"],
                "layer_index": item["layer_index"],
                "layer_role": item["layer_role"],
                "renamed_module": item["renamed_module"],
                "input_word_count": item["case_meta"]["input_word_count"],
                "output_word_count": item["case_meta"]["output_word_count"],
                "input_tdata_width": item["top_ports"]["input_stream_TDATA"],
                "output_tdata_width": item["top_ports"]["output_stream_TDATA"],
            }
            for item in specs
        ],
        "notes": [
            "Implementation variant only; do not merge into current84/fullcombo/arch84 canonical ledgers.",
            "No e2e latency is valid until this exact full-network variant is measured over COM5.",
            "Component overrides change implementation only; arch_84 layer topology and tensor shapes must remain compatible.",
            "Argmax pipeline is an implementation-only harness timing recovery knob when enabled.",
        ],
    }
    write_json(project / "harness_manifest.json", manifest)
    return manifest


def build_variant(args: argparse.Namespace) -> dict[str, Any]:
    variant_name = sanitize_name(args.variant_name)
    project = project_root(variant_name)
    if not (project / "harness_manifest.json").exists() or args.force:
        generate_variant(args)
    vivado = Path(args.vivado).expanduser().resolve()
    timeout_seconds = args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None
    result = fullcombo.build_bitstream(project, vivado, force=args.force, timeout_seconds=timeout_seconds)

    out_dir = variant_root(variant_name)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout = str(result.get("stdout", ""))
    stdout_log = logs_dir / "vivado_bitstream.out.log"
    stdout_log.write_text(stdout, encoding="utf-8", errors="replace")
    evidence = collect_vivado_evidence(project, out_dir)
    manifest = read_json(project / "harness_manifest.json")
    util_report_raw = evidence.get("placed_utilization_report") or evidence.get("synth_utilization_report")
    util = fullcombo.parse_utilization(Path(util_report_raw)) if util_report_raw else {
        "lut": None,
        "ff": None,
        "bram": None,
        "dsp": None,
    }
    timing_report = project / "reports" / "timing_summary.rpt"
    wns = fullcombo.parse_wns(timing_report) or parse_wns_from_log(stdout)
    payload = {
        "variant_name": variant_name,
        "target_name": target_name(variant_name),
        "status": result.get("status", ""),
        "returncode": result.get("returncode", ""),
        "fifo_depth": args.fifo_depth,
        "fifo_style": args.fifo_style,
        "timing_variant": args.timing_variant,
        "impl_profile": args.impl_profile,
        "argmax_pipeline": bool(manifest.get("argmax_pipeline", False)),
        "completion_mode": manifest.get("completion_mode", "axil_and_sink"),
        "bitstream": result.get("bitstream") or "",
        "wns_ns": wns,
        "lut": util.get("board_harness_lut", ""),
        "ff": util.get("board_harness_ff", ""),
        "bram_tile": util.get("board_harness_bram_tile", ""),
        "dsp": util.get("board_harness_dsp", ""),
        "global_congestion_level": parse_global_congestion(stdout),
        "failure_summary": "" if result.get("status") == "success" else summarize_build_failure(stdout),
        "project_root": str(project),
        "stdout_log": str(stdout_log),
        "component_overrides": manifest.get("component_overrides", []),
        "evidence": evidence,
        "build_result_json": str(out_dir / "build_result.json"),
        "updated_at": timestamp(),
    }
    write_json(out_dir / "build_result.json", payload)
    refresh_variants()
    return payload


def suggested_measure_command(variant_name: str, expected_hash: str = "") -> str:
    command = (
        "python hls_lut_builder\\board_harness\\scripts\\sonar_classifier_e2e_variant_build.py "
        f"measure --variant-name {variant_name} --serial-port COM5"
    )
    if expected_hash:
        command += f" --expected-bitstream-sha256 {expected_hash}"
    return command


def measurement_readiness_row(row: dict[str, Any]) -> dict[str, Any]:
    variant_name = str(row.get("variant_name", ""))
    bitstream_text = str(row.get("bitstream", ""))
    bitstream_path = Path(bitstream_text) if bitstream_text else Path()
    bitstream_exists = bool(bitstream_text) and bitstream_path.exists()
    bitstream_hash = sha256_file(bitstream_path) if bitstream_exists else ""
    wns_value = parse_float(row.get("wns_ns", ""))
    timing_clean = wns_value is not None and wns_value >= 0.0
    measurement_path = measurement_json_path(variant_name) if variant_name else Path()
    measurement_payload = read_json(measurement_path) if variant_name and measurement_path.exists() else {}
    uart_status = fullcombo.parse_int(fullcombo.frame_value(measurement_payload, "status_code")) if measurement_payload else None
    strict_latency_usable = bool(measurement_payload) and uart_status == 0 and timing_clean

    blockers: list[str] = []
    if row.get("status") != "success":
        blockers.append("implementation_not_success")
    if not bitstream_exists:
        blockers.append("bitstream_missing")
    if not timing_clean:
        blockers.append("timing_not_clean")

    if measurement_payload:
        measurement_status = "measured_timing_clean" if strict_latency_usable else "measured_not_strict_latency_usable"
        measurement_ready = False
        required_next_action = "review measurement result; strict latency is usable only when UART status_code=0 and WNS>=0"
    elif blockers:
        measurement_status = "not_ready_for_com5_measurement"
        measurement_ready = False
        required_next_action = "resolve implementation/timing readiness blockers before COM5 measurement"
    else:
        measurement_status = "ready_for_com5_measurement"
        measurement_ready = True
        required_next_action = "run variant measure only after explicit approval; do not write result into canonical e2e latency ledger"

    return {
        "variant_name": variant_name,
        "target_name": row.get("target_name", target_name(variant_name) if variant_name else ""),
        "implementation_status": row.get("status", ""),
        "status": row.get("status", ""),
        "bitstream_exists": bitstream_exists,
        "wns_ns": row.get("wns_ns", ""),
        "timing_clean": timing_clean,
        "measurement_status": measurement_status,
        "measurement_ready": measurement_ready,
        "measurement_json_path": str(measurement_path) if measurement_payload else "",
        "strict_latency_usable": strict_latency_usable,
        "readiness_blockers": "|".join(blockers),
        "required_next_action": required_next_action,
        "suggested_measure_command": suggested_measure_command(variant_name, bitstream_hash) if measurement_ready else "",
        "bitstream": bitstream_text if bitstream_exists else "",
        "bitstream_sha256": bitstream_hash,
        "preflight_report_json": str(variant_root(variant_name) / "measurement_preflight.json") if variant_name else "",
        "preflight_report_md": str(variant_root(variant_name) / "measurement_preflight.md") if variant_name else "",
        "build_result_json": row.get("build_result_json", ""),
    }


def measure_variant(args: argparse.Namespace) -> dict[str, Any]:
    variant_name = sanitize_name(args.variant_name)
    root = variant_root(variant_name)
    project = project_root(variant_name)
    build_payload = read_json(root / "build_result.json")
    if not build_payload:
        raise FileNotFoundError(f"Missing variant build result: {root / 'build_result.json'}")
    readiness = measurement_readiness_row(build_payload)
    if not readiness["bitstream_exists"]:
        raise FileNotFoundError(f"Variant bitstream is missing: {build_payload.get('bitstream', '')}")
    if not readiness["timing_clean"] and not args.allow_timing_fail_measurement:
        raise RuntimeError(
            f"Refusing timing-fail variant measurement without --allow-timing-fail-measurement: "
            f"{variant_name} WNS={readiness['wns_ns']}"
        )

    manifest_path = project / "harness_manifest.json"
    manifest = read_json(manifest_path)
    expected_target = target_name(variant_name)
    if manifest.get("implementation_variant") is not True:
        raise RuntimeError(f"Refusing measurement: harness manifest is not an implementation variant: {manifest_path}")
    if manifest.get("variant_name") != variant_name or manifest.get("case_name") != expected_target:
        raise RuntimeError(
            "Refusing measurement: manifest identity mismatch "
            f"variant={manifest.get('variant_name')} case={manifest.get('case_name')} expected={expected_target}"
        )
    if build_payload.get("target_name") != expected_target:
        raise RuntimeError(f"Refusing measurement: build result target mismatch: {build_payload.get('target_name')}")

    bitstream_path = Path(str(build_payload["bitstream"]))
    measurement_path = measurement_json_path(variant_name)
    if not is_relative_to(bitstream_path, project / "bitstream"):
        raise RuntimeError(f"Refusing measurement: bitstream is outside the variant bitstream directory: {bitstream_path}")
    if not is_relative_to(measurement_path, root / "measurements"):
        raise RuntimeError(f"Refusing measurement: JSON output is outside the variant measurement directory: {measurement_path}")
    actual_hash = sha256_file(bitstream_path)
    expected_hash = str(args.expected_bitstream_sha256 or "").strip().lower()
    if expected_hash and actual_hash.lower() != expected_hash:
        raise RuntimeError(
            "Refusing measurement: bitstream SHA256 mismatch "
            f"expected={expected_hash} actual={actual_hash}"
        )
    if measurement_path.exists() and not args.force:
        raise FileExistsError(f"Measurement JSON already exists; pass --force to overwrite: {measurement_path}")

    measurements_dir = root / "measurements"
    logs_dir = root / "logs"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurement = fullcombo.measure_harness(
        manifest=manifest_path,
        bitstream=bitstream_path,
        csv_path=measurements_dir / "measurements_raw.csv",
        json_out=measurement_path,
        serial_port=args.serial_port,
        vivado=Path(args.vivado).expanduser().resolve(),
        hw_server_url=args.hw_server_url,
        hw_device_pattern=args.hw_device_pattern,
        hw_target_index=args.hw_target_index,
        program_retries=args.program_retries,
        runs=args.runs,
        uart_timeout_seconds=args.uart_timeout_seconds,
        post_program_delay_seconds=args.post_program_delay_seconds,
        timeout_seconds=args.measurement_timeout_minutes * 60 if args.measurement_timeout_minutes > 0 else None,
        skip_measure=args.skip_measure,
    )
    (logs_dir / "variant_measurement.out.log").write_text(
        str(measurement.get("stdout", "")),
        encoding="utf-8",
        errors="replace",
    )
    refresh_variants()
    return measurement


def write_measurement_preflight(row: dict[str, Any], readiness: dict[str, Any]) -> None:
    variant_name = str(readiness.get("variant_name", ""))
    if not variant_name:
        return
    root = variant_root(variant_name)
    manifest_path = project_root(variant_name) / "harness_manifest.json"
    evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
    preflight = {
        "generated_at": timestamp(),
        "base_target_name": BASE_TARGET_NAME,
        "variant_name": variant_name,
        "target_name": readiness.get("target_name", ""),
        "measurement_scope": "isolated implementation variant",
        "measurement_not_run": not bool(readiness.get("measurement_json_path")),
        "readiness": readiness,
        "implementation": {
            "status": row.get("status", ""),
            "timing_variant": row.get("timing_variant", ""),
            "impl_profile": row.get("impl_profile", ""),
            "argmax_pipeline": row.get("argmax_pipeline", ""),
            "completion_mode": row.get("completion_mode", "axil_and_sink"),
            "fifo_depth": row.get("fifo_depth", ""),
            "fifo_style": row.get("fifo_style", ""),
            "component_overrides": row.get("component_overrides", []),
        },
        "resources": {
            "lut": row.get("lut", ""),
            "ff": row.get("ff", ""),
            "bram_tile": row.get("bram_tile", ""),
            "dsp": row.get("dsp", ""),
        },
        "artifacts": {
            "bitstream": readiness.get("bitstream", ""),
            "bitstream_sha256": readiness.get("bitstream_sha256", ""),
            "build_result_json": row.get("build_result_json", ""),
            "harness_manifest": str(manifest_path) if manifest_path.exists() else "",
            "stdout_log": row.get("stdout_log", ""),
            "timing_summary_report": evidence.get("final_timing_summary_report", ""),
            "routed_timing_summary_report": evidence.get("routed_timing_summary_report", ""),
            "route_status_report": evidence.get("routed_route_status_report", ""),
            "utilization_report": evidence.get("final_utilization_report", ""),
            "power_report": evidence.get("routed_power_report", ""),
            "clock_utilization_report": evidence.get("final_clock_utilization_report", ""),
        },
        "measurement": {
            "expected_measurement_json": str(measurement_json_path(variant_name)),
            "suggested_command": readiness.get("suggested_measure_command", ""),
            "strict_latency_rule": "UART status_code=0 and WNS>=0",
            "latency_rule": "Do not declare e2e latency until this full-network variant is measured on COM5.",
            "guardrails": [
                "Manifest must be marked implementation_variant=true.",
                "Manifest variant_name/case_name must match the requested variant.",
                "Build result target_name must match the requested variant target.",
                "Bitstream path must remain under the variant project bitstream directory.",
                "Measurement JSON path must remain under the variant measurements directory.",
                "If --expected-bitstream-sha256 is supplied, it must match the bitstream bytes.",
                "Existing measurement JSON is not overwritten unless --force is passed.",
            ],
        },
        "ledger_boundary": [
            "Do not write this variant measurement into current84/fullcombo/arch84 primitive ledgers.",
            "Do not update canonical e2e latency unless this variant is explicitly promoted.",
            "Do not use primitive or combo latency sums as e2e latency.",
        ],
    }
    write_json(root / "measurement_preflight.json", preflight)

    md = f"""# Measurement Preflight: {variant_name}

Target: `{readiness.get("target_name", "")}`

- Measurement scope: isolated implementation variant
- COM5 measurement run: no
- Measurement ready: `{readiness.get("measurement_ready", False)}`
- Measurement status: `{readiness.get("measurement_status", "")}`
- Strict latency usable: `{readiness.get("strict_latency_usable", False)}`
- WNS: `{readiness.get("wns_ns", "")}` ns
- Timing clean: `{readiness.get("timing_clean", False)}`
- Bitstream exists: `{readiness.get("bitstream_exists", False)}`
- Bitstream SHA256: `{readiness.get("bitstream_sha256", "")}`

## Artifacts

- Bitstream: `{readiness.get("bitstream", "")}`
- Build result: `{row.get("build_result_json", "")}`
- Harness manifest: `{str(manifest_path) if manifest_path.exists() else ""}`
- Timing summary: `{evidence.get("final_timing_summary_report", "")}`
- Routed timing summary: `{evidence.get("routed_timing_summary_report", "")}`
- Route status: `{evidence.get("routed_route_status_report", "")}`
- Utilization: `{evidence.get("final_utilization_report", "")}`

## Suggested Measurement Command

```powershell
{readiness.get("suggested_measure_command", "")}
```

## Boundary

- This command must write under this isolated variant directory.
- Do not declare e2e latency until the full-network harness measurement JSON exists.
- Strict latency usable requires UART `status_code=0` and WNS>=0.
- Do not back-fill primitive/fullcombo/current84/arch84 ledgers from this variant.
- The measure command checks variant manifest identity, bitstream path, JSON output path, optional bitstream SHA256 lock, and refuses accidental overwrite without `--force`.
"""
    (root / "measurement_preflight.md").write_text(md, encoding="utf-8")


def refresh_variants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if VARIANTS_ROOT.exists():
        for path in sorted(VARIANTS_ROOT.glob("*/build_result.json")):
            row = read_json(path)
            readiness = measurement_readiness_row(row)
            row.update(
                {
                    "timing_clean": readiness["timing_clean"],
                    "measurement_status": readiness["measurement_status"],
                    "measurement_ready": readiness["measurement_ready"],
                    "measurement_json_path": readiness["measurement_json_path"],
                    "strict_latency_usable": readiness["strict_latency_usable"],
                }
            )
            rows.append(row)
    readiness_rows = [measurement_readiness_row(row) for row in rows]
    readiness_by_name = {str(row.get("variant_name", "")): row for row in readiness_rows}
    for row in rows:
        write_measurement_preflight(row, readiness_by_name.get(str(row.get("variant_name", "")), {}))
    write_csv(VARIANTS_ROOT / "implementation_variants_status.csv", rows, STATUS_FIELDS)
    write_csv(VARIANTS_ROOT / "implementation_variants_measurement_readiness.csv", readiness_rows, MEASUREMENT_READY_FIELDS)
    write_json(
        VARIANTS_ROOT / "implementation_variants_summary.json",
        {
            "generated_at": timestamp(),
            "base_target_name": BASE_TARGET_NAME,
            "variant_count": len(rows),
            "success_count": sum(1 for row in rows if row.get("status") == "success"),
            "measurement_ready_count": sum(1 for row in readiness_rows if row.get("measurement_ready") is True),
            "strict_latency_usable_count": sum(1 for row in readiness_rows if row.get("strict_latency_usable") is True),
            "measurement_readiness_csv": str(VARIANTS_ROOT / "implementation_variants_measurement_readiness.csv"),
            "variants": rows,
            "measurement_readiness": readiness_rows,
        },
    )
    write_json(
        VARIANTS_ROOT / "implementation_variants_measurement_readiness.json",
        {
            "generated_at": timestamp(),
            "base_target_name": BASE_TARGET_NAME,
            "rows": readiness_rows,
        },
    )
    return rows


def main() -> int:
    args = parse_args()
    if args.mode in {"generate", "all"}:
        manifest = generate_variant(args)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.mode in {"build", "all"}:
        payload = build_variant(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.mode == "refresh":
        print(json.dumps(refresh_variants(), ensure_ascii=False, indent=2))
    if args.mode == "measure":
        payload = measure_variant(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
