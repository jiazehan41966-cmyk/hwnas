#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULT_DIR = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "sonar_classifier_end_to_end_board"
)
HARDWARE_PREFLIGHT_JSON = RESULT_DIR / "hardware_preflight_list_hw_targets.json"

ARCH_MANIFEST = REPO_ROOT / "results" / "strict40_expanded_top8_manifest_v1.json"
MULTISEED_SUMMARY = (
    REPO_ROOT
    / "results"
    / "strict40_expanded_top8_multiseed_eval10_seed42_43_44_v1"
    / "summary.json"
)
MULTISEED_REPORT = (
    REPO_ROOT
    / "results"
    / "strict40_expanded_top8_multiseed_eval10_seed42_43_44_v1"
    / "report.md"
)
SEARCH_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "formal_lut_strict40_expanded_nksid_rl200_eval10_explore_av7k325.yaml"
)

PRIMITIVE_SUMMARY = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_primitive60_board"
    / "v2_current84_primitive60_board_summary.json"
)
PRIMITIVE_OVERLAY = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_primitive60_board"
    / "board_latency_overlay_current84_primitive60.csv"
)
FULLCOMBO_SUMMARY = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_summary.json"
)
FULLCOMBO_RUNLIST = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_runlist.csv"
)
FULLCOMBO_STATUS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_board_status.csv"
)
FULLCOMBO_MEASUREMENTS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_board_measurements.csv"
)
FULLCOMBO_BLOCKERS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_blockers.csv"
)
FULLCOMBO_OVERLAY = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_current84_fullcombo_board"
    / "v2_current84_fullcombo84_latency_overlay.csv"
)
ARCH84_EXTENSION_SUMMARY = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_arch84_e2e_extension_board"
    / "arch84_e2e_extension_summary.json"
)
ARCH84_EXTENSION_RUNLIST = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_arch84_e2e_extension_board"
    / "arch84_e2e_extension_combo_runlist.csv"
)
ARCH84_EXTENSION_STATUS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_arch84_e2e_extension_board"
    / "arch84_e2e_extension_combo_status.csv"
)
ARCH84_EXTENSION_BLOCKERS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "v2_arch84_e2e_extension_board"
    / "arch84_e2e_extension_blockers.csv"
)
ARCH84_EXTENSION_PARAMS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "results"
    / "v2_arch84_e2e_extension"
    / "arch84_latency_only_parameter_manifest.json"
)
CURRENT84_COVERAGE = (
    REPO_ROOT / "hls_lut_builder" / "results" / "v2_current84" / "current84_combo_coverage.csv"
)
DEPLOYABLE_LUT_CURRENT84 = (
    REPO_ROOT
    / "hls_lut_builder"
    / "results"
    / "v2_current84_deployable_lut"
    / "deployable_lut_v2_current84.json"
)


RUNLIST_FIELDS = [
    "target_name",
    "mode",
    "architecture_source",
    "selected_rule",
    "candidate_id",
    "source_arch_id",
    "candidate_label",
    "architecture_mapping_status",
    "fullcombo_gate_status",
    "arch84_extension_gate_status",
    "primitive_gate_status",
    "planned_harness_kind",
    "board_execution_status",
    "notes",
]

TRACE_FIELDS = [
    "layer_index",
    "layer_role",
    "op",
    "kernel_size",
    "expand_ratio",
    "input_h",
    "output_h",
    "input_channels",
    "output_channels",
    "stride",
    "expected_current84_combo_query",
    "current84_combo_case_name",
    "current84_mapping_status",
    "board_measurement_namespace",
    "combo_decomposition",
    "component_roles",
    "component_case_names",
    "fullcombo_board_status",
    "fullcombo_uart_status_code",
    "fullcombo_wns_ns",
    "fullcombo_timing_clean",
    "prediction_baseline_cycles",
    "prediction_baseline_latency_ms",
    "notes",
]

STATUS_FIELDS = [
    "target_name",
    "status",
    "bitstream_status",
    "uart_status_code",
    "e2e_cycles",
    "e2e_latency_ms",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "timing_report_path",
    "blocker_count",
    "blocker_categories",
    "notes",
]

BASELINE_FIELDS = [
    "baseline_name",
    "baseline_type",
    "complete",
    "cycles",
    "latency_ms",
    "macro_f1",
    "top1",
    "lut",
    "dsp",
    "bram",
    "power_w",
    "source_path",
    "notes",
]

MEASUREMENT_FIELDS = [
    "target_name",
    "status_code",
    "cycles",
    "latency_ms",
    "word_count",
    "output_checksum",
    "classification_result",
    "measurement_json_path",
    "notes",
]

TIMING_FAILURE_FIELDS = [
    "target_name",
    "wns_ns",
    "uart_status_code",
    "status",
    "timing_report_path",
    "notes",
]

BLOCKER_FIELDS = [
    "blocker_category",
    "blocker_scope",
    "severity",
    "evidence_path",
    "evidence_detail",
    "required_resolution",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def div_up(value: int, divisor: int) -> int:
    return max(1, math.ceil(value / max(1, divisor)))


def select_architecture() -> dict[str, Any]:
    manifest = read_json(ARCH_MANIFEST)
    summary = read_json(MULTISEED_SUMMARY)
    ranked: list[dict[str, Any]] = []
    for label, info in summary["by_candidate"].items():
        ranked.append(
            {
                "candidate_label": label,
                "macro_f1_mean": info["macro_f1"]["mean"],
                "macro_f1_std_pop": info["macro_f1"]["std_pop"],
                "top1_mean": info["top1"]["mean"],
                "top1_std_pop": info["top1"]["std_pop"],
                "hardware": info["hardware"],
            }
        )
    ranked.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    selected_rank = ranked[0]
    candidates = manifest["candidates"]
    selected = next(
        candidate
        for candidate in candidates
        if candidate["candidate_label"] == selected_rank["candidate_label"]
    )
    return {
        "selection_rule": "rank1_by_multiseed_macro_f1_mean",
        "selected": selected,
        "multiseed_rank": 1,
        "multiseed_metrics": selected_rank,
        "manifest": manifest,
        "summary": summary,
    }


def find_combo(
    rows: list[dict[str, str]],
    family: str,
    query_fragments: list[str],
) -> dict[str, str] | None:
    for row in rows:
        if family and row.get("combo_family") != family:
            continue
        name = row.get("combo_case_name", "")
        if all(fragment in name for fragment in query_fragments):
            return row
    return None


def find_status(status_rows: list[dict[str, str]], combo_case_name: str) -> dict[str, str]:
    for row in status_rows:
        if row.get("combo_case_name") == combo_case_name:
            return row
    return {}


def find_extension_combo(
    rows: list[dict[str, str]],
    query_fragments: list[str],
) -> dict[str, str] | None:
    for row in rows:
        name = row.get("combo_case_name", "")
        if all(fragment in name for fragment in query_fragments):
            return row
    return None


def trace_architecture(
    selection: dict[str, Any],
    fullcombo_rows: list[dict[str, str]],
    status_rows: list[dict[str, str]],
    extension_rows: list[dict[str, str]],
    extension_status_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    arch = selection["selected"]["architecture"]
    trace: list[dict[str, Any]] = []
    layer_index = 0

    def append_layer(
        *,
        role: str,
        op: str,
        kernel: Any,
        expand: Any,
        input_h: int,
        output_h: int,
        input_channels: int,
        output_channels: int,
        stride: int,
        family: str,
        query: list[str],
        note: str = "",
    ) -> None:
        nonlocal layer_index
        layer_index += 1
        combo = find_combo(fullcombo_rows, family, query) if family else None
        namespace = "current84_fullcombo84" if combo else ""
        if combo:
            status = find_status(status_rows, combo.get("combo_case_name", ""))
            mapping_status = "mapped_to_current84_fullcombo"
        else:
            combo = find_extension_combo(extension_rows, query)
            status = find_status(extension_status_rows, combo.get("combo_case_name", "")) if combo else {}
            mapping_status = "mapped_to_arch84_extension" if combo else "missing_current84_fullcombo_case"
            namespace = "v2_arch84_e2e_extension" if combo else ""
        board_status = (
            status.get("board_execution_status")
            or status.get("measurement_status")
            or status.get("support_status")
            or status.get("status")
            or status.get("board_status")
            or ""
        )
        failure_category = status.get("failure_category", "")
        if failure_category and board_status:
            board_status = f"{board_status}:{failure_category}"
        elif failure_category:
            board_status = failure_category
        trace.append(
            {
                "layer_index": layer_index,
                "layer_role": role,
                "op": op,
                "kernel_size": kernel,
                "expand_ratio": expand,
                "input_h": input_h,
                "output_h": output_h,
                "input_channels": input_channels,
                "output_channels": output_channels,
                "stride": stride,
                "expected_current84_combo_query": "|".join(query),
                "current84_combo_case_name": combo.get("combo_case_name", "") if combo else "",
                "current84_mapping_status": mapping_status,
                "board_measurement_namespace": namespace,
                "combo_decomposition": combo.get("decomposition", "") if combo else "",
                "component_roles": combo.get("component_roles", "") if combo else "",
                "component_case_names": combo.get("component_case_names", "") if combo else "",
                "fullcombo_board_status": board_status,
                "fullcombo_uart_status_code": status.get("uart_status_code", ""),
                "fullcombo_wns_ns": status.get("wns_ns", ""),
                "fullcombo_timing_clean": status.get("timing_clean", ""),
                "prediction_baseline_cycles": combo.get("prediction_baseline_cycles", "") if combo else "",
                "prediction_baseline_latency_ms": combo.get("prediction_baseline_latency_ms", "") if combo else "",
                "notes": note,
            }
        )

    input_h = 224
    stem_stride = int(arch["stem_stride"])
    stem_channels = int(arch["stem_channels"])
    stem_out_h = div_up(input_h, stem_stride)
    append_layer(
        role="stem",
        op="stem_conv_k3_s2",
        kernel=3,
        expand=1,
        input_h=input_h,
        output_h=stem_out_h,
        input_channels=int(arch["input_channels"]),
        output_channels=stem_channels,
        stride=stem_stride,
        family="stem_conv_k3_s2",
        query=[f"stem_{input_h}_{arch['input_channels']}_{stem_channels}_s{stem_stride}"],
    )

    current_h = stem_out_h
    post_stride = int(arch.get("post_stem_downsample_stride") or 1)
    if post_stride > 1:
        current_h = div_up(current_h, post_stride)
        append_layer(
            role="post_stem_downsample",
            op="maxpool",
            kernel=3,
            expand="",
            input_h=stem_out_h,
            output_h=current_h,
            input_channels=stem_channels,
            output_channels=stem_channels,
            stride=post_stride,
            family="",
            query=["post_stem_downsample_not_in_current84_fullcombo"],
            note="software model layer; no current84 fullcombo row",
        )

    current_channels = stem_channels
    for stage_index, stage in enumerate(arch["stages"]):
        stage_channels = int(stage["channels"])
        for block_index, block in enumerate(stage["blocks"]):
            stride = int(block["stride"])
            out_h = div_up(current_h, stride)
            op = block["op"]
            kernel = int(block["kernel_size"])
            expand = int(block["expand_ratio"])
            if op == "conv":
                family = "pw_conv" if kernel == 1 else "conv"
                query = [f"pw_ir{stage_channels}_proj_{current_h}_{current_channels}_{stage_channels}"]
            elif op == "mbconv":
                family = f"mbconv_e{expand}_k{kernel}"
                query = [f"_{current_h}_{current_channels}_{stage_channels}_s{stride}_"]
            elif op == "skip":
                family = "skip"
                query = [f"skip_{current_h}_{current_channels}"]
            else:
                family = op
                query = [f"{op}_{current_h}_{current_channels}_{stage_channels}_s{stride}"]
            append_layer(
                role=f"stage{stage_index}_block{block_index}",
                op=op,
                kernel=kernel,
                expand=expand,
                input_h=current_h,
                output_h=out_h,
                input_channels=current_channels,
                output_channels=stage_channels,
                stride=stride,
                family=family,
                query=query,
                note="search architecture stage/block layer",
            )
            current_h = out_h
            current_channels = stage_channels

    head_conv_channels = arch.get("head_conv_channels")
    if head_conv_channels:
        head_conv_channels = int(head_conv_channels)
        append_layer(
            role="head_conv",
            op="head_conv1x1",
            kernel=1,
            expand=1,
            input_h=current_h,
            output_h=current_h,
            input_channels=current_channels,
            output_channels=head_conv_channels,
            stride=1,
            family="pw_conv",
            query=[f"pw_head_{current_h}_{current_channels}_{head_conv_channels}"],
            note="optional conv head from architecture config",
        )
        current_channels = head_conv_channels

    append_layer(
        role="global_avg_pool",
        op="global_avg_pool",
        kernel="",
        expand="",
        input_h=current_h,
        output_h=1,
        input_channels=current_channels,
        output_channels=current_channels,
        stride=1,
        family="global_avg_pool",
        query=[f"gap_{current_h}_{current_channels}"],
        note="architecture head GAP",
    )

    head_channels = arch.get("head_channels")
    num_classes = int(arch["num_classes"])
    if head_channels:
        append_layer(
            role="fc_hidden",
            op="fc_layer",
            kernel="",
            expand="",
            input_h=1,
            output_h=1,
            input_channels=current_channels,
            output_channels=int(head_channels),
            stride=1,
            family="fc_layer",
            query=[f"fc_1_{current_channels}_{int(head_channels)}"],
            note="optional hidden classifier layer",
        )
        append_layer(
            role="fc_classifier",
            op="fc_layer",
            kernel="",
            expand="",
            input_h=1,
            output_h=1,
            input_channels=int(head_channels),
            output_channels=num_classes,
            stride=1,
            family="fc_layer",
            query=[f"fc_1_{int(head_channels)}_{num_classes}"],
            note="classifier output layer",
        )
    else:
        append_layer(
            role="fc_classifier",
            op="fc_layer",
            kernel="",
            expand="",
            input_h=1,
            output_h=1,
            input_channels=current_channels,
            output_channels=num_classes,
            stride=1,
            family="fc_layer",
            query=[f"fc_1_{current_channels}_{num_classes}"],
            note="classifier output layer",
        )

    return trace


def summarize_fullcombo_gate(fullcombo_summary: dict[str, Any], measurement_rows: list[dict[str, str]]) -> str:
    counts = fullcombo_summary.get("counts", {})
    clean = int(counts.get("board_measured_timing_clean", 0) or 0)
    uart_tfail = int(counts.get("board_measured_uart_ok_timing_fail", 0) or 0)
    if clean + uart_tfail == 84 and len(measurement_rows) == 84:
        return "complete"
    if fullcombo_summary.get("pilot_gate_blocked"):
        return "blocked_by_fullcombo_pilot_gate"
    return "incomplete"


def summarize_arch84_extension_gate(extension_summary: dict[str, Any], status_rows: list[dict[str, str]]) -> str:
    if not extension_summary:
        return "missing"
    measured = [
        row
        for row in status_rows
        if row.get("status") in {"board_measured_timing_clean", "board_measured_uart_ok_timing_fail"}
    ]
    if len(status_rows) == 3 and len(measured) == 3:
        return "complete"
    if extension_summary.get("blocker_count"):
        return "blocked"
    return "incomplete"


def build_prediction_baselines(
    selection: dict[str, Any],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = selection["selected"]
    observed = selected
    multiseed = selection["multiseed_metrics"]
    mapped_rows = [row for row in trace_rows if row["current84_mapping_status"] == "mapped_to_current84_fullcombo"]
    non_current84_rows = [
        row for row in trace_rows if row["current84_mapping_status"] != "mapped_to_current84_fullcombo"
    ]

    mapped_cycles = 0
    mapped_ms = 0.0
    for row in mapped_rows:
        if row.get("prediction_baseline_cycles"):
            mapped_cycles += int(float(row["prediction_baseline_cycles"]))
        if row.get("prediction_baseline_latency_ms"):
            mapped_ms += float(row["prediction_baseline_latency_ms"])

    return [
        {
            "baseline_name": "search_observed_single_run_cost_estimator",
            "baseline_type": "prediction_only_not_board_measurement",
            "complete": "true",
            "cycles": "",
            "latency_ms": observed.get("observed_latency_ms", ""),
            "macro_f1": observed.get("observed_macro_f1", ""),
            "top1": observed.get("observed_top1", ""),
            "lut": observed.get("observed_lut", ""),
            "dsp": observed.get("observed_dsp", ""),
            "bram": observed.get("observed_bram", ""),
            "power_w": observed.get("observed_power_w", ""),
            "source_path": rel(ARCH_MANIFEST),
            "notes": "source candidate metrics; not end-to-end board latency",
        },
        {
            "baseline_name": "multiseed_mean_proxy_metrics",
            "baseline_type": "prediction_only_not_board_measurement",
            "complete": "true",
            "cycles": "",
            "latency_ms": multiseed["hardware"]["latency_ms"],
            "macro_f1": multiseed["macro_f1_mean"],
            "top1": multiseed["top1_mean"],
            "lut": multiseed["hardware"]["lut"],
            "dsp": multiseed["hardware"]["dsp"],
            "bram": multiseed["hardware"]["bram"],
            "power_w": multiseed["hardware"]["power_w"],
            "source_path": rel(MULTISEED_SUMMARY),
            "notes": "ranked by multi-seed macro_f1 mean; hardware latency remains estimator metadata",
        },
        {
            "baseline_name": "mapped_current84_combo_prediction_sum_partial",
            "baseline_type": "prediction_only_not_board_measurement",
            "complete": "false" if non_current84_rows else "true",
            "cycles": mapped_cycles,
            "latency_ms": f"{mapped_ms:.6f}",
            "macro_f1": "",
            "top1": "",
            "lut": "",
            "dsp": "",
            "bram": "",
            "power_w": "",
            "source_path": rel(FULLCOMBO_RUNLIST),
            "notes": (
                f"sum over mapped current84 planning rows only; non_current84_rows={len(non_current84_rows)}; "
                "not a measured full-network latency"
            ),
        },
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    primitive_summary = read_json(PRIMITIVE_SUMMARY)
    fullcombo_summary = read_json(FULLCOMBO_SUMMARY)
    fullcombo_rows = read_csv(FULLCOMBO_RUNLIST)
    fullcombo_status_rows = read_csv(FULLCOMBO_STATUS)
    fullcombo_measurement_rows = read_csv(FULLCOMBO_MEASUREMENTS)
    fullcombo_blocker_rows = read_csv(FULLCOMBO_BLOCKERS)
    current84_coverage_rows = read_csv(CURRENT84_COVERAGE)
    primitive_overlay_rows = read_csv(PRIMITIVE_OVERLAY)
    fullcombo_overlay_rows = read_csv(FULLCOMBO_OVERLAY)
    extension_summary = read_json(ARCH84_EXTENSION_SUMMARY) if ARCH84_EXTENSION_SUMMARY.exists() else {}
    extension_rows = read_csv(ARCH84_EXTENSION_RUNLIST)
    extension_status_rows = read_csv(ARCH84_EXTENSION_STATUS)
    extension_blocker_rows = read_csv(ARCH84_EXTENSION_BLOCKERS)
    extension_parameter_manifest = read_json(ARCH84_EXTENSION_PARAMS) if ARCH84_EXTENSION_PARAMS.exists() else {}
    hardware_preflight = read_json(HARDWARE_PREFLIGHT_JSON) if HARDWARE_PREFLIGHT_JSON.exists() else {}

    selection = select_architecture()
    selected = selection["selected"]
    trace_rows = trace_architecture(
        selection,
        fullcombo_rows,
        fullcombo_status_rows,
        extension_rows,
        extension_status_rows,
    )
    missing_mapping = [
        row
        for row in trace_rows
        if row["current84_mapping_status"]
        not in {"mapped_to_current84_fullcombo", "mapped_to_arch84_extension"}
    ]

    primitive_counts = primitive_summary.get("counts", {})
    fullcombo_gate = summarize_fullcombo_gate(fullcombo_summary, fullcombo_measurement_rows)
    arch84_extension_gate = summarize_arch84_extension_gate(extension_summary, extension_status_rows)
    primitive_gate = (
        "complete"
        if int(primitive_counts.get("total_entries", 0) or 0) == 60
        and int(primitive_counts.get("uart_ok", 0) or 0) == 60
        else "incomplete"
    )

    mapping_status = "complete_with_arch84_extension" if not missing_mapping else "blocked_by_architecture_mapping_gaps"
    blocker_rows: list[dict[str, str]] = []
    if primitive_gate != "complete":
        blocker_rows.append(
            {
                "blocker_category": "primitive60_board_not_complete",
                "blocker_scope": "e2e_execution_gate",
                "severity": "blocking",
                "evidence_path": rel(PRIMITIVE_SUMMARY),
                "evidence_detail": (
                    f"primitive_gate={primitive_gate}; "
                    f"counts={json.dumps(primitive_summary.get('counts', {}), sort_keys=True)}"
                ),
                "required_resolution": "complete current84 primitive60 board evidence before e2e bitstream/measurement",
            }
        )
    if fullcombo_gate != "complete":
        blocker_rows.append(
            {
                "blocker_category": "fullcombo_board_not_complete",
                "blocker_scope": "e2e_execution_gate",
                "severity": "blocking",
                "evidence_path": rel(FULLCOMBO_SUMMARY),
                "evidence_detail": (
                    f"fullcombo_gate={fullcombo_gate}; measurement_rows={len(fullcombo_measurement_rows)}; "
                    f"counts={json.dumps(fullcombo_summary.get('counts', {}), sort_keys=True)}"
                ),
                "required_resolution": "complete current84 fullcombo84 pilot and real COM5 measurements before e2e bitstream/measurement",
            }
        )
    if arch84_extension_gate != "complete":
        blocker_rows.append(
            {
                "blocker_category": "arch84_extension_board_not_complete",
                "blocker_scope": "e2e_execution_gate",
                "severity": "blocking",
                "evidence_path": rel(ARCH84_EXTENSION_SUMMARY),
                "evidence_detail": (
                    f"arch84_extension_gate={arch84_extension_gate}; "
                    f"status_rows={len(extension_status_rows)}; "
                    f"summary={json.dumps(extension_summary, sort_keys=True)}"
                ),
                "required_resolution": "close 5 extension primitives and measure 3 arch84 extension combos before full e2e harness run",
            }
        )
    if hardware_preflight and hardware_preflight.get("status") != "ok":
        hw_detail = hardware_preflight.get("failure_reason", "")
        vivado_hw = hardware_preflight.get("vivado_hardware")
        if isinstance(vivado_hw, dict):
            hw_detail = (
                f"returncode={vivado_hw.get('returncode')}; "
                f"targets={len(vivado_hw.get('hw_targets') or [])}; "
                f"devices={len(vivado_hw.get('hw_devices') or [])}; "
                f"hw_server_url={vivado_hw.get('hw_server_url')}"
            )
        blocker_rows.append(
            {
                "blocker_category": "vivado_hw_target_not_visible",
                "blocker_scope": "hardware_preflight",
                "severity": "blocking",
                "evidence_path": rel(HARDWARE_PREFLIGHT_JSON),
                "evidence_detail": hw_detail,
                "required_resolution": "connect/power AV7K325 JTAG target and rerun measure_op.py --list-hw-targets before board measurements",
            }
        )
    fullcombo_execution_blockers = [
        row
        for row in fullcombo_blocker_rows
        if row.get("failure_category")
        and row.get("failure_category") not in {"timing_violation", "pilot_gate_blocked"}
    ]
    if fullcombo_gate != "complete":
        for row in fullcombo_execution_blockers:
            blocker_rows.append(
                {
                    "blocker_category": "fullcombo_execution_blocker",
                    "blocker_scope": row.get("combo_case_name", "fullcombo_case"),
                    "severity": "blocking",
                    "evidence_path": row.get("measurement_json_path") or rel(FULLCOMBO_BLOCKERS),
                    "evidence_detail": (
                        f"status={row.get('status', '')}; category={row.get('failure_category', '')}; "
                        f"reason={row.get('failure_reason', '')}; wns_ns={row.get('wns_ns', '')}; "
                        f"stdout_log={row.get('stdout_log', '')}"
                    ),
                    "required_resolution": "resolve fullcombo runtime/build blockers and rerun affected COM5 measurements",
                }
            )
    if missing_mapping:
        blocker_rows.append(
            {
                "blocker_category": "architecture_to_current84_mapping_incomplete",
                "blocker_scope": "e2e_harness_planning",
                "severity": "blocking",
                "evidence_path": rel(FULLCOMBO_RUNLIST),
                "evidence_detail": "; ".join(
                    f"L{row['layer_index']}:{row['layer_role']}:{row['expected_current84_combo_query']}"
                    for row in missing_mapping
                ),
                "required_resolution": "add missing current84 or arch84 extension combo cases before e2e harness build",
            }
        )

    board_status = "planning_only_blocked" if blocker_rows else "planning_only_ready_for_e2e_execution"
    blocker_categories = sorted({row["blocker_category"] for row in blocker_rows})
    bitstream_status = (
        "not_run_blocked_by_gate" if blocker_rows else "not_run_ready_for_e2e_harness_build"
    )
    execution_reason = (
        "e2e measurement was not run because blocking gates remain: "
        + ", ".join(blocker_categories)
        if blocker_rows
        else "all planning gates are clear; this planning-only script does not build or measure the e2e harness"
    )

    target_name = f"sonar_classifier_{selected['candidate_id']}_{selected['source_arch_id']}_e2e"
    runlist_rows = [
        {
            "target_name": target_name,
            "mode": "planning_only",
            "architecture_source": rel(ARCH_MANIFEST),
            "selected_rule": selection["selection_rule"],
            "candidate_id": selected["candidate_id"],
            "source_arch_id": selected["source_arch_id"],
            "candidate_label": selected["candidate_label"],
            "architecture_mapping_status": mapping_status,
            "fullcombo_gate_status": fullcombo_gate,
            "arch84_extension_gate_status": arch84_extension_gate,
            "primitive_gate_status": primitive_gate,
            "planned_harness_kind": "single_full_network_harness_after_gates_clear",
            "board_execution_status": board_status,
            "notes": "No e2e bitstream or COM5 measurement executed in this planning-only gate.",
        }
    ]

    status_rows = [
        {
            "target_name": target_name,
            "status": board_status,
            "bitstream_status": bitstream_status,
            "uart_status_code": "",
            "e2e_cycles": "",
            "e2e_latency_ms": "",
            "wns_ns": "",
            "timing_clean": "false",
            "measurement_json_path": "",
            "bitstream_path": "",
            "timing_report_path": "",
            "blocker_count": len(blocker_rows),
            "blocker_categories": "|".join(blocker_categories),
            "notes": (
                "End-to-end latency is not declared. Timing-clean requires UART status_code=0 and WNS>=0."
            ),
        }
    ]

    baseline_rows = build_prediction_baselines(selection, trace_rows)
    measurements_rows: list[dict[str, Any]] = []
    timing_failure_rows: list[dict[str, Any]] = []

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "planning_only",
        "output_dir": rel(RESULT_DIR),
        "target_name": target_name,
        "architecture": {
            "selection_rule": selection["selection_rule"],
            "candidate_id": selected["candidate_id"],
            "source_arch_id": selected["source_arch_id"],
            "candidate_label": selected["candidate_label"],
            "candidate_key": selected["candidate_key"],
            "architecture_source": rel(ARCH_MANIFEST),
            "search_config": rel(SEARCH_CONFIG),
            "multiseed_summary": rel(MULTISEED_SUMMARY),
            "multiseed_report": rel(MULTISEED_REPORT),
            "observed_metrics": {
                "macro_f1": selected["observed_macro_f1"],
                "top1": selected["observed_top1"],
                "latency_ms": selected["observed_latency_ms"],
                "lut": selected["observed_lut"],
                "dsp": selected["observed_dsp"],
                "bram": selected["observed_bram"],
                "power_w": selected["observed_power_w"],
            },
            "multiseed_metrics": selection["multiseed_metrics"],
        },
        "board_gates": {
            "primitive60": {
                "gate": primitive_gate,
                "counts": primitive_summary.get("counts", {}),
                "summary_path": rel(PRIMITIVE_SUMMARY),
                "overlay_rows": len(primitive_overlay_rows),
            },
            "fullcombo84": {
                "gate": fullcombo_gate,
                "counts": fullcombo_summary.get("counts", {}),
                "summary_path": rel(FULLCOMBO_SUMMARY),
                "measurement_rows": len(fullcombo_measurement_rows),
                "blocker_rows": len(fullcombo_blocker_rows),
                "overlay_rows": len(fullcombo_overlay_rows),
            },
            "arch84_e2e_extension": {
                "gate": arch84_extension_gate,
                "summary_path": rel(ARCH84_EXTENSION_SUMMARY),
                "status_rows": len(extension_status_rows),
                "blocker_rows": len(extension_blocker_rows),
                "parameter_manifest": rel(ARCH84_EXTENSION_PARAMS),
                "parameter_mode": extension_parameter_manifest.get("parameter_mode", ""),
                "parameter_manifest_hash_sha256": extension_parameter_manifest.get("manifest_hash_sha256", ""),
            },
            "hardware_preflight": {
                "path": rel(HARDWARE_PREFLIGHT_JSON),
                "status": hardware_preflight.get("status", "not_run") if hardware_preflight else "not_run",
            },
            "current84_combo_coverage": {
                "rows": len(current84_coverage_rows),
                "source_path": rel(CURRENT84_COVERAGE),
            },
        },
        "architecture_trace": {
            "rows": len(trace_rows),
            "mapped_rows": len(trace_rows) - len(missing_mapping),
            "missing_mapping_rows": len(missing_mapping),
            "mapping_status": mapping_status,
        },
        "execution": {
            "bitstream_build": "not_run",
            "measurement": "not_run",
            "uart_status_code": None,
            "e2e_cycles": None,
            "e2e_latency_ms": None,
            "wns_ns": None,
            "timing_clean": False,
            "reason": execution_reason,
        },
        "blockers": blocker_rows,
        "inputs": {
            rel(ARCH_MANIFEST): sha256(ARCH_MANIFEST),
            rel(MULTISEED_SUMMARY): sha256(MULTISEED_SUMMARY),
            rel(SEARCH_CONFIG): sha256(SEARCH_CONFIG),
            rel(PRIMITIVE_SUMMARY): sha256(PRIMITIVE_SUMMARY),
            rel(PRIMITIVE_OVERLAY): sha256(PRIMITIVE_OVERLAY),
            rel(FULLCOMBO_SUMMARY): sha256(FULLCOMBO_SUMMARY),
            rel(FULLCOMBO_RUNLIST): sha256(FULLCOMBO_RUNLIST),
            rel(FULLCOMBO_STATUS): sha256(FULLCOMBO_STATUS),
            rel(FULLCOMBO_MEASUREMENTS): sha256(FULLCOMBO_MEASUREMENTS),
            rel(FULLCOMBO_BLOCKERS): sha256(FULLCOMBO_BLOCKERS),
            rel(FULLCOMBO_OVERLAY): sha256(FULLCOMBO_OVERLAY),
            rel(ARCH84_EXTENSION_SUMMARY): sha256(ARCH84_EXTENSION_SUMMARY),
            rel(ARCH84_EXTENSION_STATUS): sha256(ARCH84_EXTENSION_STATUS),
            rel(ARCH84_EXTENSION_BLOCKERS): sha256(ARCH84_EXTENSION_BLOCKERS),
            rel(ARCH84_EXTENSION_PARAMS): sha256(ARCH84_EXTENSION_PARAMS),
            rel(HARDWARE_PREFLIGHT_JSON): sha256(HARDWARE_PREFLIGHT_JSON),
            rel(CURRENT84_COVERAGE): sha256(CURRENT84_COVERAGE),
            rel(DEPLOYABLE_LUT_CURRENT84): sha256(DEPLOYABLE_LUT_CURRENT84),
        },
        "outputs": {
            "runlist_csv": rel(RESULT_DIR / "sonar_classifier_e2e_runlist.csv"),
            "architecture_trace_csv": rel(RESULT_DIR / "sonar_classifier_e2e_architecture_trace.csv"),
            "status_csv": rel(RESULT_DIR / "sonar_classifier_e2e_status.csv"),
            "prediction_baseline_csv": rel(RESULT_DIR / "sonar_classifier_e2e_prediction_baseline.csv"),
            "measurements_csv": rel(RESULT_DIR / "sonar_classifier_e2e_measurements.csv"),
            "timing_failures_csv": rel(RESULT_DIR / "sonar_classifier_e2e_timing_failures.csv"),
            "blockers_csv": rel(RESULT_DIR / "sonar_classifier_e2e_blockers.csv"),
            "summary_json": rel(RESULT_DIR / "sonar_classifier_e2e_summary.json"),
            "latency_overlay_json": rel(RESULT_DIR / "sonar_classifier_e2e_latency_overlay.json"),
            "harness_spec_json": rel(RESULT_DIR / "sonar_classifier_e2e_harness_spec.json"),
            "readme": rel(RESULT_DIR / "README.md"),
        },
    }

    harness_spec = {
        "target_name": target_name,
        "harness_kind": "single_full_network_latency_only_harness",
        "status": "not_generated_until_gates_complete" if blocker_rows else "ready_to_generate",
        "required_gates": {
            "primitive60": primitive_gate,
            "fullcombo84": fullcombo_gate,
            "arch84_e2e_extension": arch84_extension_gate,
            "architecture_mapping": mapping_status,
        },
        "topology": [
            {
                "layer_index": row["layer_index"],
                "layer_role": row["layer_role"],
                "op": row["op"],
                "combo_case_name": row["current84_combo_case_name"],
                "measurement_namespace": row["board_measurement_namespace"],
                "decomposition": row["combo_decomposition"],
                "component_roles": row["component_roles"],
                "component_case_names": row["component_case_names"],
            }
            for row in trace_rows
        ],
        "streaming": {
            "layer_boundary": "stream_fifo_between_layers",
            "output_word_count": 8,
            "uart_protocol": "status_code, cycles, checksum, word_count",
        },
        "parameter_manifest": rel(ARCH84_EXTENSION_PARAMS),
        "parameter_mode": extension_parameter_manifest.get("parameter_mode", "latency_only_deterministic"),
        "accuracy_claim": "none",
        "notes": [
            "This spec is reproducible planning evidence for the complete-network harness.",
            "Actual e2e latency remains unavailable until one complete-network bitstream is built and measured over COM5.",
        ],
    }

    latency_overlay = {
        "target_name": target_name,
        "mode": "planning_only",
        "e2e_board_measurement": {
            "available": False,
            "uart_status_code": None,
            "cycles": None,
            "latency_ms": None,
            "wns_ns": None,
            "timing_clean": False,
            "measurement_json_path": "",
            "bitstream_path": "",
            "timing_report_path": "",
        },
        "prediction_baselines": baseline_rows,
        "architecture_trace_csv": rel(RESULT_DIR / "sonar_classifier_e2e_architecture_trace.csv"),
        "delta_vs_prediction": "not_applicable_without_e2e_board_measurement",
        "strict_board_latency_usable": False,
        "notes": [
            "End-to-end latency must come from one complete network harness COM5 measurement.",
            "Primitive and combo latencies are prediction/reference only here.",
            "Timing-clean requires UART status_code=0 and WNS>=0.",
        ],
    }

    gate_explanation = (
        "No end-to-end bitstream build or COM5 measurement was run in this bundle, because "
        f"blocking gates remain: {', '.join(blocker_categories)}."
        if blocker_rows
        else "No end-to-end bitstream build or COM5 measurement was run in this planning bundle; "
        "the board evidence gates are clear for the next execution stage."
    )
    blocker_explanation = (
        f"See `sonar_classifier_e2e_blockers.csv`. Current blocking categories: {', '.join(blocker_categories)}."
        if blocker_rows
        else "No blocking e2e planning gates remain in `sonar_classifier_e2e_blockers.csv`."
    )

    readme = f"""# Sonar Classifier End-to-End Board Planning

This directory is the AV7K325 end-to-end board validation planning bundle for the sonar classifier.

## Gate status

- Mode: planning_only
- Target: `{target_name}`
- Architecture source: `{rel(ARCH_MANIFEST)}`
- Selection rule: `{selection['selection_rule']}`
- Primitive60 board gate: `{primitive_gate}`
- Fullcombo84 board gate: `{fullcombo_gate}`
- arch84 extension board gate: `{arch84_extension_gate}`
- Current84 combo coverage rows: `{len(current84_coverage_rows)}`

{gate_explanation} The existing fullcombo measurement table has `{len(fullcombo_measurement_rows)}` measured rows.

## Selected architecture

- Candidate: `{selected['candidate_id']}` / `{selected['source_arch_id']}`
- Label: `{selected['candidate_label']}`
- Observed single-run macro_f1/top1: `{selected['observed_macro_f1']}` / `{selected['observed_top1']}`
- Multi-seed macro_f1_mean/top1_mean: `{selection['multiseed_metrics']['macro_f1_mean']}` / `{selection['multiseed_metrics']['top1_mean']}`
- Estimator latency/resource metadata: latency_ms=`{selection['multiseed_metrics']['hardware']['latency_ms']}`, LUT=`{selection['multiseed_metrics']['hardware']['lut']}`, DSP=`{selection['multiseed_metrics']['hardware']['dsp']}`, BRAM=`{selection['multiseed_metrics']['hardware']['bram']}`, power_w=`{selection['multiseed_metrics']['hardware']['power_w']}`

## Important scope notes

- End-to-end latency must be measured by a single complete-network harness on AV7K325.
- Primitive latency and combo latency are prediction/reference baselines only.
- Combo latency sums are not reported as true end-to-end board latency.
- If the end-to-end run is not timing-clean, it cannot be used as strict board latency.
- Strict timing-clean means UART `status_code=0` and WNS >= 0.
- The e2e parameter mode is latency-only deterministic. Classification output is only checksum/argmax
  channel sanity and is not a trained sonar accuracy claim.
- This bundle does not modify deployable LUT files or old board measured LUT files.

## Blockers

{blocker_explanation}
"""

    write_csv(RESULT_DIR / "sonar_classifier_e2e_runlist.csv", runlist_rows, RUNLIST_FIELDS)
    write_csv(RESULT_DIR / "sonar_classifier_e2e_architecture_trace.csv", trace_rows, TRACE_FIELDS)
    write_csv(RESULT_DIR / "sonar_classifier_e2e_status.csv", status_rows, STATUS_FIELDS)
    write_csv(
        RESULT_DIR / "sonar_classifier_e2e_prediction_baseline.csv",
        baseline_rows,
        BASELINE_FIELDS,
    )
    write_csv(
        RESULT_DIR / "sonar_classifier_e2e_measurements.csv",
        measurements_rows,
        MEASUREMENT_FIELDS,
    )
    write_csv(
        RESULT_DIR / "sonar_classifier_e2e_timing_failures.csv",
        timing_failure_rows,
        TIMING_FAILURE_FIELDS,
    )
    write_csv(RESULT_DIR / "sonar_classifier_e2e_blockers.csv", blocker_rows, BLOCKER_FIELDS)
    write_json(RESULT_DIR / "sonar_classifier_e2e_summary.json", summary)
    write_json(RESULT_DIR / "sonar_classifier_e2e_latency_overlay.json", latency_overlay)
    write_json(RESULT_DIR / "sonar_classifier_e2e_harness_spec.json", harness_spec)
    (RESULT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
