#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.hardware import parse_vivado_timing_summary_text, parse_vivado_utilization_text


DEFAULT_INPUT_MANIFESTS = [
    BUILDER_ROOT / "results" / "stageB_core_pilot_hls_manifest.yaml",
    BUILDER_ROOT / "results" / "stageB_ext_partial_hls_manifest.yaml",
    BUILDER_ROOT / "results" / "stageB_ext_p_ext_hls_manifest.yaml",
]

DEFAULT_OPERATOR_MANIFEST = BUILDER_ROOT / "configs" / "operator_manifest.yaml"
DEFAULT_BOARD_RESULTS = BUILDER_ROOT / "board_harness" / "results" / "board_measured_lut.csv"
DEFAULT_OUTPUT_CSV = BUILDER_ROOT / "results" / "three_tier_operator_summary.csv"
DEFAULT_OUTPUT_MANIFEST = BUILDER_ROOT / "results" / "three_tier_operator_summary_manifest.yaml"
DEFAULT_OUTPUT_JSON = BUILDER_ROOT / "results" / "three_tier_operator_summary.json"
DEFAULT_EXTENSION_MD = BUILDER_ROOT / "docs" / "extension_operator_disposition.md"

CSV_FIELDS = [
    "case_name",
    "op_id",
    "op_type",
    "operator_group",
    "input_resolution",
    "Cin",
    "Cout",
    "K",
    "stride",
    "groups",
    "expand",
    "target_clock_mhz",
    "measurement_contract_id",
    "hls_cycles",
    "hls_latency_ms",
    "hls_ii",
    "hls_depth",
    "hls_fmax_mhz",
    "hls_lut",
    "hls_ff",
    "hls_bram18",
    "hls_dsp",
    "vivado_status",
    "vivado_post_route_fmax_mhz",
    "vivado_post_route_wns_ns",
    "vivado_post_route_lut",
    "vivado_post_route_ff",
    "vivado_post_route_bram_tile",
    "vivado_post_route_ramb18",
    "vivado_post_route_dsp",
    "board_measured",
    "board_cycles",
    "board_latency_ms",
    "board_checksum",
    "board_word_count",
    "board_timing_wns_ns",
    "board_timing_met_200mhz",
    "board_lut",
    "board_ff",
    "board_bram_tile",
    "board_dsp",
    "search_enabled",
    "search_policy_default_enabled",
    "policy_deployment_status",
    "policy_screening_status",
    "policy_reason",
    "deployability_label",
    "deployability_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HLS, Vivado downstream, and board-harness results into one operator summary."
    )
    parser.add_argument(
        "--input-manifest",
        action="append",
        dest="input_manifests",
        default=None,
        help="Manifest(s) with hls_estimate + vivado_actual entries. Defaults to core + extension pilot manifests.",
    )
    parser.add_argument(
        "--operator-manifest",
        default=str(DEFAULT_OPERATOR_MANIFEST),
        help="Operator manifest with search/deployment policy declarations.",
    )
    parser.add_argument(
        "--board-results",
        default=str(DEFAULT_BOARD_RESULTS),
        help="Board measurement CSV produced by measure_op.py / batch_measure_ops.py.",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--extension-doc", default=str(DEFAULT_EXTENSION_MD))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifests = [Path(path).resolve() for path in (args.input_manifests or DEFAULT_INPUT_MANIFESTS)]
    operator_manifest_path = Path(args.operator_manifest).resolve()
    board_results_path = Path(args.board_results).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_json = Path(args.output_json).resolve()
    extension_doc = Path(args.extension_doc).resolve()

    operator_manifest = _load_yaml(operator_manifest_path)
    measurement_contract = operator_manifest.get("measurement_contract", {})
    operator_status = operator_manifest.get("operator_status", {})
    search_policy = operator_manifest.get("search_policy", {})
    core_ops = set((operator_manifest.get("operators", {}) or {}).get("core", []))
    extension_ops = set((operator_manifest.get("operators", {}) or {}).get("extensions", []))

    entries = _load_entries(input_manifests)
    board_results = _load_board_results(board_results_path)

    rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    summary_counts: dict[str, int] = {}

    for case_name in sorted(entries):
        entry = entries[case_name]
        op_spec = entry.get("op_spec", {})
        metadata = entry.get("metadata", {})
        op_type = str(op_spec.get("op") or metadata.get("op_type") or "")
        policy = operator_status.get(op_type, {})
        board_actual = board_results.get(case_name)

        row = _build_row(
            case_name=case_name,
            entry=entry,
            board_actual=board_actual,
            core_ops=core_ops,
            extension_ops=extension_ops,
            search_policy=search_policy,
            policy=policy,
        )
        rows.append(row)
        manifest_entries.append(_build_manifest_entry(entry, board_actual, row))
        summary_counts[row["deployability_label"]] = summary_counts.get(row["deployability_label"], 0) + 1

    _write_csv(output_csv, rows)
    _write_yaml(
        output_manifest,
        {
            "generated_at": datetime.now().astimezone().isoformat(),
            "measurement_contract": measurement_contract,
            "entries": manifest_entries,
        },
    )

    summary_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_manifests": [str(path) for path in input_manifests if path.exists()],
        "operator_manifest": str(operator_manifest_path),
        "board_results": str(board_results_path),
        "entries_total": len(rows),
        "board_measured_cases": sum(1 for row in rows if row["board_measured"] == "yes"),
        "deployability_counts": summary_counts,
        "output_csv": str(output_csv),
        "output_manifest": str(output_manifest),
        "extension_doc": str(extension_doc),
    }
    _write_json(output_json, summary_payload)
    _write_extension_doc(extension_doc, rows, operator_status)

    print(f"Summarized {len(rows)} operator cases.")
    print(f"CSV: {output_csv}")
    print(f"Manifest: {output_manifest}")
    print(f"Summary JSON: {output_json}")
    print(f"Extension doc: {extension_doc}")


def _build_row(
    *,
    case_name: str,
    entry: dict[str, Any],
    board_actual: dict[str, Any] | None,
    core_ops: set[str],
    extension_ops: set[str],
    search_policy: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    op_spec = entry.get("op_spec", {})
    metadata = entry.get("metadata", {})
    op_type = str(op_spec.get("op") or "")
    hls = (entry.get("hls_estimate", {}) or {}).get("metrics", {}) or {}
    vivado_actual = entry.get("vivado_actual", {}) or {}
    vivado_metrics = (vivado_actual.get("metrics", {}) or {}).get("post_route", {}) or {}

    input_resolution = op_spec.get("input_resolution") or [0, 0]
    input_resolution_str = f"{int(input_resolution[0])}x{int(input_resolution[1])}" if len(input_resolution) == 2 else ""
    operator_group = "core" if op_type in core_ops else "extension" if op_type in extension_ops else "other"

    board_measurement = (board_actual or {}).get("measurement", {})
    board_timing = (board_actual or {}).get("timing", {})
    board_util = (board_actual or {}).get("utilization", {})

    search_enabled = _coerce_bool(policy.get("search_enabled"), default=True)
    search_default = op_type in set(search_policy.get("default_mainline_search_ops", []))
    policy_deployment_status = str(policy.get("deployment_status", "ready"))
    policy_screening_status = str(policy.get("screening_status", "unknown"))
    policy_reason = str(policy.get("reason", ""))

    deployability_label, deployability_reason = _classify_case(
        policy=policy,
        vivado_metrics=vivado_metrics,
        board_timing=board_timing,
        board_measurement=board_measurement,
    )

    row = {
        "case_name": case_name,
        "op_id": metadata.get("op_id", ""),
        "op_type": op_type,
        "operator_group": operator_group,
        "input_resolution": input_resolution_str,
        "Cin": int(op_spec.get("in_channels", 0)),
        "Cout": int(op_spec.get("out_channels", 0)),
        "K": int(op_spec.get("kernel_size", 1)),
        "stride": int(op_spec.get("stride", 1)),
        "groups": int(op_spec.get("groups", 1)),
        "expand": int(op_spec.get("expand_ratio", 1)),
        "target_clock_mhz": float(entry.get("clock_mhz") or op_spec.get("target_clock_mhz") or 0.0),
        "measurement_contract_id": metadata.get("measurement_contract_id", ""),
        "hls_cycles": int(hls.get("cycles") or 0),
        "hls_latency_ms": _float_or_empty(hls.get("latency_ms")),
        "hls_ii": int(hls.get("ii") or 0),
        "hls_depth": int(hls.get("depth") or 0),
        "hls_fmax_mhz": _float_or_empty(hls.get("fmax_est_mhz")),
        "hls_lut": int(hls.get("lut") or 0),
        "hls_ff": int(hls.get("ff") or 0),
        "hls_bram18": int(hls.get("bram_18k") or hls.get("bram") or 0),
        "hls_dsp": int(hls.get("dsp") or 0),
        "vivado_status": vivado_actual.get("status", "missing"),
        "vivado_post_route_fmax_mhz": _float_or_empty(vivado_metrics.get("fmax_est_mhz")),
        "vivado_post_route_wns_ns": _float_or_empty(vivado_metrics.get("setup_wns_ns")),
        "vivado_post_route_lut": int(vivado_metrics.get("lut") or 0),
        "vivado_post_route_ff": int(vivado_metrics.get("ff") or 0),
        "vivado_post_route_bram_tile": _float_or_empty(vivado_metrics.get("block_ram_tile")),
        "vivado_post_route_ramb18": int(vivado_metrics.get("ramb18") or 0),
        "vivado_post_route_dsp": int(vivado_metrics.get("dsp") or 0),
        "board_measured": "yes" if board_measurement else "no",
        "board_cycles": int(board_measurement.get("cycles") or 0),
        "board_latency_ms": _float_or_empty(board_measurement.get("latency_ms")),
        "board_checksum": int(board_measurement.get("checksum") or 0),
        "board_word_count": int(board_measurement.get("word_count") or 0),
        "board_timing_wns_ns": _float_or_empty(board_timing.get("setup_wns_ns")),
        "board_timing_met_200mhz": _yes_no(board_timing.get("setup_wns_ns") is not None and float(board_timing["setup_wns_ns"]) >= 0.0)
        if board_timing.get("setup_wns_ns") is not None else "",
        "board_lut": int(board_util.get("lut") or 0),
        "board_ff": int(board_util.get("ff") or 0),
        "board_bram_tile": _float_or_empty(board_util.get("block_ram_tile")),
        "board_dsp": int(board_util.get("dsp") or 0),
        "search_enabled": _yes_no(search_enabled),
        "search_policy_default_enabled": _yes_no(search_default),
        "policy_deployment_status": policy_deployment_status,
        "policy_screening_status": policy_screening_status,
        "policy_reason": policy_reason,
        "deployability_label": deployability_label,
        "deployability_reason": deployability_reason,
    }
    return row


def _build_manifest_entry(entry: dict[str, Any], board_actual: dict[str, Any] | None, row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    payload["board_actual"] = board_actual
    payload["deployment"] = {
        "label": row["deployability_label"],
        "reason": row["deployability_reason"],
        "policy_deployment_status": row["policy_deployment_status"],
        "policy_screening_status": row["policy_screening_status"],
        "search_enabled": row["search_enabled"] == "yes",
    }
    payload.setdefault("metadata", {})
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["board_measured"] = row["board_measured"] == "yes"
    payload["metadata"]["three_tier_selected_source"] = "board_actual" if row["board_measured"] == "yes" else payload["metadata"].get(
        "selected_metrics_source", "vivado_actual.post_route"
    )
    return payload


def _classify_case(
    *,
    policy: dict[str, Any],
    vivado_metrics: dict[str, Any],
    board_timing: dict[str, Any],
    board_measurement: dict[str, Any],
) -> tuple[str, str]:
    deployment_status = str(policy.get("deployment_status", "ready"))
    screening_status = str(policy.get("screening_status", "unknown"))

    if deployment_status == "dropped_current_target":
        return "dropped_current_target", screening_status or "policy dropped for current target"
    if deployment_status == "paused":
        return "paused", screening_status or "paused by extension screening policy"

    dsp = int(vivado_metrics.get("dsp") or 0)
    bram18 = int(vivado_metrics.get("ramb18") or 0)
    fmax = _float_or_none(vivado_metrics.get("fmax_est_mhz"))
    board_wns = _float_or_none(board_timing.get("setup_wns_ns"))

    if dsp > 840:
        return "non_deployable_resource", f"DSP over budget ({dsp} > 840)"
    if bram18 > 445:
        return "non_deployable_resource", f"BRAM18 over budget ({bram18} > 445)"
    if fmax is not None and fmax < 150.0:
        return "non_deployable_timing", f"post-route Fmax below 150 MHz ({fmax:.2f} MHz)"

    if board_measurement:
        if board_wns is not None and board_wns < 0.0:
            return "board_measured_timing_violation", f"board harness timing WNS is negative ({board_wns:.3f} ns)"
        if deployment_status == "conditional":
            return "conditional_research_only", "board measured, but policy marks this operator as research-only"
        return "deployable", "board measured and board-harness timing meets 200 MHz"

    if fmax is None:
        return "pending_characterization", "missing Vivado post-route timing data"
    if fmax < 200.0:
        if deployment_status == "conditional":
            return "conditional_research_only", f"post-route Fmax is {fmax:.2f} MHz and policy marks it research-only"
        return "timing_marginal", f"post-route Fmax is below 200 MHz ({fmax:.2f} MHz)"
    if deployment_status == "conditional":
        return "conditional_research_only", "timing meets target, but policy still marks it research-only"
    return "deployable_candidate", "Vivado post-route timing/resources meet current screening thresholds"


def _load_entries(paths: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = _load_yaml(path)
        for entry in payload.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata", {}) or {}
            case_name = str(metadata.get("case_name") or "")
            if not case_name:
                continue
            merged[case_name] = entry
    return merged


def _load_board_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    latest_by_case: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_name = str(row.get("case_name") or "").strip()
            if not case_name:
                continue
            existing = latest_by_case.get(case_name)
            if existing is None or str(row.get("timestamp") or "") > str(existing.get("timestamp") or ""):
                latest_by_case[case_name] = row

    results: dict[str, dict[str, Any]] = {}
    for case_name, row in latest_by_case.items():
        manifest_path = Path(str(row.get("manifest") or "")).expanduser()
        project_root = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            project_root = Path(manifest["generated"]["project_root"])
        elif manifest_path.parent.exists():
            project_root = manifest_path.parent

        timing_metrics: dict[str, Any] = {}
        util_metrics: dict[str, Any] = {}
        if project_root is not None:
            timing_report = project_root / "reports" / "timing_summary.rpt"
            util_report = project_root / "reports" / "utilization.rpt"
            if timing_report.exists():
                timing_metrics = parse_vivado_timing_summary_text(timing_report.read_text(encoding="utf-8", errors="ignore"))
            if util_report.exists():
                util_metrics = parse_vivado_utilization_text(util_report.read_text(encoding="utf-8", errors="ignore"))

        results[case_name] = {
            "measurement": {
                "timestamp": row.get("timestamp"),
                "module_name": row.get("module_name"),
                "serial_port": row.get("serial_port"),
                "status_code": int(row.get("status_code") or 0),
                "status_label": row.get("status_label"),
                "cycles": int(row.get("cycles") or 0),
                "latency_ns": _float_or_none(row.get("latency_ns")),
                "latency_ms": _float_or_none(row.get("latency_ms")),
                "checksum": int(row.get("checksum") or 0),
                "word_count": int(row.get("word_count") or 0),
                "clock_freq_hz": int(row.get("clock_freq_hz") or 0),
                "bitstream": row.get("bitstream"),
                "board_config": row.get("board_config"),
                "manifest": row.get("manifest"),
            },
            "timing": timing_metrics,
            "utilization": util_metrics,
        }
    return results


def _write_extension_doc(path: Path, rows: list[dict[str, Any]], operator_status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension_rows = [row for row in rows if row["operator_group"] == "extension"]
    by_op = {row["op_type"]: row for row in extension_rows}

    lines = [
        "# Extension Operator Disposition",
        "",
        "This document records the current keep/pause/drop decision for extension operators under the AV7K325 target and the `packed_stream_v1` contract.",
        "",
        "## Decision Summary",
        "",
        "| Operator | Search Enabled | Deployment Status | Screening Status | Post-route Fmax (MHz) | LUT | DSP | Decision | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for op_name in sorted(operator_status):
        row = by_op.get(op_name)
        policy = operator_status.get(op_name, {})
        if row is None:
            decision = _decision_from_policy(policy)
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{op_name}`",
                        _yes_no(_coerce_bool(policy.get("search_enabled"), default=True)),
                        str(policy.get("deployment_status", "unknown")),
                        str(policy.get("screening_status", "unknown")),
                        "-",
                        "-",
                        "-",
                        decision,
                        str(policy.get("reason", "No usable HLS/Vivado summary row was produced for this operator.")),
                    ]
                )
                + " |"
            )
            continue

        decision = _decision_from_label(row["deployability_label"])
        notes = policy.get("reason", row["deployability_reason"])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{op_name}`",
                    row["search_enabled"],
                    row["policy_deployment_status"],
                    row["policy_screening_status"],
                    _fmt_number(row["vivado_post_route_fmax_mhz"]),
                    str(row["vivado_post_route_lut"]),
                    str(row["vivado_post_route_dsp"]),
                    decision,
                    str(notes),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `continue`: keep the operator in the research path or conditional search space.",
            "- `pause`: remove it from the default mainline search space, but keep the implementation for possible revisit.",
            "- `drop`: stop investing on the current target/contract unless a dedicated redesign is planned.",
            "",
            "## Source of Truth",
            "",
            "- Policy declarations come from `hls_lut_builder/configs/operator_manifest.yaml`.",
            "- Hardware evidence comes from the merged HLS + Vivado post-route summary generated by `summarize_three_tier_results.py`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_from_label(label: str) -> str:
    if label in {"deployable", "deployable_candidate", "conditional_research_only", "timing_marginal"}:
        return "continue"
    if label == "paused":
        return "pause"
    if label in {"dropped_current_target", "non_deployable_resource", "non_deployable_timing"}:
        return "drop"
    return "review"


def _decision_from_policy(policy: dict[str, Any]) -> str:
    deployment_status = str(policy.get("deployment_status", "unknown"))
    if deployment_status == "conditional":
        return "continue"
    if deployment_status == "paused":
        return "pause"
    if deployment_status == "dropped_current_target":
        return "drop"
    return "review"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def _float_or_empty(value: Any) -> str:
    if value in (None, "", "null"):
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _fmt_number(value: Any) -> str:
    if value in (None, "", "null"):
        return "-"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
