#!/usr/bin/env python3
"""Independent arch_84 e2e extension board evidence driver.

This script owns only the v2_arch84_e2e_extension namespace. It does not update
current84 deployable LUTs, current84 primitive60/fullcombo84 ledgers, or the
legacy board_measured_lut.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BUILDER_ROOT = ROOT / "hls_lut_builder"
RESULTS_DIR = BUILDER_ROOT / "results"
BOARD_HARNESS_DIR = BUILDER_ROOT / "board_harness"
CONFIG_PATH = BUILDER_ROOT / "configs" / "candidate_kernels_v2_arch84_e2e_extension.yaml"
RESULT_DIR = RESULTS_DIR / "v2_arch84_e2e_extension"
BOARD_OUT_DIR = BOARD_HARNESS_DIR / "results" / "v2_arch84_e2e_extension_board"
CASE_RESULTS_DIR = BOARD_OUT_DIR / "case_results"
TIMING_RECOVERY_DIR = BOARD_OUT_DIR / "timing_recovery"
PARAM_MANIFEST = RESULT_DIR / "arch84_latency_only_parameter_manifest.json"

sys.path.insert(0, str(BUILDER_ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from common import build_cases, load_config  # noqa: E402
from batch_measure_ops import DEFAULT_BOARD_CONFIG, DEFAULT_VITIS_HLS, DEFAULT_VIVADO  # noqa: E402
import current84_fullcombo_board_validation as fullcombo  # noqa: E402


PRIMITIVE_SHAPES = {
    "pw_expand": "a84_pwe",
    "dw": "a84_dw",
    "pw_project": "a84_pwp",
    "gap": "a84_gap",
    "fc": "a84_fc",
}

COMBO_DEFS = [
    {
        "combo_case_name": "mbconv_e3_k3_arch84_stage3_28_32_32_s1_latency_only_main_5ns",
        "decomposition": "pw_expand+dw+pw_project",
        "harness_kind": "stitched_mbconv_harness",
        "component_roles": "pw_expand|dw|pw_project",
        "shape_roles": "pw_expand|dw|pw_project",
        "notes": "arch_84 extension MBConv; must be measured with stitched harness, not primitive sum",
    },
    {
        "combo_case_name": "gap_28_32_arch84_latency_only_main_5ns",
        "decomposition": "direct",
        "harness_kind": "single_op_harness",
        "component_roles": "gap",
        "shape_roles": "gap",
        "notes": "arch_84 extension direct GAP combo",
    },
    {
        "combo_case_name": "fc_1_32_8_arch84_latency_only_main_5ns",
        "decomposition": "direct",
        "harness_kind": "single_op_harness",
        "component_roles": "fc",
        "shape_roles": "fc",
        "notes": "arch_84 extension direct FC combo; output word_count must be 8 for e2e classifier logits",
    },
]

PRIMITIVE_FIELDS = [
    "role",
    "component_case_name",
    "operator",
    "shape_name",
    "formal_status",
    "hls_status",
    "downstream_status",
    "wns_ns",
    "case_dir",
    "config_path",
    "notes",
]

COMBO_FIELDS = [
    "combo_case_name",
    "decomposition",
    "harness_kind",
    "component_roles",
    "component_case_names",
    "status",
    "measurement_status",
    "uart_status_code",
    "uart_status_label",
    "board_combo_cycles",
    "board_combo_latency_ms",
    "board_word_count",
    "board_checksum",
    "wns_ns",
    "timing_clean",
    "measurement_json_path",
    "bitstream_path",
    "harness_project_dir",
    "timing_report_path",
    "utilization_report_path",
    "stdout_log",
    "notes",
]

BLOCKER_FIELDS = [
    "scope",
    "case_name",
    "blocker_category",
    "severity",
    "evidence_path",
    "evidence_detail",
    "required_resolution",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="arch_84 e2e extension validation")
    parser.add_argument("mode", choices=("init", "refresh", "run-primitives", "run-combos", "recover-combos", "all"))
    parser.add_argument("--board-config", default=str(DEFAULT_BOARD_CONFIG))
    parser.add_argument("--serial-port", default="COM5")
    parser.add_argument("--vitis-hls", default=str(DEFAULT_VITIS_HLS))
    parser.add_argument("--vivado", default=str(DEFAULT_VIVADO))
    parser.add_argument("--hw-server-url", default="TCP:localhost:3121")
    parser.add_argument("--hw-device-pattern", default="*xc7k325t*")
    parser.add_argument("--hw-target-index", type=int, default=0)
    parser.add_argument("--program-retries", type=int, default=2)
    parser.add_argument("--uart-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--post-program-delay-seconds", type=float, default=1.0)
    parser.add_argument("--hls-timeout-minutes", type=int, default=60)
    parser.add_argument("--downstream-timeout-minutes", type=int, default=90)
    parser.add_argument("--harness-timeout-minutes", type=int, default=10)
    parser.add_argument("--bitstream-timeout-minutes", type=int, default=120)
    parser.add_argument("--measurement-timeout-minutes", type=int, default=20)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument(
        "--recovery-variants",
        default="performance_explore,post_route_phys_opt,impl_phys_opt,route_no_timing_post_route_phys_opt_explore",
        help="Comma-separated timing recovery variants for arch84 stitched MBConv combos.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-measure", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cases_by_shape() -> dict[str, Any]:
    config = load_config(str(CONFIG_PATH))
    return {case.shape_name: case for case in build_cases(config)}


def case_by_role() -> dict[str, Any]:
    by_shape = cases_by_shape()
    return {role: by_shape[shape] for role, shape in PRIMITIVE_SHAPES.items()}


def component_names_for_combo(combo: dict[str, str], roles: dict[str, Any]) -> list[str]:
    return [roles[role].case_name for role in combo["shape_roles"].split("|")]


def latency_only_parameter_manifest() -> dict[str, Any]:
    seed = 840325
    topology = [
        "stem_224_1_32_s2",
        "pw_112_32_16",
        "mbconv_e3_k3_112_16_24_s2",
        "mbconv_e6_k3_56_24_32_s2",
        "mbconv_e3_k3_28_32_32_s1",
        "gap_28_32",
        "fc_1_32_8",
    ]
    payload = {
        "parameter_mode": "latency_only_deterministic",
        "seed": seed,
        "generator": "xorshift32 signed int8 values; classification output is channel/checksum sanity only",
        "topology": topology,
        "output_word_count": 8,
        "accuracy_claim": "none",
        "notes": [
            "This manifest is for board latency and channel integrity only.",
            "It is not a trained sonar classifier checkpoint and must not be used for accuracy reporting.",
        ],
    }
    payload["manifest_hash_sha256"] = sha256_text(json.dumps(payload, sort_keys=True))
    return payload


def hls_status_for_case(case: Any) -> tuple[str, str, str]:
    status = read_json(case.project_dir / "synthesis_status.json")
    downstream = read_json(case.vivado_downstream_status_path)
    hls_status = str(status.get("status") or "not_run")
    downstream_status = str(downstream.get("status") or "not_run")
    wns = ""
    metrics = downstream.get("metrics")
    if isinstance(metrics, dict):
        post_route = metrics.get("post_route")
        if isinstance(post_route, dict):
            wns = str(post_route.get("wns_ns") or post_route.get("wns") or "")
    formal_closed = hls_status in {"success", "skipped_existing"} and downstream_status in {"success", "skipped_existing"}
    return ("formal_closed" if formal_closed else "pending_formal_closure", hls_status, downstream_status), wns


def build_primitive_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, case in case_by_role().items():
        (formal_status, hls_status, downstream_status), wns = hls_status_for_case(case)
        rows.append(
            {
                "role": role,
                "component_case_name": case.case_name,
                "operator": case.operator,
                "shape_name": case.shape_name,
                "formal_status": formal_status,
                "hls_status": hls_status,
                "downstream_status": downstream_status,
                "wns_ns": wns,
                "case_dir": str(case.project_dir),
                "config_path": str(CONFIG_PATH),
                "notes": "arch_84 extension primitive; not a current84 deployable LUT row",
            }
        )
    return rows


def combo_runlist_rows() -> list[dict[str, Any]]:
    roles = case_by_role()
    rows: list[dict[str, Any]] = []
    for combo in COMBO_DEFS:
        component_names = component_names_for_combo(combo, roles)
        rows.append(
            {
                **combo,
                "component_case_names": "|".join(component_names),
                "selected_existing_case_names": "|".join(component_names),
                "status": "pending",
                "timing_clean": "False",
            }
        )
    return rows


def result_path(combo_case_name: str) -> Path:
    return CASE_RESULTS_DIR / f"{combo_case_name}.result.json"


def write_result(combo_case_name: str, payload: dict[str, Any]) -> None:
    write_json(result_path(combo_case_name), payload)


def load_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not CASE_RESULTS_DIR.exists():
        return results
    for path in CASE_RESULTS_DIR.glob("*.result.json"):
        payload = read_json(path)
        combo = str(payload.get("combo_case_name") or path.stem.replace(".result", ""))
        results[combo] = payload
    return results


def run_command(command: list[str], cwd: Path, timeout_seconds: int | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


def run_primitives(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(BUILDER_ROOT / "scripts" / "run_synthesis.py"),
        "--config",
        str(CONFIG_PATH),
        "--downstream-check",
        "--vitis-hls",
        str(Path(args.vitis_hls).expanduser().resolve()),
        "--vivado",
        str(Path(args.vivado).expanduser().resolve()),
        "--timeout-minutes",
        str(args.hls_timeout_minutes),
        "--downstream-timeout-minutes",
        str(args.downstream_timeout_minutes),
    ]
    if args.force:
        command.append("--force")
    for case in case_by_role().values():
        command.extend(["--case", case.case_name])
    log_dir = RESULT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(command, ROOT, None)
    (log_dir / "run_primitives.out.log").write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        write_json(
            RESULT_DIR / "run_primitives_failure.json",
            {
                "status": "failed",
                "returncode": result.returncode,
                "command": command,
                "stdout_log": str(log_dir / "run_primitives.out.log"),
                "updated_at": timestamp(),
            },
        )


def _combo_output_dir(row: dict[str, Any]) -> Path:
    subdir = "mbconv_stitched" if row["decomposition"] == "pw_expand+dw+pw_project" else "direct"
    return BOARD_OUT_DIR / "combo_measurement" / subdir / row["combo_case_name"]


def _measure_combo(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    combo = row["combo_case_name"]
    output_dir = _combo_output_dir(row)
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_projects_dir = output_dir / "harness_projects"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    board_config = Path(args.board_config).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row["decomposition"],
        "harness_kind": row["harness_kind"],
        "component_case_names": row["component_case_names"],
        "started_at": timestamp(),
    }

    component_cases = row["component_case_names"].split("|")
    try:
        if row["decomposition"] == "direct":
            component_case = component_cases[0]
            case_dir = fullcombo.resolve_case_dir(component_case)
            harness = fullcombo.run_generate_harness(
                case_dir,
                board_config,
                harness_projects_dir,
                force=args.force,
                timeout_seconds=args.harness_timeout_minutes * 60 if args.harness_timeout_minutes > 0 else None,
            )
            (logs_dir / "harness.out.log").write_text(str(harness.get("stdout", "")), encoding="utf-8", errors="replace")
            if harness.get("status") not in {"success", "skipped_existing"}:
                raise RuntimeError(str(harness.get("stdout", ""))[:2000])
            harness_project = harness_projects_dir / f"harness_{component_case}"
        else:
            harness_project = harness_projects_dir / f"harness_{combo}"
            fullcombo.generate_stitched_harness(row, board_config, harness_project)
            (logs_dir / "harness.out.log").write_text("stitched harness generated\n", encoding="utf-8")

        bitstream = fullcombo.build_bitstream(
            harness_project,
            vivado,
            force=args.force,
            timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
        )
        (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
        bitstream_path = harness_project / "bitstream" / f"{combo if row['decomposition'] != 'direct' else component_cases[0]}.bit"
        if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
            raise RuntimeError(str(bitstream.get("stdout", ""))[:2000])

        json_out = measurements_dir / f"{combo}.measurement.json"
        measurement_csv = output_dir / "measurements_raw.csv"
        measurement = fullcombo.measure_harness(
            manifest=harness_project / "harness_manifest.json",
            bitstream=bitstream_path,
            csv_path=measurement_csv,
            json_out=json_out,
            serial_port=args.serial_port,
            vivado=vivado,
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
        (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")
        measurement_payload = read_json(json_out)
        wns = fullcombo.parse_wns(harness_project / "reports" / "timing_summary.rpt")
        status, board_execution, failure_category, failure_reason, timing_clean = fullcombo.classify_measurement(measurement_payload, wns)
        measurement_payload.update(
            {
                "combo_case_name": combo,
                "arch84_extension_latency_claim": "real_board_latency_if_status_code_0_and_timing_clean",
                "parameter_mode": "latency_only_deterministic",
                "board_timing_wns_ns": wns,
                "board_timing_clean": timing_clean,
                "updated_at": timestamp(),
            }
        )
        write_json(json_out, measurement_payload)
        util = fullcombo.parse_utilization(harness_project / "reports" / "utilization.rpt")
        result_payload.update(
            {
                "completed_at": timestamp(),
                "status": status,
                "board_execution_status": board_execution,
                "failure_category": failure_category,
                "failure_reason": failure_reason,
                "measurement_status": measurement.get("status", ""),
                "uart_status_code": fullcombo.frame_value(measurement_payload, "status_code"),
                "uart_status_label": measurement_payload.get("status_label", ""),
                "board_combo_cycles": fullcombo.frame_value(measurement_payload, "cycles"),
                "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
                "board_word_count": fullcombo.frame_value(measurement_payload, "word_count"),
                "board_checksum": fullcombo.frame_value(measurement_payload, "checksum"),
                "wns_ns": wns,
                "timing_clean": timing_clean,
                "measurement_json_path": str(json_out),
                "bitstream_path": str(bitstream_path),
                "harness_project_dir": str(harness_project),
                "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
                "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
                "stdout_log": str(logs_dir / "measurement.out.log"),
            }
        )
        result_payload.update(util)
    except Exception as exc:
        result_payload.update(
            {
                "completed_at": timestamp(),
                "status": "harness_or_measurement_blocked",
                "measurement_status": "failed",
                "failure_category": "extension_combo_flow_failed",
                "failure_reason": str(exc)[:2000],
                "timing_clean": False,
                "stdout_log": str(logs_dir / "harness.out.log"),
            }
        )
    write_result(combo, result_payload)
    return result_payload


def run_combos(args: argparse.Namespace) -> None:
    rows = combo_runlist_rows()
    if args.combo:
        requested = set(args.combo)
        rows = [row for row in rows if row["combo_case_name"] in requested]
    for row in rows:
        _measure_combo(row, args)


def _recovery_result_path(combo_case_name: str, variant: str) -> Path:
    return TIMING_RECOVERY_DIR / combo_case_name / variant / "recovery_result.json"


def _measure_mbconv_recovery(row: dict[str, Any], args: argparse.Namespace, variant: str) -> dict[str, Any]:
    combo = row["combo_case_name"]
    output_dir = TIMING_RECOVERY_DIR / combo / variant
    logs_dir = output_dir / "logs"
    measurements_dir = output_dir / "measurements"
    harness_project = output_dir / "harness_project"
    logs_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)

    board_config = Path(args.board_config).expanduser().resolve()
    vivado = Path(args.vivado).expanduser().resolve()
    fifo_style = fullcombo.fifo_style_for_recovery_variant(variant)
    param_reset_style = fullcombo.param_reset_style_for_recovery_variant(variant)
    result_payload: dict[str, Any] = {
        "combo_case_name": combo,
        "decomposition": row["decomposition"],
        "harness_kind": row["harness_kind"],
        "component_case_names": row["component_case_names"],
        "timing_recovery_variant": variant,
        "fifo_style": fifo_style,
        "param_reset_style": param_reset_style,
        "started_at": timestamp(),
        "strict_latency_usable": False,
    }

    try:
        fullcombo.generate_stitched_harness(
            row,
            board_config,
            harness_project,
            recovery_variant=variant,
            fifo_style=fifo_style,
            param_reset_style=param_reset_style,
        )
        (logs_dir / "harness.out.log").write_text(
            f"stitched harness generated with recovery variant {variant}\n",
            encoding="utf-8",
        )

        bitstream = fullcombo.build_bitstream(
            harness_project,
            vivado,
            force=True,
            timeout_seconds=args.bitstream_timeout_minutes * 60 if args.bitstream_timeout_minutes > 0 else None,
        )
        (logs_dir / "vivado_bitstream.out.log").write_text(str(bitstream.get("stdout", "")), encoding="utf-8", errors="replace")
        bitstream_path = harness_project / "bitstream" / f"{combo}.bit"
        if bitstream.get("status") not in {"success", "skipped_existing"} or not bitstream_path.exists():
            raise RuntimeError(str(bitstream.get("stdout", ""))[:2000])

        json_out = measurements_dir / f"{combo}.recovery_{variant}.measurement.json"
        measurement_csv = output_dir / "measurements_raw.csv"
        measurement = fullcombo.measure_harness(
            manifest=harness_project / "harness_manifest.json",
            bitstream=bitstream_path,
            csv_path=measurement_csv,
            json_out=json_out,
            serial_port=args.serial_port,
            vivado=vivado,
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
        (logs_dir / "measurement.out.log").write_text(str(measurement.get("stdout", "")), encoding="utf-8", errors="replace")

        measurement_payload = read_json(json_out)
        wns = fullcombo.parse_wns(harness_project / "reports" / "timing_summary.rpt")
        status, board_execution, failure_category, failure_reason, timing_clean = fullcombo.classify_measurement(measurement_payload, wns)
        measurement_payload.update(
            {
                "combo_case_name": combo,
                "arch84_extension_latency_claim": "real_board_latency_if_status_code_0_and_timing_clean",
                "parameter_mode": "latency_only_deterministic",
                "timing_recovery_variant": variant,
                "fifo_style": fifo_style,
                "param_reset_style": param_reset_style,
                "board_timing_wns_ns": wns,
                "board_timing_clean": timing_clean,
                "updated_at": timestamp(),
            }
        )
        write_json(json_out, measurement_payload)
        util = fullcombo.parse_utilization(harness_project / "reports" / "utilization.rpt")
        result_payload.update(
            {
                "completed_at": timestamp(),
                "status": status,
                "board_execution_status": board_execution,
                "failure_category": failure_category,
                "failure_reason": failure_reason,
                "measurement_status": measurement.get("status", ""),
                "uart_status_code": fullcombo.frame_value(measurement_payload, "status_code"),
                "uart_status_label": measurement_payload.get("status_label", ""),
                "board_combo_cycles": fullcombo.frame_value(measurement_payload, "cycles"),
                "board_combo_latency_ms": measurement_payload.get("latency_ms", ""),
                "board_word_count": fullcombo.frame_value(measurement_payload, "word_count"),
                "board_checksum": fullcombo.frame_value(measurement_payload, "checksum"),
                "wns_ns": wns,
                "timing_clean": timing_clean,
                "measurement_json_path": str(json_out),
                "measurement_csv_path": str(measurement_csv),
                "bitstream_path": str(bitstream_path),
                "harness_project_dir": str(harness_project),
                "timing_report_path": str(harness_project / "reports" / "timing_summary.rpt"),
                "utilization_report_path": str(harness_project / "reports" / "utilization.rpt"),
                "stdout_log": str(logs_dir / "measurement.out.log"),
                "strict_latency_usable": timing_clean,
            }
        )
        result_payload.update(util)
    except Exception as exc:
        result_payload.update(
            {
                "completed_at": timestamp(),
                "status": "timing_recovery_failed",
                "measurement_status": "failed",
                "failure_category": "extension_timing_recovery_failed",
                "failure_reason": str(exc)[:2000],
                "timing_clean": False,
                "stdout_log": str(logs_dir / "vivado_bitstream.out.log"),
            }
        )

    write_json(_recovery_result_path(combo, variant), result_payload)
    existing = read_json(result_path(combo))
    existing_wns = fullcombo.parse_float(existing.get("wns_ns"))
    new_wns = fullcombo.parse_float(result_payload.get("wns_ns"))
    improved_fail = (
        result_payload.get("status") == "board_measured_uart_ok_timing_fail"
        and existing.get("status") != "board_measured_timing_clean"
        and new_wns is not None
        and (existing_wns is None or new_wns > existing_wns)
    )
    if result_payload.get("status") == "board_measured_timing_clean" or improved_fail:
        write_result(combo, result_payload)
    return result_payload


def recover_combos(args: argparse.Namespace) -> None:
    rows = [row for row in combo_runlist_rows() if row["decomposition"] == "pw_expand+dw+pw_project"]
    if args.combo:
        requested = set(args.combo)
        rows = [row for row in rows if row["combo_case_name"] in requested]
    variants = fullcombo.parse_recovery_variants(args.recovery_variants)
    outputs: list[dict[str, Any]] = []
    for row in rows:
        existing = read_json(result_path(row["combo_case_name"]))
        if existing.get("status") == "board_measured_timing_clean" and not args.force:
            outputs.append(existing)
            continue
        for variant in variants:
            result = _measure_mbconv_recovery(row, args, variant)
            outputs.append(result)
            build_outputs("incremental-recover-combos")
            if result.get("status") == "board_measured_timing_clean":
                break
    summary = {
        "generated_at": timestamp(),
        "output_dir": str(TIMING_RECOVERY_DIR),
        "variants": variants,
        "result_count": len(outputs),
        "counts": dict(Counter(str(result.get("status", "")) for result in outputs)),
        "results": outputs,
    }
    write_json(TIMING_RECOVERY_DIR / "arch84_timing_recovery_summary.json", summary)


def build_outputs(mode: str) -> dict[str, Any]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PARAM_MANIFEST, latency_only_parameter_manifest())
    primitive_rows = build_primitive_rows()
    runlist_rows = combo_runlist_rows()
    results = load_results()
    combo_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []

    for row in runlist_rows:
        result = results.get(row["combo_case_name"], {})
        out = {**row, **result}
        if not result:
            out["status"] = "pending_board_measurement"
            out["timing_clean"] = "False"
        combo_rows.append(out)

    for row in primitive_rows:
        if row["formal_status"] != "formal_closed":
            blocker_rows.append(
                {
                    "scope": "primitive",
                    "case_name": row["component_case_name"],
                    "blocker_category": "primitive_formal_not_closed",
                    "severity": "blocking",
                    "evidence_path": row["case_dir"],
                    "evidence_detail": f"hls={row['hls_status']} downstream={row['downstream_status']}",
                    "required_resolution": "run arch84 extension HLS and Vivado downstream formal closure",
                }
            )
    for row in combo_rows:
        if row.get("status") not in {"board_measured_timing_clean", "board_measured_uart_ok_timing_fail"}:
            blocker_rows.append(
                {
                    "scope": "combo",
                    "case_name": row["combo_case_name"],
                    "blocker_category": "combo_board_not_measured",
                    "severity": "blocking",
                    "evidence_path": row.get("measurement_json_path", ""),
                    "evidence_detail": str(row.get("failure_reason") or row.get("status") or "pending"),
                    "required_resolution": "run real AV7K325 COM5 combo measurement",
                }
            )
        elif row.get("status") == "board_measured_uart_ok_timing_fail":
            blocker_rows.append(
                {
                    "scope": "combo",
                    "case_name": row["combo_case_name"],
                    "blocker_category": "combo_board_timing_not_clean",
                    "severity": "strict_latency_blocking",
                    "evidence_path": row.get("timing_report_path", ""),
                    "evidence_detail": f"UART status_code={row.get('uart_status_code')} but WNS={row.get('wns_ns')}",
                    "required_resolution": "run timing recovery until UART status_code=0 and WNS>=0, or keep as non-strict evidence",
                }
            )

    counts = Counter(str(row.get("status", "")) for row in combo_rows)
    summary = {
        "generated_at": timestamp(),
        "mode": mode,
        "namespace": "v2_arch84_e2e_extension",
        "config_path": str(CONFIG_PATH),
        "parameter_manifest": str(PARAM_MANIFEST),
        "primitive_total": len(primitive_rows),
        "primitive_formal_closed": sum(1 for row in primitive_rows if row["formal_status"] == "formal_closed"),
        "combo_total": len(combo_rows),
        "combo_counts": dict(counts),
        "combo_uart_ok": sum(1 for row in combo_rows if str(row.get("uart_status_code")) in {"0", "0.0"}),
        "combo_timing_clean": sum(1 for row in combo_rows if str(row.get("status")) == "board_measured_timing_clean"),
        "blocker_count": len(blocker_rows),
        "strict_latency_usable": len(combo_rows) == 3 and all(str(row.get("status")) == "board_measured_timing_clean" for row in combo_rows),
        "outputs": {
            "primitive_status_csv": str(RESULT_DIR / "arch84_e2e_extension_primitive_status.csv"),
            "combo_runlist_csv": str(BOARD_OUT_DIR / "arch84_e2e_extension_combo_runlist.csv"),
            "combo_status_csv": str(BOARD_OUT_DIR / "arch84_e2e_extension_combo_status.csv"),
            "blockers_csv": str(BOARD_OUT_DIR / "arch84_e2e_extension_blockers.csv"),
            "summary_json": str(BOARD_OUT_DIR / "arch84_e2e_extension_summary.json"),
            "readme": str(BOARD_OUT_DIR / "README.md"),
        },
    }

    readme = f"""# arch_84 E2E Extension Board Evidence

This directory contains independent arch_84 extension evidence. It is not part of
current84 primitive60, not part of current84 fullcombo84, and does not update any
deployable LUT JSON or legacy board measured CSV.

- Primitive extension rows: 5
- Combo extension rows: 3
- Parameter mode: latency-only deterministic; classification/channel output is
  only a checksum/argmax sanity signal and is not a trained sonar accuracy claim.
- Timing-clean is strict: UART status_code=0 and WNS>=0.
- MBConv extension latency is valid only from the stitched harness COM5
  measurement, never from primitive latency sums.

Current strict latency usable: `{summary['strict_latency_usable']}`.
See `arch84_e2e_extension_blockers.csv` for unresolved blockers.
"""

    write_csv(RESULT_DIR / "arch84_e2e_extension_primitive_status.csv", primitive_rows, PRIMITIVE_FIELDS)
    write_csv(BOARD_OUT_DIR / "arch84_e2e_extension_combo_runlist.csv", runlist_rows, COMBO_FIELDS)
    write_csv(BOARD_OUT_DIR / "arch84_e2e_extension_combo_status.csv", combo_rows, COMBO_FIELDS)
    write_csv(BOARD_OUT_DIR / "arch84_e2e_extension_blockers.csv", blocker_rows, BLOCKER_FIELDS)
    write_json(BOARD_OUT_DIR / "arch84_e2e_extension_summary.json", summary)
    (BOARD_OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.mode in {"run-primitives", "all"}:
        run_primitives(args)
    if args.mode in {"run-combos", "all"}:
        run_combos(args)
    if args.mode == "recover-combos":
        recover_combos(args)
    summary = build_outputs(args.mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
