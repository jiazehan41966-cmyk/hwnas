#!/usr/bin/env python3
"""Audit acceptance gates for Phase0 v3 retrained-weight board reinjection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_ARCHES = {"rl_arch_186", "rl_arch_242"}


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gate(name: str, passed: bool, observed: Any, expected: str) -> dict[str, Any]:
    return {"gate": name, "pass": bool(passed), "observed": observed, "expected": expected}


def unique_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values})


def row_for_arch(rows: list[dict[str, Any]], arch_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("arch_id") == arch_id:
            return row
    return {}


def audit_run(result_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_name = str(row.get("run_name", ""))
    run_root = result_root / run_name
    build = read_json(run_root / "build_result.json")
    preflight = read_json(run_root / "measurement_preflight.json")
    stability = read_json(run_root / "measurements" / "stability_runs.json")
    trained = read_json(run_root / "harness_project" / "trained_weight_manifest.json")

    bitstream = Path(str(row.get("bitstream") or build.get("bitstream") or ""))
    bitstream_hash = sha256_file(bitstream)
    readiness = preflight.get("readiness", {}) if isinstance(preflight.get("readiness"), dict) else {}

    wns = as_float(row.get("wns_ns"))
    tns = as_float(row.get("tns_ns"))
    dsp = as_int(row.get("dsp"))
    status_code = as_int(row.get("com5_status_code"))
    run_count = as_int(row.get("com5_run_count"))
    latency = as_float(row.get("real_e2e_latency_ms"))
    cycles = as_int(row.get("cycles"))

    status_codes = unique_values(stability.get("status_codes"))
    cycle_values = unique_values(stability.get("cycles"))
    checksum_values = unique_values(stability.get("checksums"))

    exports = trained.get("exports", []) if isinstance(trained.get("exports"), list) else []
    gates = [
        gate("expected_arch", row.get("arch_id") in EXPECTED_ARCHES, row.get("arch_id"), "one of rl_arch_186, rl_arch_242"),
        gate("parameter_mode", row.get("parameter_mode") == "trained_checkpoint_mem", row.get("parameter_mode"), "trained_checkpoint_mem"),
        gate("accuracy_claim_boundary", trained.get("accuracy_claim") == "none", trained.get("accuracy_claim"), "none"),
        gate("checkpoint_sha256_present", bool(row.get("checkpoint_sha256")), row.get("checkpoint_sha256"), "non-empty"),
        gate("trained_export_count", len(exports) == 12, len(exports), "12 exported roles"),
        gate("build_success", build.get("status") == "success", build.get("status"), "success"),
        gate("timing_clean", build.get("timing_clean") is True, build.get("timing_clean"), "true"),
        gate("wns_nonnegative", wns is not None and wns >= 0.0, row.get("wns_ns"), ">= 0.0 ns"),
        gate("tns_zero_or_nonpositive_fail_free", tns is not None and tns >= 0.0, row.get("tns_ns"), ">= 0.0 ns"),
        gate("dsp_limit", dsp is not None and dsp <= 700, row.get("dsp"), "<= 700"),
        gate("bitstream_exists", bitstream.exists(), str(bitstream), "exists"),
        gate("bitstream_sha256_matches_report", bitstream_hash == str(row.get("bitstream_sha256")), bitstream_hash, str(row.get("bitstream_sha256"))),
        gate("preflight_sha_matches_report", readiness.get("bitstream_sha256") == row.get("bitstream_sha256"), readiness.get("bitstream_sha256"), str(row.get("bitstream_sha256"))),
        gate("com5_status_code", status_code == 0, row.get("com5_status_code"), "0"),
        gate("com5_run_count", run_count is not None and run_count >= 5, row.get("com5_run_count"), ">= 5"),
        gate("com5_stable_flag", row.get("com5_stable") is True, row.get("com5_stable"), "true"),
        gate("stability_status_codes", status_codes == ["0"], status_codes, "['0']"),
        gate("stability_cycles_single_value", len(cycle_values) == 1 and str(cycles) in cycle_values, cycle_values, "single value matching report cycles"),
        gate("stability_checksums_single_value", len(checksum_values) == 1 and str(row.get("board_checksum")) in checksum_values, checksum_values, "single value matching report checksum"),
        gate("latency_present", latency is not None and latency > 0.0, row.get("real_e2e_latency_ms"), "> 0 ms"),
    ]
    return {
        "run_name": run_name,
        "arch_id": row.get("arch_id", ""),
        "positioning": row.get("positioning", ""),
        "pass": all(item["pass"] for item in gates),
        "gates": gates,
        "summary": {
            "retrain_macro_f1": row.get("retrain_macro_f1", ""),
            "retrain_top1": row.get("retrain_top1", ""),
            "retrain_weighted_f1": row.get("retrain_weighted_f1", ""),
            "bitstream_sha256": row.get("bitstream_sha256", ""),
            "wns_ns": row.get("wns_ns", ""),
            "tns_ns": row.get("tns_ns", ""),
            "dsp": row.get("dsp", ""),
            "com5_status_code": row.get("com5_status_code", ""),
            "cycles": row.get("cycles", ""),
            "real_e2e_latency_ms": row.get("real_e2e_latency_ms", ""),
            "board_checksum": row.get("board_checksum", ""),
            "board_argmax": row.get("board_argmax", ""),
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v3 Retrained-Weight Board Reinjection Acceptance Audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- result_root: `{payload['result_root']}`",
        f"- overall_pass: `{payload['overall_pass']}`",
        "- boundary: COM5 gates validate deterministic harness-input board sanity; validation-set accuracy remains PyTorch retrain150.",
        "",
        "| Arch | Positioning | Overall | Failed gates | WNS ns | TNS ns | DSP | COM5 | Runs | cycles | real_e2e_ms | checksum | argmax |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        failed = [item["gate"] for item in run["gates"] if not item["pass"]]
        summary = run["summary"]
        lines.append(
            "| `{arch_id}` | {positioning} | {overall} | {failed} | {wns_ns} | {tns_ns} | {dsp} | {com5} | {runs} | {cycles} | {latency} | {checksum} | {argmax} |".format(
                arch_id=run["arch_id"],
                positioning=run["positioning"],
                overall="PASS" if run["pass"] else "FAIL",
                failed=", ".join(failed) if failed else "-",
                wns_ns=summary["wns_ns"],
                tns_ns=summary["tns_ns"],
                dsp=summary["dsp"],
                com5=summary["com5_status_code"],
                runs=next((item["observed"] for item in run["gates"] if item["gate"] == "com5_run_count"), ""),
                cycles=summary["cycles"],
                latency=summary["real_e2e_latency_ms"],
                checksum=summary["board_checksum"],
                argmax=summary["board_argmax"],
            )
        )
    lines.extend(["", "## Gate Details", ""])
    for run in payload["runs"]:
        lines.extend([f"### {run['arch_id']}", ""])
        for item in run["gates"]:
            status = "PASS" if item["pass"] else "FAIL"
            lines.append(f"- `{status}` `{item['gate']}`: observed `{item['observed']}`, expected `{item['expected']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(result_root: Path) -> dict[str, Any]:
    report_path = result_root / "reports" / "retrained_board_reinject_comparison.json"
    report = read_json(report_path)
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    audits = [audit_run(result_root, row_for_arch(rows, arch_id)) for arch_id in sorted(EXPECTED_ARCHES)]
    present_arches = {str(row.get("arch_id")) for row in rows if isinstance(row, dict)}
    root_gates = [
        gate("comparison_report_exists", report_path.exists(), str(report_path), "exists"),
        gate("expected_arches_present", EXPECTED_ARCHES.issubset(present_arches), sorted(present_arches), sorted(EXPECTED_ARCHES)),
    ]
    return {
        "generated_at": timestamp(),
        "result_root": str(result_root.resolve()),
        "overall_pass": all(item["pass"] for item in root_gates) and all(run["pass"] for run in audits),
        "root_gates": root_gates,
        "runs": audits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit retrained board reinjection acceptance gates.")
    parser.add_argument(
        "--result-root",
        default="hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    payload = audit(result_root)
    reports = result_root / "reports"
    json_path = reports / "retrained_board_reinject_acceptance_audit.json"
    md_path = reports / "retrained_board_reinject_acceptance_audit.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(json.dumps({"overall_pass": payload["overall_pass"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if payload["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
