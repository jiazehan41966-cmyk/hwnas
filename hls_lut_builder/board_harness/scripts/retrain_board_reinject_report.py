#!/usr/bin/env python3
"""Summarize Phase0 v3 retrained-weight board reinjection runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


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


def parse_design_timing_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for row in lines[idx + 1 : idx + 5]:
            values = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", row)
            if len(values) >= 2:
                return {"wns_ns": values[0], "tns_ns": values[1]}
    return {}


def frame_value(payload: dict[str, Any], key: str) -> Any:
    frame = payload.get("frame")
    if isinstance(frame, dict) and key in frame:
        return frame[key]
    return payload.get(key, "")


def checksum_argmax(checksum: Any) -> str:
    try:
        return str((int(checksum) >> 24) & 0xFF)
    except (TypeError, ValueError):
        return ""


def checksum24(checksum: Any) -> str:
    try:
        return str(int(checksum) & 0xFFFFFF)
    except (TypeError, ValueError):
        return ""


def positioning(arch_id: str) -> str:
    if arch_id == "rl_arch_186":
        return "accuracy-first"
    if arch_id == "rl_arch_242":
        return "latency-efficient"
    return "claimable-extra"


def row_for_run(run_root: Path) -> dict[str, Any]:
    project = run_root / "harness_project"
    harness = read_json(project / "harness_manifest.json")
    trained = read_json(project / "trained_weight_manifest.json")
    parity = read_json(project / "software_board_parity.json")
    build = read_json(run_root / "build_result.json")
    measurement_files = sorted((run_root / "measurements").glob("*.measurement.json"))
    measurement = read_json(measurement_files[0]) if measurement_files else {}
    stability = read_json(run_root / "measurements" / "stability_runs.json")

    candidate = trained.get("candidate", {})
    checkpoint = trained.get("checkpoint", {})
    metrics = checkpoint.get("metrics", {})
    best_eval = metrics.get("best_eval", {}) if isinstance(metrics.get("best_eval"), dict) else {}
    bitstream_text = str(build.get("bitstream", "") or "")
    bitstream = Path(bitstream_text) if bitstream_text else None
    bitstream_sha = sha256_file(bitstream) if bitstream is not None else ""
    arch_id = str(candidate.get("arch_id", run_root.name))
    timing = parse_design_timing_summary(run_root / "vivado_reports" / "timing_summary.rpt")

    if measurement:
        board_status = "measured"
        stable = stability.get("stable", "")
        stable_ok = stable is True or stable == "" or stable is None
        pass_fail = "PASS" if str(frame_value(measurement, "status_code")) == "0" and stable_ok else "MEASURE_FAIL"
    elif build.get("status"):
        board_status = "built_not_measured"
        pass_fail = "PENDING_COM5"
    else:
        board_status = "pre_board_export_only"
        pass_fail = "PENDING_VIVADO_COM5"

    return {
        "run_name": run_root.name,
        "arch_id": arch_id,
        "positioning": positioning(arch_id),
        "parameter_mode": harness.get("parameter_mode", ""),
        "checkpoint": checkpoint.get("path", ""),
        "checkpoint_sha256": checkpoint.get("sha256", ""),
        "retrain_best_epoch": metrics.get("best_epoch", ""),
        "retrain_macro_f1": best_eval.get("macro_f1", ""),
        "retrain_top1": best_eval.get("top1", ""),
        "retrain_weighted_f1": best_eval.get("weighted_f1", ""),
        "trained_weight_manifest": str(project / "trained_weight_manifest.json") if trained else "",
        "software_parity_json": str(project / "software_board_parity.json") if parity else "",
        "software_checksum_argmax_packed": parity.get("checksum_argmax_packed", ""),
        "software_checksum24": parity.get("checksum24", ""),
        "software_argmax": parity.get("argmax", ""),
        "bitstream": str(bitstream) if bitstream is not None else "",
        "bitstream_sha256": bitstream_sha,
        "wns_ns": build.get("wns_ns", "") or timing.get("wns_ns", ""),
        "tns_ns": build.get("tns_ns", "") or timing.get("tns_ns", ""),
        "lut": build.get("lut", ""),
        "dsp": build.get("dsp", ""),
        "bram": build.get("bram", ""),
        "com5_status_code": frame_value(measurement, "status_code"),
        "cycles": frame_value(measurement, "cycles"),
        "real_e2e_latency_ms": measurement.get("latency_ms", ""),
        "board_checksum": frame_value(measurement, "checksum"),
        "board_argmax": checksum_argmax(frame_value(measurement, "checksum")),
        "board_checksum24": checksum24(frame_value(measurement, "checksum")),
        "com5_run_count": stability.get("run_count", ""),
        "com5_stable": stability.get("stable", ""),
        "board_status": board_status,
        "pass_fail_note": pass_fail,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    lines = [
        "# Phase0 v3 Retrained-Weight Board Reinjection Comparison",
        "",
        f"- generated_at: `{timestamp()}`",
        "- scope: retrained checkpoint weights exported into isolated full-network board harness projects.",
        "- boundary: PyTorch retrain metrics are final software accuracy; software checksum/argmax is deterministic harness sanity only; COM5 results are board-level latency/output sanity for the deterministic harness input, not full NKSID validation-set board accuracy.",
        f"- csv: `{csv_path}`",
        f"- json: `{json_path}`",
        "",
        "| Arch | Positioning | Retrain checkpoint | Retrain macro_f1 | Retrain top1 | Retrain weighted_f1 | New bitstream SHA256 | WNS ns | TNS ns | LUT | DSP | BRAM | COM5 status_code | cycles | real_e2e_ms | Board checksum | Board argmax | COM5 runs | Stable | Status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{arch_id}` | {positioning} | `{checkpoint}` | {retrain_macro_f1} | {retrain_top1} | {retrain_weighted_f1} | {bitstream_sha256} | {wns_ns} | {tns_ns} | {lut} | {dsp} | {bram} | {com5_status_code} | {cycles} | {real_e2e_latency_ms} | {board_checksum} | {board_argmax} | {com5_run_count} | {com5_stable} | {pass_fail_note} |".format(
                **{key: row.get(key, "") for key in row}
            )
        )
    lines.extend(
        [
            "",
            "## Evidence Notes",
            "",
            "1. Search eval10, PyTorch retrain150, and board COM5 are separate result layers.",
            "2. The COM5 checksum/argmax values are from the routed retrained-weight bitstream and the deterministic harness input.",
            "3. This report does not claim full validation-set board accuracy.",
            "4. Power is not promoted as a measured board-level metric in this run.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write retrained board reinjection comparison reports.")
    parser.add_argument("--result-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.result_root).expanduser().resolve()
    rows = [row_for_run(path) for path in sorted(root.glob("rl_arch_*_trainedmem"))]
    reports = root / "reports"
    json_path = reports / "retrained_board_reinject_comparison.json"
    csv_path = reports / "retrained_board_reinject_comparison.csv"
    md_path = reports / "retrained_board_reinject_comparison.md"
    fields = [
        "arch_id",
        "positioning",
        "checkpoint",
        "checkpoint_sha256",
        "retrain_best_epoch",
        "retrain_macro_f1",
        "retrain_top1",
        "retrain_weighted_f1",
        "trained_weight_manifest",
        "software_parity_json",
        "software_checksum_argmax_packed",
        "software_checksum24",
        "software_argmax",
        "bitstream",
        "bitstream_sha256",
        "wns_ns",
        "tns_ns",
        "lut",
        "dsp",
        "bram",
        "com5_status_code",
        "cycles",
        "real_e2e_latency_ms",
        "board_checksum",
        "board_argmax",
        "board_checksum24",
        "com5_run_count",
        "com5_stable",
        "board_status",
        "pass_fail_note",
    ]
    write_json(json_path, {"generated_at": timestamp(), "result_root": str(root), "rows": rows})
    write_csv(csv_path, rows, fields)
    write_markdown(md_path, rows, csv_path, json_path)
    print(json.dumps({"rows": len(rows), "markdown": str(md_path), "json": str(json_path), "csv": str(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
