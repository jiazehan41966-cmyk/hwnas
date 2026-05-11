#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILDER_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import build_cases, load_config
from hwnas_fpga.hardware import parse_vivado_utilization_text
from hwnas_fpga.hardware.lookup_table import LutEntry, LutTable, OpSpec


DEFAULT_CONFIG = BUILDER_ROOT / "configs" / "candidate_kernels.yaml"
DEFAULT_BOARD_RESULTS = BUILDER_ROOT / "board_harness" / "results" / "board_measured_lut.csv"
DEFAULT_STATUS_CSV = BUILDER_ROOT / "board_harness" / "results" / "board_measure_status_current_impl.csv"
DEFAULT_DEFERRED_CSV = BUILDER_ROOT / "board_harness" / "results" / "board_measure_deferred_cases_current_impl.csv"
DEFAULT_LUT_JSON = BUILDER_ROOT / "results" / "formal_lut_v1.json"
DEFAULT_STATUS_JSON = BUILDER_ROOT / "results" / "formal_lut_status_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate formal defer-aware LUT files from board measurement results."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="candidate_kernels.yaml path")
    parser.add_argument("--board-results", default=str(DEFAULT_BOARD_RESULTS), help="board_measured_lut.csv path")
    parser.add_argument("--status-csv", default=str(DEFAULT_STATUS_CSV), help="current implementation status CSV")
    parser.add_argument("--deferred-csv", default=str(DEFAULT_DEFERRED_CSV), help="deferred classification CSV")
    parser.add_argument("--output-lut-json", default=str(DEFAULT_LUT_JSON), help="output measured LUT JSON")
    parser.add_argument("--output-status-json", default=str(DEFAULT_STATUS_JSON), help="output formal status JSON")
    parser.add_argument(
        "--status-authoritative",
        action="store_true",
        help=(
            "Treat the status CSV as the only measured/deferred authority. "
            "Successful board rows will provide metrics, but will not override "
            "deferred status rows into measured entries."
        ),
    )
    parser.add_argument(
        "--implementation-version",
        default="packed_stream_v1_current_impl_2026_05_06",
        help="implementation label written into metadata",
    )
    return parser.parse_args()


def _latest_by_case(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    latest: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_name = str(row.get("case_name") or "").strip()
            if not case_name:
                continue
            previous = latest.get(case_name)
            if previous is None or str(row.get("timestamp") or "") > str(previous.get("timestamp") or ""):
                latest[case_name] = row
    return latest


def _rows_by_case(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_name = str(row.get("case_name") or "").strip()
            if case_name:
                rows[case_name] = row
    return rows


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _is_successful_board_row(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    status_code = str(row.get("status_code") or "").strip()
    status_label = str(row.get("status_label") or "").strip().lower()
    if status_code and status_code not in {"0", "0.0"}:
        return False
    if status_label and status_label not in {"ok", "success", "passed"}:
        return False
    return bool(str(row.get("cycles") or "").strip() and str(row.get("latency_ms") or "").strip())


def _query_opspec(raw: dict[str, Any]) -> OpSpec:
    spec = dict(raw)
    spec["input_resolution"] = tuple(spec.get("input_resolution", (224, 224)))
    # Search-time OpSpec does not include a clock field. Strip it so strict
    # formal lookup uses the same key as the NAS query path.
    spec["target_clock_mhz"] = None
    return OpSpec.from_dict(spec)


def _utilization_for_case(case_name: str, case_project_dir: Path, board_row: dict[str, str]) -> dict[str, Any]:
    candidates = [
        case_project_dir / "vivado_downstream" / "reports" / "post_route_utilization.rpt",
    ]
    manifest_path = Path(str(board_row.get("manifest") or "")).expanduser()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            project_root = Path(str(manifest["generated"]["project_root"]))
            candidates.append(project_root / "reports" / "utilization.rpt")
        except Exception:
            pass

    for report in candidates:
        if report.exists():
            metrics = parse_vivado_utilization_text(
                report.read_text(encoding="utf-8", errors="ignore")
            )
            metrics["report"] = str(report)
            return metrics

    return {"lut": 0, "ff": 0, "block_ram_tile": 0.0, "dsp": 0, "report": None}


def _status_payload(
    *,
    case_name: str,
    status_row: dict[str, str],
    deferred_row: dict[str, str] | None,
    op_spec: OpSpec,
    case_operator: str,
    board_row: dict[str, str] | None,
    status_source: str,
    original_status: str | None = None,
) -> dict[str, Any]:
    status = str(status_row.get("status") or "missing").strip()
    detail = str(status_row.get("detail") or "").strip()
    root_cause = None if deferred_row is None else str(deferred_row.get("root_cause_bucket") or "").strip() or None
    defer_reason = None
    if status != "measured":
        defer_reason = (
            str(deferred_row.get("defer_reason") or "").strip()
            if deferred_row is not None
            else detail
        ) or None

    return {
        "case_name": case_name,
        "status": status,
        "op_type": case_operator,
        "lookup_op": op_spec.op,
        "defer_reason": defer_reason,
        "root_cause_bucket": root_cause,
        "status_source": status_source,
        "original_status": original_status,
        "op_spec": op_spec.to_dict(),
        "board_cycles": None if board_row is None else _to_int(board_row.get("cycles")),
        "board_latency_ms": None if board_row is None else _to_float(board_row.get("latency_ms")),
        "detail": detail,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    board_results_path = Path(args.board_results).expanduser().resolve()
    status_csv_path = Path(args.status_csv).expanduser().resolve()
    deferred_csv_path = Path(args.deferred_csv).expanduser().resolve()
    output_lut_json = Path(args.output_lut_json).expanduser().resolve()
    output_status_json = Path(args.output_status_json).expanduser().resolve()

    config = load_config(config_path)
    cases = build_cases(config)
    by_case = {case.case_name: case for case in cases}
    measured_rows = _latest_by_case(board_results_path)
    status_rows = _rows_by_case(status_csv_path)
    deferred_rows = _rows_by_case(deferred_csv_path)

    missing_status = sorted(set(by_case) - set(status_rows))
    if missing_status:
        raise ValueError(f"Status CSV is missing {len(missing_status)} candidate cases")

    entries: list[LutEntry] = []
    status_entries: list[dict[str, Any]] = []
    resource_sources = Counter()
    board_overrides: list[str] = []

    for case in cases:
        op_spec = _query_opspec(case.op_spec)
        status_row = status_rows[case.case_name]
        board_row = measured_rows.get(case.case_name)
        deferred_row = deferred_rows.get(case.case_name)
        original_status = str(status_row.get("status") or "missing").strip()
        effective_status_row = dict(status_row)
        status_source = "status_csv"
        if not args.status_authoritative and _is_successful_board_row(board_row):
            effective_status_row["status"] = "measured"
            if original_status != "measured":
                effective_status_row["detail"] = "board_results_overlay"
                status_source = "board_results_overlay"
                board_overrides.append(case.case_name)

        status_entries.append(
            _status_payload(
                case_name=case.case_name,
                status_row=effective_status_row,
                deferred_row=deferred_row,
                op_spec=op_spec,
                case_operator=case.operator,
                board_row=board_row,
                status_source=status_source,
                original_status=None if original_status == effective_status_row["status"] else original_status,
            )
        )

        if effective_status_row.get("status") != "measured":
            continue
        if not _is_successful_board_row(board_row):
            if args.status_authoritative:
                raise ValueError(
                    "Status CSV marks case as measured, but board results do not "
                    f"contain a successful metric row: {case.case_name}"
                )
            continue

        utilization = _utilization_for_case(case.case_name, case.project_dir, board_row)
        source = utilization.get("report") or "missing"
        resource_sources[str(source)] += 1
        entries.append(
            LutEntry(
                op_spec=op_spec,
                latency_ms=_to_float(board_row.get("latency_ms")),
                cycles=_to_int(board_row.get("cycles")),
                dsp=_to_int(utilization.get("dsp")),
                bram=int(round(_to_float(utilization.get("block_ram_tile")))),
                lut=_to_int(utilization.get("lut")),
                power_w=0.0,
                energy_mj=0.0,
            )
        )

    table = LutTable(entries)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "implementation_version": args.implementation_version,
        "measurement_contract": config.get("measurement_contract"),
        "source_config": str(config_path),
        "source_board_results": str(board_results_path),
        "source_status_csv": str(status_csv_path),
        "source_deferred_csv": str(deferred_csv_path),
        "status_authoritative": bool(args.status_authoritative),
        "candidate_cases": len(cases),
        "measured_entries": len(entries),
        "status_counts": dict(Counter(entry["status"] for entry in status_entries)),
        "board_result_status_overrides": len(board_overrides),
        "board_result_status_override_cases": board_overrides,
        "resource_source_counts": dict(resource_sources),
        "query_key_note": "target_clock_mhz is intentionally set to null to match search-time OpSpec keys.",
    }

    output_lut_json.parent.mkdir(parents=True, exist_ok=True)
    table.save_json(str(output_lut_json), metadata=metadata)

    status_payload = {
        "metadata": metadata,
        "entries": status_entries,
    }
    output_status_json.parent.mkdir(parents=True, exist_ok=True)
    output_status_json.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Generated formal LUT: "
        f"{len(entries)} measured entries, "
        f"{len(status_entries)} status entries -> {output_lut_json}, {output_status_json}"
    )


if __name__ == "__main__":
    main()
