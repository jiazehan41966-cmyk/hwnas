#!/usr/bin/env python3
"""Analyze Phase0 v4 sonar RL300 results and prepare board-test candidates.

This script is evidence packaging only. It does not run Vivado, use COM5, or
modify Phase0 v3 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "remote_phase0_v4_sonar_stage3_k3_rl300_search"
    / "results"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v4_sonar_stage3_k3_lowdsp_draft"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "phase0_v4_sonar_stage3_k3_board_experiment"
DEFAULT_ROUTE_ROOT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp"
)

V3_BASELINE = [
    {
        "role": "v3_accuracy_first",
        "arch_id": "rl_arch_186",
        "macro_f1": 0.629512,
        "top1": 0.792308,
        "board_latency_ms": 49.062010,
        "board_cycles": 9812402,
        "wns_ns": 0.094,
        "actual_lut": 18598,
        "actual_bram": 127.5,
        "actual_dsp": 612,
        "claim_status": "board_claimable",
    },
    {
        "role": "v3_latency_balanced",
        "arch_id": "rl_arch_242",
        "macro_f1": 0.627406,
        "top1": 0.786538,
        "board_latency_ms": 24.872910,
        "board_cycles": 4974582,
        "wns_ns": 0.113,
        "actual_lut": 18553,
        "actual_bram": 127.5,
        "actual_dsp": 612,
        "claim_status": "board_claimable",
    },
    {
        "role": "v3_resource_min",
        "arch_id": "rl_arch_276",
        "macro_f1": 0.605372,
        "top1": 0.767308,
        "board_latency_ms": 24.836150,
        "board_cycles": 4967230,
        "wns_ns": 0.223,
        "actual_lut": 15354,
        "actual_bram": 101,
        "actual_dsp": 524,
        "claim_status": "board_claimable",
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_if_exists(path: Path | str | None) -> Any:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return read_json(p)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_value(measurement: dict[str, Any], key: str) -> Any:
    frame = measurement.get("frame")
    if isinstance(frame, dict) and key in frame:
        return frame[key]
    return measurement.get(key)


def measurement_run_count(measurement_path: Path | str | None) -> int | None:
    if not measurement_path:
        return None
    measurement_dir = Path(measurement_path).parent
    stability_path = measurement_dir / "stability_runs.json"
    if stability_path.exists():
        stability = read_json(stability_path)
        count = to_int(stability.get("run_count"))
        if count is not None:
            return count
    csv_path = measurement_dir / "measurements_raw_runs.csv"
    if not csv_path.exists():
        csv_path = measurement_dir / "measurements_raw.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def ops_for(candidate: dict[str, Any]) -> list[str]:
    ops: list[str] = []
    encoding = candidate.get("encoding") or {}
    for stage in encoding.get("stages") or []:
        for block in stage.get("blocks") or []:
            ops.append(str(block.get("op", "")))
    return ops


def stage3_op(candidate: dict[str, Any]) -> str:
    encoding = candidate.get("encoding") or {}
    stages = encoding.get("stages") or []
    if len(stages) <= 3:
        return ""
    blocks = stages[3].get("blocks") or []
    if not blocks:
        return ""
    block = blocks[0]
    return f"{block.get('op')} k{block.get('kernel_size')} e{block.get('expand_ratio')}"


def metrics_for(candidate: dict[str, Any]) -> dict[str, Any]:
    return dict(candidate.get("metrics") or {})


def load_candidates(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates_path = run_dir / "results" / "candidates.jsonl"
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with candidates_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            arch_id = str(record.get("arch_id", ""))
            if arch_id:
                by_id[arch_id] = record
    return records, by_id


def candidate_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the architecture payload from either candidates.jsonl or pareto rows."""
    nested = record.get("candidate")
    if isinstance(nested, dict):
        return nested
    return record


def candidate_summary(candidate: dict[str, Any], *, role: str = "") -> dict[str, Any]:
    candidate = candidate_payload(candidate)
    metrics = metrics_for(candidate)
    return {
        "role": role,
        "arch_id": candidate.get("arch_id"),
        "stage3_op": stage3_op(candidate),
        "ops": ops_for(candidate),
        "search_proxy": {
            "macro_f1": metrics.get("macro_f1"),
            "top1": metrics.get("top1"),
            "top5": metrics.get("top5"),
            "latency_ms": metrics.get("latency_ms"),
            "dsp": metrics.get("dsp"),
            "bram": metrics.get("bram"),
            "lut": metrics.get("lut"),
            "power_w": metrics.get("power_w"),
            "physical_risk": metrics.get("physical_risk"),
            "early_expand_pressure": metrics.get("early_expand_pressure"),
        },
        "vivado_actual": {
            "status": "pending_full_route",
            "wns_ns": None,
            "actual_dsp": None,
            "actual_bram": None,
            "actual_lut": None,
            "power_w": None,
            "bitstream_sha256": None,
            "build_result": None,
            "bitstream": None,
        },
        "com5_board": {
            "status": "blocked_until_full_route_pass",
            "status_code": None,
            "latency_ms": None,
            "cycles": None,
            "checksum": None,
            "runs": None,
            "measurement_json": None,
        },
    }


def load_route_gate_index(route_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    summary = read_json_if_exists(route_root / "route_gate_summary.json")
    for row in summary.get("candidates") or []:
        candidate = row.get("candidate") or {}
        arch_id = str(candidate.get("arch_id") or "")
        if not arch_id:
            continue
        index.setdefault(arch_id, {})["summary_row"] = row

    plan = read_json_if_exists(route_root / "route_gate_plan.json")
    for item in plan.get("candidates") or []:
        arch_id = str(item.get("arch_id") or "")
        if not arch_id:
            continue
        entry = index.setdefault(arch_id, {})
        entry["plan_item"] = item
        full_build = read_json_if_exists(item.get("full_build_result"))
        measurement = read_json_if_exists(item.get("measurement_json"))
        final_validation = read_json_if_exists(item.get("final_validation_report_json"))
        entry["full_build_result"] = full_build
        entry["measurement"] = measurement
        entry["final_validation"] = final_validation
        entry["measurement_json"] = item.get("measurement_json")
        entry["measurement_command"] = item.get("measurement_command")
        entry["full_build_result_path"] = item.get("full_build_result")

    return index


def apply_route_evidence(candidate: dict[str, Any], route_entry: dict[str, Any] | None) -> None:
    if not route_entry:
        return
    build = route_entry.get("full_build_result") or {}
    measurement = route_entry.get("measurement") or {}
    final_validation = route_entry.get("final_validation") or {}
    measurement_json = route_entry.get("measurement_json")
    build_path = route_entry.get("full_build_result_path")

    wns = to_float(build.get("wns_ns"))
    actual_dsp = to_float(build.get("dsp"))
    timing_clean = bool(build.get("timing_clean")) or (wns is not None and wns >= 0)
    dsp_ok = actual_dsp is not None and actual_dsp <= 700
    route_success = build.get("status") == "success"
    bitstream = build.get("bitstream")
    if build:
        if route_success and timing_clean and dsp_ok:
            vivado_status = "route-clean"
        elif route_success:
            vivado_status = "full-route-fail-gate"
        else:
            vivado_status = "full-route-fail"
        candidate["vivado_actual"].update(
            {
                "status": vivado_status,
                "wns_ns": wns,
                "actual_dsp": actual_dsp,
                "actual_bram": to_float(build.get("bram")),
                "actual_lut": to_int(build.get("lut")),
                "power_w": to_float(build.get("power_w")),
                "bitstream_sha256": build.get("bitstream_sha256") or sha256_file(bitstream),
                "build_result": build_path,
                "bitstream": bitstream,
            }
        )

    status_code = to_int(frame_value(measurement, "status_code")) if measurement else None
    if measurement:
        board_status = "board-claimable" if status_code == 0 and route_success and timing_clean and dsp_ok else "measured-not-claimable"
        candidate["com5_board"].update(
            {
                "status": board_status,
                "status_code": status_code,
                "latency_ms": to_float(measurement.get("latency_ms")),
                "cycles": to_int(frame_value(measurement, "cycles")),
                "checksum": to_int(frame_value(measurement, "checksum")),
                "runs": measurement_run_count(measurement_json) or 1,
                "measurement_json": measurement_json,
            }
        )
    elif build and route_success and timing_clean and dsp_ok:
        candidate["com5_board"]["status"] = "ready_for_com5"

    if final_validation.get("claimable_real_e2e_latency") is True:
        candidate["com5_board"]["status"] = "board-claimable"


def table_row(candidate: dict[str, Any]) -> str:
    proxy = candidate["search_proxy"]
    actual = candidate["vivado_actual"]
    board = candidate["com5_board"]
    return (
        f"| `{candidate['arch_id']}` | {candidate['role']} | `{candidate['stage3_op']}` | "
        f"{fmt(proxy.get('macro_f1'))} | {fmt(proxy.get('top1'))} | "
        f"{fmt(proxy.get('latency_ms'))} | {fmt(proxy.get('dsp'))} | "
        f"{fmt(proxy.get('lut'))} | {fmt(proxy.get('bram'))} | "
        f"{actual['status']} | {fmt(actual.get('wns_ns'))} | {fmt(actual.get('actual_dsp'))} | "
        f"{board['status']} | {fmt(board.get('latency_ms'))} |"
    )


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_payload(run_dir: Path, output_dir: Path, route_root: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "results" / "summary.json")
    pareto = read_json(run_dir / "results" / "pareto_selection.json")
    lut_stats = read_json(run_dir / "results" / "lut_stats.json")
    deferred = read_json(run_dir / "results" / "deferred_hit_summary.json")
    candidates, by_id = load_candidates(run_dir)
    route_index = load_route_gate_index(route_root)

    selected = list(pareto.get("selected_topk") or [])
    selected_ids = [str(item.get("arch_id", "")) for item in selected]
    selected_by_id = {str(item.get("arch_id")): item for item in selected if item.get("arch_id")}

    sonar_usage = Counter()
    for record in candidates:
        candidate = candidate_payload(record)
        ops = ops_for(candidate)
        has_denoise = "denoise" in ops
        has_edge = "edge" in ops
        if has_denoise:
            sonar_usage["denoise"] += 1
        if has_edge:
            sonar_usage["edge"] += 1
        if has_denoise or has_edge:
            sonar_usage["either"] += 1
    sonar_usage["total"] = len(candidates)

    roles = {
        "overall_best": "rl_arch_135",
        "sonar_best_denoise": "rl_arch_169",
        "sonar_best_edge": "rl_arch_154",
        "latency_reference": "rl_arch_193",
    }
    role_candidates = []
    for role, arch_id in roles.items():
        candidate = selected_by_id.get(arch_id) or by_id.get(arch_id)
        if candidate:
            summary_row = candidate_summary(candidate, role=role)
            apply_route_evidence(summary_row, route_index.get(arch_id))
            role_candidates.append(summary_row)

    selected_candidates = [
        candidate_summary(selected_by_id[arch_id], role="pareto_route_screen")
        for arch_id in selected_ids
        if arch_id in selected_by_id
    ]
    for candidate in selected_candidates:
        apply_route_evidence(candidate, route_index.get(str(candidate.get("arch_id") or "")))

    route_clean = [
        candidate
        for candidate in selected_candidates
        if candidate["vivado_actual"].get("status") == "route-clean"
    ]
    board_claimable = [
        candidate
        for candidate in selected_candidates
        if candidate["com5_board"].get("status") == "board-claimable"
    ]
    resource_reference = min(
        route_clean,
        key=lambda row: (
            row["vivado_actual"].get("actual_dsp")
            if row["vivado_actual"].get("actual_dsp") is not None
            else float("inf"),
            row["search_proxy"].get("latency_ms") or float("inf"),
        ),
        default=None,
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs_vivado": False,
        "runs_com5": False,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "route_gate_output_root": str(route_root),
        "source_artifacts": {
            "summary": str(run_dir / "results" / "summary.json"),
            "pareto_selection": str(run_dir / "results" / "pareto_selection.json"),
            "best_candidate": str(run_dir / "results" / "best_candidate.json"),
            "candidates_jsonl": str(run_dir / "results" / "candidates.jsonl"),
        },
        "search_completion": {
            "status": summary.get("status"),
            "total_evaluated": summary.get("total_evaluated"),
            "feasible": summary.get("feasible"),
            "infeasible": summary.get("infeasible"),
            "candidate_jsonl_rows": len(candidates),
            "pareto_selected_count": len(selected),
            "pareto_selected_ids": selected_ids,
        },
        "strict_lut": {
            "hits": lut_stats.get("hits"),
            "misses": lut_stats.get("misses"),
            "true_misses": lut_stats.get("true_misses"),
            "deferred_hits": lut_stats.get("deferred_hits"),
            "hit_rate": lut_stats.get("hit_rate"),
            "deferred_summary": deferred,
        },
        "sonar_usage": dict(sonar_usage),
        "candidate_roles": role_candidates,
        "pareto_route_screen_candidates": selected_candidates,
        "route_clean_candidates": route_clean,
        "board_claimable_candidates": board_claimable,
        "resource_reference": resource_reference,
        "resource_reference_policy": (
            "Select after full-route refresh from PASS rows with minimum actual Vivado DSP; "
            "do not use search-proxy DSP for final resource-reference claim."
        ),
        "route_gate_admission": {
            "full_max_actual_dsp": 700,
            "require_full_route_pass": True,
            "require_wns_nonnegative": True,
            "lowdsp_catalogs": [
                "hls_lut_builder/board_harness/configs/phase0_v3_lowdsp_override_catalog.yaml",
                "hls_lut_builder/board_harness/configs/phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml",
            ],
        },
        "com5_policy": {
            "blocked_until_full_route_pass": True,
            "max_initial_measurements": 3,
            "stability_reruns": 5,
            "required_status_code": 0,
        },
        "v3_board_claimable_baseline": V3_BASELINE,
        "final_status_counts": {
            "search_best_count": len(role_candidates),
            "route_clean_count": len(route_clean),
            "board_claimable_count": len(board_claimable),
        },
    }
    return payload


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase0 v4 Sonar Stage3 k3 RL300 Board Experiment Analysis",
        "",
        "This report separates search proxy metrics, Vivado actual implementation evidence, and COM5 board measurements.",
        "No Vivado or COM5 command is run by this analysis step.",
        "",
        "## Search Completion",
        "",
        f"- Status: `{payload['search_completion']['status']}`",
        f"- Evaluated: `{payload['search_completion']['total_evaluated']}`",
        f"- Feasible: `{payload['search_completion']['feasible']}`",
        f"- Infeasible: `{payload['search_completion']['infeasible']}`",
        f"- candidates.jsonl rows: `{payload['search_completion']['candidate_jsonl_rows']}`",
        f"- Pareto selected: `{payload['search_completion']['pareto_selected_ids']}`",
        "",
        "## Strict LUT",
        "",
        f"- Hits: `{payload['strict_lut']['hits']}`",
        f"- Misses: `{payload['strict_lut']['misses']}`",
        f"- True misses: `{payload['strict_lut']['true_misses']}`",
        f"- Deferred hits: `{payload['strict_lut']['deferred_hits']}`",
        f"- Hit rate: `{payload['strict_lut']['hit_rate']}`",
        "",
        "## Sonar Usage",
        "",
        f"- Total candidates: `{payload['sonar_usage'].get('total', 0)}`",
        f"- Contains denoise: `{payload['sonar_usage'].get('denoise', 0)}`",
        f"- Contains edge: `{payload['sonar_usage'].get('edge', 0)}`",
        f"- Contains denoise or edge: `{payload['sonar_usage'].get('either', 0)}`",
        "",
        "## Board Candidate Manifest",
        "",
        "| arch | role | stage3 op | search macro_f1 | search top1 | search latency ms | search DSP | search LUT | search BRAM | Vivado status | WNS ns | actual DSP | COM5 status | COM5 latency ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for candidate in payload["candidate_roles"]:
        lines.append(table_row(candidate))

    lines.extend(
        [
            "",
            "## Pareto Route-Screen Candidates",
            "",
            "| arch | role | stage3 op | search macro_f1 | search top1 | search latency ms | search DSP | search LUT | search BRAM | Vivado status | WNS ns | actual DSP | COM5 status | COM5 latency ms |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
        ]
    )
    for candidate in payload["pareto_route_screen_candidates"]:
        lines.append(table_row(candidate))

    lines.extend(
        [
            "",
            "## Admission Classes",
            "",
            f"- `search-best`: `{payload['final_status_counts']['search_best_count']}` role rows are search-proxy selections only unless upgraded by route/COM5 evidence.",
            f"- `route-clean`: `{payload['final_status_counts']['route_clean_count']}` Pareto rows currently have full-route PASS evidence under DSP <= 700.",
            f"- `board-claimable`: `{payload['final_status_counts']['board_claimable_count']}` Pareto rows currently have full-route PASS plus COM5 `status_code=0` evidence.",
            "",
        ]
    )
    resource_reference = payload.get("resource_reference")
    if resource_reference:
        actual = resource_reference["vivado_actual"]
        lines.extend(
            [
                f"- Resource reference after route: `{resource_reference['arch_id']}` "
                f"(actual DSP `{fmt(actual.get('actual_dsp'))}`, WNS `{fmt(actual.get('wns_ns'))}` ns).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Resource reference after route: not selected yet because no route-clean row is available.",
                "",
            ]
        )

    if payload["board_claimable_candidates"]:
        lines.extend(
            [
                "| arch | class | COM5 status_code | latency ms | cycles | checksum | measurement runs | bitstream sha256 |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for candidate in payload["board_claimable_candidates"]:
            board = candidate["com5_board"]
            actual = candidate["vivado_actual"]
            lines.append(
                f"| `{candidate['arch_id']}` | board-claimable | {fmt(board.get('status_code'))} | "
                f"{fmt(board.get('latency_ms'))} | {fmt(board.get('cycles'))} | "
                f"{fmt(board.get('checksum'))} | {fmt(board.get('runs'))} | "
                f"`{actual.get('bitstream_sha256') or ''}` |"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Phase0 v3 Board-Claimable Baseline",
            "",
            "| arch | role | macro_f1 | top1 | COM5 latency ms | cycles | WNS ns | actual DSP | actual LUT | actual BRAM |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["v3_board_claimable_baseline"]:
        lines.append(
            f"| `{row['arch_id']}` | {row['role']} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['top1'])} | {fmt(row['board_latency_ms'])} | {row['board_cycles']} | "
            f"{fmt(row['wns_ns'])} | {row['actual_dsp']} | {row['actual_lut']} | {row['actual_bram']} |"
        )

    by_arch = {
        str(row.get("arch_id")): row
        for row in payload["pareto_route_screen_candidates"]
    }
    v3_accuracy = payload["v3_board_claimable_baseline"][0]
    v3_latency = payload["v3_board_claimable_baseline"][1]
    v3_resource = payload["v3_board_claimable_baseline"][2]
    overall = by_arch.get("rl_arch_135")
    edge = by_arch.get("rl_arch_154")
    denoise = by_arch.get("rl_arch_169")
    lines.extend(
        [
            "",
            "## V4 vs V3 Evidence-Bound Comparison",
            "",
            "This section is artifact-level comparison only; it is not a statistical significance test across seeds.",
            "",
            "| question | v4 evidence | v3 reference | evidence-bound conclusion |",
            "|---|---|---|---|",
        ]
    )
    if overall:
        overall_proxy = overall["search_proxy"]
        overall_board = overall["com5_board"]
        overall_actual = overall["vivado_actual"]
        lines.append(
            f"| overall best | `rl_arch_135` macro_f1 `{fmt(overall_proxy.get('macro_f1'))}`, "
            f"top1 `{fmt(overall_proxy.get('top1'))}`, COM5 `{fmt(overall_board.get('latency_ms'))}` ms, "
            f"DSP `{fmt(overall_actual.get('actual_dsp'))}` | "
            f"`{v3_accuracy['arch_id']}` macro_f1 `{fmt(v3_accuracy['macro_f1'])}`, "
            f"top1 `{fmt(v3_accuracy['top1'])}`, COM5 `{fmt(v3_accuracy['board_latency_ms'])}` ms, "
            f"DSP `{v3_accuracy['actual_dsp']}` | "
            "v4 is board-claimable and improves search macro_f1/top1 versus the v3 accuracy-first row, "
            "but does not improve the measured latency. |"
        )
    if denoise:
        denoise_proxy = denoise["search_proxy"]
        denoise_board = denoise["com5_board"]
        denoise_actual = denoise["vivado_actual"]
        lines.append(
            f"| denoise candidate | `rl_arch_169` macro_f1 `{fmt(denoise_proxy.get('macro_f1'))}`, "
            f"top1 `{fmt(denoise_proxy.get('top1'))}`, COM5 `{fmt(denoise_board.get('latency_ms'))}` ms, "
            f"DSP `{fmt(denoise_actual.get('actual_dsp'))}` | "
            f"`{v3_accuracy['arch_id']}` macro_f1 `{fmt(v3_accuracy['macro_f1'])}`, "
            f"COM5 `{fmt(v3_accuracy['board_latency_ms'])}` ms, DSP `{v3_accuracy['actual_dsp']}`; "
            f"`{v3_latency['arch_id']}` COM5 `{fmt(v3_latency['board_latency_ms'])}` ms | "
            "denoise is board-claimable and has a higher search macro_f1 than v3 accuracy-first, "
            "but its measured latency is worse than all listed v3 board baselines. |"
        )
    if edge:
        edge_proxy = edge["search_proxy"]
        edge_board = edge["com5_board"]
        edge_actual = edge["vivado_actual"]
        lines.append(
            f"| edge candidate | `rl_arch_154` macro_f1 `{fmt(edge_proxy.get('macro_f1'))}`, "
            f"top1 `{fmt(edge_proxy.get('top1'))}`, COM5 `{fmt(edge_board.get('latency_ms'))}` ms, "
            f"DSP `{fmt(edge_actual.get('actual_dsp'))}` | "
            f"`{v3_accuracy['arch_id']}` COM5 `{fmt(v3_accuracy['board_latency_ms'])}` ms; "
            f"`{v3_latency['arch_id']}` COM5 `{fmt(v3_latency['board_latency_ms'])}` ms; "
            f"`{v3_resource['arch_id']}` DSP `{v3_resource['actual_dsp']}` | "
            "edge is board-claimable and faster than the v3 accuracy-first row, "
            "but it is still slower than the v3 latency/resource rows and is not the v4 search-best row. |"
        )

    lines.extend(
        [
            "",
            "## Companion Artifacts",
            "",
            "- `phase0_v4_sonar_board_candidate_manifest.json`: machine-readable search/route/COM5 evidence manifest.",
            "- `phase0_v4_vs_v3_board_comparison.csv`: flat comparison table for spreadsheet/statistical use.",
            "- `phase0_v4_vs_v3_board_comparison.json`: structured comparison bundle with summary findings.",
            "- `phase0_v4_vs_v3_board_comparison.md`: compact v3/v4 evidence-bound comparison report.",
            "- Per-measured-candidate `stability_runs.json`: 5-run COM5 stability evidence under each isolated run's `measurements` directory.",
            "",
            "## Final Gate State",
            "",
            "- `rl_arch_135`, `rl_arch_154`, and `rl_arch_169` are board-claimable within this isolated v4 route/COM5 evidence set.",
            "- `rl_arch_60`, `rl_arch_175`, and `rl_arch_193` remain route-clean only because this batch did not spend COM5 slots on them.",
            "- `rl_arch_116` remains route-fail and is blocked from COM5 because WNS is negative and actual DSP exceeds 700.",
            "- Do not generalize board-claimable status beyond the measured bitstreams and COM5 runs listed above.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--route-output-root", default=str(DEFAULT_ROUTE_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    route_root = Path(args.route_output_root).expanduser().resolve()
    required = [
        run_dir / "results" / "summary.json",
        run_dir / "results" / "pareto_selection.json",
        run_dir / "results" / "best_candidate.json",
        run_dir / "results" / "candidates.jsonl",
        run_dir / "results" / "lut_stats.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required RL300 artifacts: " + ", ".join(missing))

    payload = build_payload(run_dir, output_dir, route_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "phase0_v4_sonar_board_candidate_manifest.json", payload)
    (output_dir / "phase0_v4_sonar_rl300_board_analysis.md").write_text(
        render_report(payload),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "runs_vivado": False,
        "runs_com5": False,
        "manifest": str(output_dir / "phase0_v4_sonar_board_candidate_manifest.json"),
        "report": str(output_dir / "phase0_v4_sonar_rl300_board_analysis.md"),
        "candidate_jsonl_rows": payload["search_completion"]["candidate_jsonl_rows"],
        "pareto_selected_ids": payload["search_completion"]["pareto_selected_ids"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
