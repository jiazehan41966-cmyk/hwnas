#!/usr/bin/env python3
"""Build Phase0 v4 three-lane closure artifacts.

The default mode is packaging-only: it reads existing search, route, COM5,
and retrain artifacts, then writes evidence tables and reproducible command
plans. Long-running retrain or board measurement actions require explicit
flags.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_V4_MANIFEST = (
    REPO_ROOT
    / "results"
    / "phase0_v4_sonar_stage3_k3_board_experiment"
    / "phase0_v4_sonar_board_candidate_manifest.json"
)
DEFAULT_V4_OUTPUT_DIR = REPO_ROOT / "results" / "phase0_v4_sonar_stage3_k3_board_experiment"
DEFAULT_V4_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml"
)
DEFAULT_ROUTE_PLAN = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp"
    / "route_gate_plan.json"
)
DEFAULT_V3_RETRAIN_COMPARISON = (
    REPO_ROOT
    / "results"
    / "retrain_phase0_v3_claimable_20260606"
    / "phase0_v3_claimable_retrain_comparison.json"
)
DEFAULT_RETRAIN_ROOT = REPO_ROOT / "results" / "retrain_phase0_v4_sonar_stage3_k3_topk_20260621"
DEFAULT_ABLATION_ROOT = REPO_ROOT / "results" / "phase0_v4_sonar_ablation_rl300_20260621"
DEFAULT_ABLATION_CONFIG_DIR = REPO_ROOT / "configs" / "search" / "phase0_v4_sonar_ablation_20260621"
DEFAULT_PYTHON = REPO_ROOT / ".venv_cuda" / "Scripts" / "python.exe"

TOPK_RETRAIN_IDS = ["rl_arch_135", "rl_arch_154", "rl_arch_169", "rl_arch_193", "rl_arch_60"]
V4_COM5_INITIAL_IDS = ["rl_arch_60", "rl_arch_175", "rl_arch_193"]
V4_COM5_RECHECK_IDS = ["rl_arch_135", "rl_arch_154", "rl_arch_169"]

EVIDENCE_FIELDS = [
    "arch_id",
    "phase",
    "role",
    "stage3_op",
    "macro_f1_source",
    "top1_source",
    "search_macro_f1",
    "search_top1",
    "retrain_status",
    "retrain_macro_f1",
    "retrain_top1",
    "retrain_summary",
    "checkpoint",
    "vivado_status",
    "actual_wns_ns",
    "actual_dsp",
    "actual_lut",
    "actual_bram",
    "com5_status",
    "com5_status_code",
    "com5_latency_ms",
    "com5_cycles",
    "measurement_runs",
    "board_val_accuracy_status",
    "power_energy_status",
    "claim_boundary",
]


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_if_exists(path: Path | str | None) -> Any:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return read_json(p)


def resolve_repo_path(path: Path | str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def sha256_file(path: Path | str | None) -> str:
    p = resolve_repo_path(path)
    if not p or not p.exists() or not p.is_file():
        return ""
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def load_route_plan_index(route_plan_path: Path) -> dict[str, dict[str, Any]]:
    plan = read_json_if_exists(route_plan_path)
    index: dict[str, dict[str, Any]] = {}
    for item in plan.get("candidates") or []:
        arch_id = str(item.get("arch_id") or "")
        if arch_id:
            index[arch_id] = item
    return index


def row_by_arch(rows: list[dict[str, Any]], arch_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("arch_id")) == arch_id:
            return row
    return {}


def load_v3_retrain_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_if_exists(path)
    rows = []
    if isinstance(payload, dict):
        rows = payload.get("records") or payload.get("rows") or []
    return {
        str(row.get("arch_id")): row
        for row in rows or []
        if isinstance(row, dict) and row.get("arch_id")
    }


def load_retrain_summary(run_dir: Path) -> tuple[str, dict[str, Any], Path, Path]:
    summary_path = run_dir / "results" / "retrain_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "final_best_model.pt"
    summary = read_json_if_exists(summary_path)
    if summary and checkpoint_path.exists():
        return "completed", summary, summary_path, checkpoint_path
    if summary:
        return "summary_without_checkpoint", summary, summary_path, checkpoint_path
    return "not run", {}, summary_path, checkpoint_path


def metric_from_retrain_summary(summary: dict[str, Any], key: str) -> Any:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    if key in metrics:
        return metrics.get(key)
    best_eval = metrics.get("best_eval") if isinstance(metrics.get("best_eval"), dict) else {}
    if key in best_eval:
        return best_eval.get(key)
    # Older retrain records use best_val_acc for top1.
    if key == "top1":
        return metrics.get("best_val_acc")
    return None


def build_v4_retrain_run_name(arch_id: str, device: str, seed: int, epochs: int) -> str:
    safe_device = device.replace(":", "_").replace("\\", "_").replace("/", "_")
    return f"{arch_id}_retrain{epochs}_seed{seed}_{safe_device}"


def command_for_display(command: list[str]) -> str:
    parts = []
    for part in command:
        text = str(part)
        if any(ch.isspace() for ch in text) or "\\" in text:
            parts.append('"' + text.replace('"', '`"') + '"')
        else:
            parts.append(text)
    return " ".join(parts)


def build_retrain_artifacts(
    *,
    manifest: dict[str, Any],
    route_index: dict[str, dict[str, Any]],
    v3_retrain_index: dict[str, dict[str, Any]],
    output_root: Path,
    config_path: Path,
    python_path: Path,
    seed: int,
    epochs: int,
    device: str,
    run_retrain: bool,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    v4_candidates = manifest.get("pareto_route_screen_candidates") or []

    for arch_id in TOPK_RETRAIN_IDS:
        route_item = route_index.get(arch_id, {})
        candidate_path = route_item.get("candidate_path") or ""
        candidate = row_by_arch(v4_candidates, arch_id)
        run_name = build_v4_retrain_run_name(arch_id, device, seed, epochs)
        run_dir = output_root / run_name
        status, summary, summary_path, checkpoint_path = load_retrain_summary(run_dir)
        command = [
            str(python_path),
            "run_retrain.py",
            "--config",
            rel(config_path),
            "--candidate-path",
            candidate_path,
            "--output-dir",
            rel(output_root),
            "--run-name",
            run_name,
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        commands.append(
            {
                "arch_id": arch_id,
                "candidate_path": candidate_path,
                "run_name": run_name,
                "command": command,
                "command_string": command_for_display(command),
            }
        )
        if run_retrain and status != "completed":
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            logs = output_root / "launcher_logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{run_name}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (logs / f"{run_name}.stderr.log").write_text(result.stderr, encoding="utf-8")
            status, summary, summary_path, checkpoint_path = load_retrain_summary(run_dir)
            if result.returncode != 0 and status == "not run":
                status = f"failed_rc_{result.returncode}"

        proxy = candidate.get("search_proxy", {}) if isinstance(candidate, dict) else {}
        rows.append(
            {
                "phase": "v4",
                "arch_id": arch_id,
                "role": candidate.get("role", "") if isinstance(candidate, dict) else "",
                "stage3_op": candidate.get("stage3_op", "") if isinstance(candidate, dict) else "",
                "candidate_path": candidate_path,
                "config": rel(config_path),
                "seed": seed,
                "epochs": epochs,
                "device": device,
                "search_macro_f1": proxy.get("macro_f1"),
                "search_top1": proxy.get("top1"),
                "search_latency_ms": proxy.get("latency_ms"),
                "search_dsp": proxy.get("dsp"),
                "search_bram": proxy.get("bram"),
                "search_lut": proxy.get("lut"),
                "retrain_status": status,
                "retrain_macro_f1": metric_from_retrain_summary(summary, "macro_f1"),
                "retrain_top1": metric_from_retrain_summary(summary, "top1"),
                "retrain_weighted_f1": metric_from_retrain_summary(summary, "weighted_f1"),
                "best_epoch": (summary.get("metrics") or {}).get("best_epoch") if summary else None,
                "checkpoint": rel(checkpoint_path) if checkpoint_path.exists() else "",
                "retrain_summary": rel(summary_path) if summary_path.exists() else "",
                "command": command_for_display(command),
            }
        )

    for arch_id, v3_row in sorted(v3_retrain_index.items()):
        retrain = v3_row.get("retrain150") if isinstance(v3_row.get("retrain150"), dict) else {}
        search_proxy = (
            v3_row.get("search_proxy")
            if isinstance(v3_row.get("search_proxy"), dict)
            else v3_row.get("search_eval10")
            if isinstance(v3_row.get("search_eval10"), dict)
            else {}
        )
        rows.append(
            {
                "phase": "v3_reference",
                "arch_id": arch_id,
                "role": v3_row.get("positioning", v3_row.get("role", "")),
                "stage3_op": "",
                "candidate_path": v3_row.get("candidate_path", ""),
                "config": v3_row.get("config", ""),
                "seed": retrain.get("seed", seed),
                "epochs": retrain.get("epochs", epochs),
                "device": retrain.get("device", device),
                "search_macro_f1": search_proxy.get("macro_f1"),
                "search_top1": search_proxy.get("top1"),
                "search_latency_ms": search_proxy.get("latency_ms"),
                "search_dsp": search_proxy.get("dsp"),
                "search_bram": search_proxy.get("bram"),
                "search_lut": search_proxy.get("lut"),
                "retrain_status": "completed",
                "retrain_macro_f1": retrain.get("macro_f1"),
                "retrain_top1": retrain.get("top1"),
                "retrain_weighted_f1": retrain.get("weighted_f1"),
                "best_epoch": retrain.get("best_epoch"),
                "checkpoint": retrain.get("checkpoint", ""),
                "retrain_summary": retrain.get("summary", ""),
                "command": v3_row.get("command", ""),
            }
        )

    fields = [
        "phase",
        "arch_id",
        "role",
        "stage3_op",
        "candidate_path",
        "config",
        "seed",
        "epochs",
        "device",
        "search_macro_f1",
        "search_top1",
        "search_latency_ms",
        "search_dsp",
        "search_bram",
        "search_lut",
        "retrain_status",
        "retrain_macro_f1",
        "retrain_top1",
        "retrain_weighted_f1",
        "best_epoch",
        "checkpoint",
        "retrain_summary",
        "command",
    ]
    payload = {
        "generated_at": timestamp(),
        "runs_retrain": run_retrain,
        "scope": "Phase0 v4 Top-K retrain150 packaging and optional launcher",
        "rows": rows,
        "commands": commands,
        "acceptance": {
            "required_v4_arches": TOPK_RETRAIN_IDS,
            "completed_v4_arches": [
                row["arch_id"]
                for row in rows
                if row.get("phase") == "v4" and row.get("retrain_status") == "completed"
            ],
        },
    }
    write_json(output_root / "phase0_v4_topk_retrain150_comparison.json", payload)
    write_csv(output_root / "phase0_v4_topk_retrain150_comparison.csv", rows, fields)
    write_retrain_markdown(output_root / "phase0_v4_topk_retrain150_comparison.md", payload)
    write_json(output_root / "phase0_v4_topk_retrain150_commands.json", commands)
    write_powershell(output_root / "run_phase0_v4_topk_retrain150.ps1", [item["command"] for item in commands])
    return payload


def write_retrain_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v4 Top-K Retrain150 Comparison",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- runs_retrain: `{payload['runs_retrain']}`",
        "- boundary: v4 rows remain search-proxy until their retrain_summary and checkpoint exist.",
        "- v3 reference rows reuse existing retrain150 artifacts and are not rerun here.",
        "",
        "| phase | arch | role | stage3 op | search macro_f1 | search top1 | retrain status | retrain macro_f1 | retrain top1 | checkpoint |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row.get('phase')} | `{row.get('arch_id')}` | {row.get('role', '')} | "
            f"`{row.get('stage3_op', '')}` | {fmt(row.get('search_macro_f1'))} | "
            f"{fmt(row.get('search_top1'))} | {row.get('retrain_status')} | "
            f"{fmt(row.get('retrain_macro_f1'))} | {fmt(row.get('retrain_top1'))} | "
            f"`{row.get('checkpoint', '')}` |"
        )
    lines.extend(["", "## Commands", ""])
    for item in payload["commands"]:
        lines.append(f"- `{item['arch_id']}`: `{item['command_string']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_powershell(path: Path, commands: list[list[str]]) -> None:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "Set-Location -LiteralPath $PSScriptRoot",
        "Set-Location -LiteralPath '..\\..'",
        "",
    ]
    for command in commands:
        exe = command[0]
        args = command[1:]
        quoted_args = " ".join("'" + str(arg).replace("'", "''") + "'" for arg in args)
        lines.append(f"& '{exe}' {quoted_args}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_class(row: dict[str, Any], com5_state: dict[str, Any] | None = None) -> str:
    com5 = com5_state or row.get("com5_board", {})
    vivado = row.get("vivado_actual", {})
    if com5.get("status") == "board-claimable":
        return "board-claimable"
    if vivado.get("status") == "route-clean":
        return "route-clean"
    if str(vivado.get("status", "")).startswith("full-route-fail"):
        return "route-fail"
    return "search-only"


def load_com5_measurement_state(
    *,
    candidate: dict[str, Any],
    route_item: dict[str, Any],
) -> dict[str, Any]:
    """Overlay current COM5 measurement evidence without mutating the manifest."""
    state = dict(candidate.get("com5_board", {}) or {})
    vivado = candidate.get("vivado_actual", {}) or {}
    measurement_path = state.get("measurement_json") or route_item.get("measurement_json")
    measurement_file = resolve_repo_path(measurement_path)
    if not measurement_file or not measurement_file.exists():
        return state

    measurement = read_json_if_exists(measurement_file)
    frame = measurement.get("frame") if isinstance(measurement.get("frame"), dict) else {}
    stability = read_json_if_exists(measurement_file.parent / "stability_runs.json")
    bitstream = measurement.get("bitstream") or vivado.get("bitstream")
    observed_sha256 = sha256_file(bitstream)
    expected_sha256 = str(vivado.get("bitstream_sha256") or "")
    sha_match = bool(
        expected_sha256
        and observed_sha256
        and expected_sha256.lower() == observed_sha256.lower()
    )
    run_count = stability.get("run_count") if isinstance(stability, dict) else None
    status_codes = stability.get("status_codes") if isinstance(stability, dict) else None
    stable = stability.get("stable") if isinstance(stability, dict) else None
    all_runs_ok = True
    if status_codes:
        all_runs_ok = all(str(code) == "0" for code in status_codes)

    status_code = frame.get("status_code")
    route_ok = route_clean_for(candidate)
    try:
        enough_runs = run_count is None or int(run_count) >= 5
    except (TypeError, ValueError):
        enough_runs = False
    measured_ok = (
        route_ok
        and str(status_code) == "0"
        and all_runs_ok
        and enough_runs
        and stable is not False
        and sha_match
    )
    status = "board-claimable" if measured_ok else "measured-not-claimable"
    state.update(
        {
            "status": status,
            "status_code": status_code,
            "latency_ms": measurement.get("latency_ms"),
            "cycles": frame.get("cycles"),
            "checksum": frame.get("checksum"),
            "runs": run_count,
            "measurement_json": str(measurement_file),
            "bitstream_sha256_expected": expected_sha256,
            "bitstream_sha256_observed": observed_sha256,
            "bitstream_sha256_match": sha_match,
            "stability_stable": stable,
            "stability_status_codes": status_codes,
        }
    )
    return state


def build_evidence_boundary_rows(
    *,
    manifest: dict[str, Any],
    route_index: dict[str, dict[str, Any]],
    retrain_payload: dict[str, Any],
    v3_retrain_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    retrain_rows = {
        (str(row.get("phase")), str(row.get("arch_id"))): row
        for row in retrain_payload.get("rows", [])
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for candidate in manifest.get("pareto_route_screen_candidates") or []:
        arch_id = str(candidate.get("arch_id") or "")
        proxy = candidate.get("search_proxy", {})
        vivado = candidate.get("vivado_actual", {})
        com5 = load_com5_measurement_state(
            candidate=candidate,
            route_item=route_index.get(arch_id, {}),
        )
        retrain = retrain_rows.get(("v4", arch_id), {})
        klass = evidence_class(candidate, com5)
        rows.append(
            {
                "arch_id": arch_id,
                "phase": "v4",
                "role": candidate.get("role", ""),
                "stage3_op": candidate.get("stage3_op", ""),
                "macro_f1_source": "retrain150 PyTorch validation" if retrain.get("retrain_status") == "completed" else "search proxy eval10",
                "top1_source": "retrain150 PyTorch validation" if retrain.get("retrain_status") == "completed" else "search proxy eval10",
                "search_macro_f1": proxy.get("macro_f1"),
                "search_top1": proxy.get("top1"),
                "retrain_status": retrain.get("retrain_status", "not run"),
                "retrain_macro_f1": retrain.get("retrain_macro_f1"),
                "retrain_top1": retrain.get("retrain_top1"),
                "retrain_summary": retrain.get("retrain_summary", ""),
                "checkpoint": retrain.get("checkpoint", ""),
                "vivado_status": vivado.get("status", "not run"),
                "actual_wns_ns": vivado.get("wns_ns"),
                "actual_dsp": vivado.get("actual_dsp"),
                "actual_lut": vivado.get("actual_lut"),
                "actual_bram": vivado.get("actual_bram"),
                "com5_status": com5.get("status", "not run"),
                "com5_status_code": com5.get("status_code"),
                "com5_latency_ms": com5.get("latency_ms"),
                "com5_cycles": com5.get("cycles"),
                "measurement_runs": com5.get("runs"),
                "board_val_accuracy_status": "not run",
                "power_energy_status": "not measured",
                "claim_boundary": claim_boundary_for(klass),
            }
        )

    for baseline in manifest.get("v3_board_claimable_baseline") or []:
        arch_id = str(baseline.get("arch_id") or "")
        v3_retrain = v3_retrain_index.get(arch_id, {})
        retrain150 = v3_retrain.get("retrain150") if isinstance(v3_retrain.get("retrain150"), dict) else {}
        source = "retrain150 PyTorch validation" if retrain150 else "Phase0 v3 board handoff search/proxy"
        rows.append(
            {
                "arch_id": arch_id,
                "phase": "v3_reference",
                "role": baseline.get("role", ""),
                "stage3_op": "",
                "macro_f1_source": source,
                "top1_source": source,
                "search_macro_f1": baseline.get("macro_f1"),
                "search_top1": baseline.get("top1"),
                "retrain_status": "completed" if retrain150 else "not run",
                "retrain_macro_f1": retrain150.get("macro_f1"),
                "retrain_top1": retrain150.get("top1"),
                "retrain_summary": retrain150.get("summary", ""),
                "checkpoint": retrain150.get("checkpoint", ""),
                "vivado_status": "route-clean",
                "actual_wns_ns": baseline.get("wns_ns"),
                "actual_dsp": baseline.get("actual_dsp"),
                "actual_lut": baseline.get("actual_lut"),
                "actual_bram": baseline.get("actual_bram"),
                "com5_status": baseline.get("claim_status", "board_claimable").replace("_", "-"),
                "com5_status_code": 0,
                "com5_latency_ms": baseline.get("board_latency_ms"),
                "com5_cycles": baseline.get("board_cycles"),
                "measurement_runs": "",
                "board_val_accuracy_status": "not run; validation-set accuracy source is PyTorch retrain when available",
                "power_energy_status": "not measured",
                "claim_boundary": "v3 COM5 is deterministic harness-input latency/output sanity, not full validation-set board accuracy",
            }
        )
    return rows


def claim_boundary_for(klass: str) -> str:
    if klass == "board-claimable":
        return "board-claimable only for listed bitstream SHA256 and COM5 deterministic harness run set"
    if klass == "route-clean":
        return "route-clean only; not board-claimable until COM5 status_code=0 measurement exists"
    if klass == "route-fail":
        return "blocked from COM5 until full-route PASS, WNS>=0, and actual DSP<=700"
    return "search proxy only; no full-route or COM5 claim"


def write_evidence_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase0 v4 Evidence Boundary Table",
        "",
        f"- generated_at: `{timestamp()}`",
        "- boundary: search proxy, retrain150, Vivado route, COM5, board validation accuracy, and measured power/energy are separate evidence layers.",
        "- power/energy remains `not measured` unless an external meter or readable board monitor CSV passes the acceptance audit.",
        "",
        "| phase | arch | role | macro_f1 source | retrain | Vivado | COM5 | board val accuracy | power/energy | claim boundary |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | `{row['arch_id']}` | {row.get('role', '')} | "
            f"{row.get('macro_f1_source', '')} | {row.get('retrain_status', '')} | "
            f"{row.get('vivado_status', '')} | {row.get('com5_status', '')} | "
            f"{row.get('board_val_accuracy_status', '')} | {row.get('power_energy_status', '')} | "
            f"{row.get('claim_boundary', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evidence_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at": timestamp(),
        "rows": rows,
        "boundaries": [
            "Do not treat search latency/DSP/LUT as real board metrics.",
            "Do not treat COM5 deterministic harness measurements as full NKSID validation-set board accuracy.",
            "Do not treat Vivado power estimates or search proxy power_w as measured board power.",
        ],
    }
    write_json(output_dir / "evidence_boundary_table.json", payload)
    write_csv(output_dir / "evidence_boundary_table.csv", rows, EVIDENCE_FIELDS)
    write_evidence_markdown(output_dir / "evidence_boundary_table.md", rows)


def stage3_choices_for_variant(variant: str) -> list[dict[str, Any]]:
    base = [
        {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
        {"op": "skip", "kernel_size": 1, "expand_ratio": 1},
    ]
    if variant in {"denoise_only", "denoise_edge"}:
        base.append({"op": "denoise", "kernel_size": 3, "expand_ratio": 1})
    if variant in {"edge_only", "denoise_edge"}:
        base.append({"op": "edge", "kernel_size": 3, "expand_ratio": 1})
    return base


def op_choices_for_variant(variant: str) -> list[str]:
    ops = ["conv", "mbconv", "skip"]
    if variant in {"denoise_only", "denoise_edge"}:
        ops.append("denoise")
    if variant in {"edge_only", "denoise_edge"}:
        ops.append("edge")
    return ops


def build_ablation_artifacts(
    *,
    base_config_path: Path,
    output_root: Path,
    config_dir: Path,
    python_path: Path,
) -> dict[str, Any]:
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    variants = ["no_sonar", "denoise_only", "edge_only", "denoise_edge"]
    config_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for variant in variants:
        cfg = copy.deepcopy(base_config)
        run_name = f"phase0_v4_sonar_ablation_{variant}_rl300_eval10_seed42"
        cfg.setdefault("project", {})["run_name"] = run_name
        cfg.setdefault("project", {})["output_dir"] = rel(output_root)
        cfg.setdefault("search_space", {})["op_choices"] = op_choices_for_variant(variant)
        stage_choices = cfg.setdefault("search_space", {}).setdefault("stage_block_choices", [])
        if len(stage_choices) < 4:
            raise ValueError("base config must contain four stage_block_choices entries")
        stage_choices[3] = stage3_choices_for_variant(variant)
        cfg.setdefault("notes", {})["ablation_variant"] = variant
        cfg["notes"]["ablation_scope"] = "Only stage3 sonar admission changes across the four variants."
        cfg["notes"]["claim_boundary"] = "Search-space ablation output is search/proxy only until route and COM5 evidence are added."
        config_path = config_dir / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        run_dir = output_root / run_name
        summary = read_json_if_exists(run_dir / "results" / "summary.json")
        pareto = read_json_if_exists(run_dir / "results" / "pareto_selection.json")
        best = read_json_if_exists(run_dir / "results" / "best_candidate.json")
        candidate = best.get("candidate") if isinstance(best.get("candidate"), dict) else {}
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        command = [
            str(python_path),
            "run_rl_search.py",
            "--config",
            rel(config_path),
            "--device",
            "cuda",
        ]
        row = {
            "variant": variant,
            "status": summary.get("status", "not run") if summary else "not run",
            "config": rel(config_path),
            "run_dir": rel(run_dir),
            "stage3_choices": json.dumps(stage_choices[3], ensure_ascii=False),
            "total_evaluated": summary.get("total_evaluated") if summary else "",
            "feasible": summary.get("feasible") if summary else "",
            "pareto_count": len(pareto.get("selected_topk") or []) if pareto else "",
            "best_arch_id": candidate.get("arch_id", ""),
            "macro_f1": metrics.get("macro_f1"),
            "top1": metrics.get("top1"),
            "search_latency_ms": metrics.get("latency_ms"),
            "lut": metrics.get("lut"),
            "dsp": metrics.get("dsp"),
            "bram": metrics.get("bram"),
            "route_status": "not run",
            "command": command_for_display(command),
        }
        rows.append(row)
        commands.append({"variant": variant, "command": command, "command_string": row["command"]})

    payload = {
        "generated_at": timestamp(),
        "runs_search": False,
        "scope": "Four-way Phase0 v4 search-space sonar ablation plan",
        "rows": rows,
        "commands": commands,
        "interpretation_boundaries": [
            "Ablation rows compare search/proxy evidence unless later route and COM5 evidence are attached.",
            "Only stage3 sonar admission changes between variants.",
            "Stage1/stage2, k5, and mixconv remain outside this claim scope.",
        ],
    }
    fields = [
        "variant",
        "status",
        "config",
        "run_dir",
        "stage3_choices",
        "total_evaluated",
        "feasible",
        "pareto_count",
        "best_arch_id",
        "macro_f1",
        "top1",
        "search_latency_ms",
        "lut",
        "dsp",
        "bram",
        "route_status",
        "command",
    ]
    write_json(output_root / "phase0_v4_sonar_ablation_summary.json", payload)
    write_csv(output_root / "phase0_v4_sonar_ablation_summary.csv", rows, fields)
    write_ablation_markdown(output_root / "phase0_v4_sonar_ablation_summary.md", payload)
    write_json(output_root / "phase0_v4_sonar_ablation_commands.json", commands)
    write_powershell(output_root / "run_phase0_v4_sonar_ablation_searches.ps1", [item["command"] for item in commands])
    return payload


def write_ablation_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v4 Sonar Search-Space Ablation",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- boundary: generated configs differ only in stage3 sonar admission.",
        "- comparison layers: search-space contribution, sonar-op candidate contribution, and hardware deployability must be reported separately.",
        "",
        "| variant | status | best arch | macro_f1 | top1 | latency ms | LUT | DSP | BRAM | config |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row['status']} | `{row.get('best_arch_id', '')}` | "
            f"{fmt(row.get('macro_f1'))} | {fmt(row.get('top1'))} | "
            f"{fmt(row.get('search_latency_ms'))} | {fmt(row.get('lut'))} | "
            f"{fmt(row.get('dsp'))} | {fmt(row.get('bram'))} | `{row.get('config')}` |"
        )
    lines.extend(["", "## Commands", ""])
    for item in payload["commands"]:
        lines.append(f"- `{item['variant']}`: `{item['command_string']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def route_clean_for(candidate: dict[str, Any]) -> bool:
    vivado = candidate.get("vivado_actual", {})
    return (
        vivado.get("status") == "route-clean"
        and (vivado.get("wns_ns") is not None and float(vivado.get("wns_ns")) >= 0.0)
        and (vivado.get("actual_dsp") is not None and float(vivado.get("actual_dsp")) <= 700.0)
    )


def build_hardware_artifacts(
    *,
    manifest: dict[str, Any],
    route_index: dict[str, dict[str, Any]],
    output_dir: Path,
    run_hardware: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    candidates = {
        str(row.get("arch_id")): row
        for row in manifest.get("pareto_route_screen_candidates") or []
        if row.get("arch_id")
    }
    ordered = V4_COM5_INITIAL_IDS + V4_COM5_RECHECK_IDS + ["rl_arch_116"]
    for arch_id in ordered:
        candidate = candidates.get(arch_id, {})
        route_item = route_index.get(arch_id, {})
        vivado = candidate.get("vivado_actual", {}) if candidate else {}
        com5 = (
            load_com5_measurement_state(candidate=candidate, route_item=route_item)
            if candidate
            else {}
        )
        if arch_id in V4_COM5_INITIAL_IDS:
            action = "initial_com5_5_runs"
        elif arch_id in V4_COM5_RECHECK_IDS:
            action = "stability_recheck_5_runs"
        else:
            action = "blocked"
        if not route_clean_for(candidate):
            action = "blocked_until_full_route_pass"

        measurement_command = list(route_item.get("measurement_command") or [])
        if measurement_command and "--runs" not in measurement_command:
            measurement_command.extend(["--runs", "5"])
        if action == "stability_recheck_5_runs" and measurement_command and "--force" not in measurement_command:
            measurement_command.append("--force")
        command_string = command_for_display(measurement_command) if measurement_command else ""
        commands.append({"arch_id": arch_id, "action": action, "command": measurement_command, "command_string": command_string})
        rows.append(
            {
                "arch_id": arch_id,
                "action": action,
                "stage3_op": candidate.get("stage3_op", ""),
                "vivado_status": vivado.get("status", "not run"),
                "wns_ns": vivado.get("wns_ns"),
                "actual_dsp": vivado.get("actual_dsp"),
                "bitstream_sha256": vivado.get("bitstream_sha256"),
                "current_com5_status": com5.get("status", "not run"),
                "com5_status_code": com5.get("status_code"),
                "current_com5_latency_ms": com5.get("latency_ms"),
                "current_runs": com5.get("runs"),
                "measurement_json": com5.get("measurement_json") or route_item.get("measurement_json", ""),
                "bitstream_sha256_match": com5.get("bitstream_sha256_match"),
                "stability_stable": com5.get("stability_stable"),
                "claim_boundary": claim_boundary_for(evidence_class(candidate, com5) if candidate else "search-only"),
                "command": command_string,
            }
        )
        if run_hardware and action in {"initial_com5_5_runs", "stability_recheck_5_runs"} and measurement_command:
            result = subprocess.run(
                measurement_command,
                cwd=str(REPO_ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            logs = output_dir / "hardware_launcher_logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{arch_id}.stdout.log").write_text(result.stdout, encoding="utf-8")
            (logs / f"{arch_id}.stderr.log").write_text(result.stderr, encoding="utf-8")
            rows[-1]["launcher_returncode"] = result.returncode

    payload = {
        "generated_at": timestamp(),
        "runs_hardware": run_hardware,
        "rows": rows,
        "commands": commands,
        "acceptance": {
            "claimable_requires": [
                "full-route PASS",
                "WNS >= 0",
                "actual DSP <= 700",
                "COM5 status_code = 0",
                "bitstream SHA256 match",
            ],
            "board_validation_accuracy": "not run; deterministic COM5 harness input is not full NKSID validation-set accuracy",
        },
    }
    fields = [
        "arch_id",
        "action",
        "stage3_op",
        "vivado_status",
        "wns_ns",
        "actual_dsp",
        "bitstream_sha256",
        "current_com5_status",
        "com5_status_code",
        "current_com5_latency_ms",
        "current_runs",
        "measurement_json",
        "bitstream_sha256_match",
        "stability_stable",
        "claim_boundary",
        "command",
        "launcher_returncode",
    ]
    write_json(output_dir / "hardware_closure_plan.json", payload)
    write_csv(output_dir / "hardware_closure_plan.csv", rows, fields)
    write_hardware_markdown(output_dir / "hardware_closure_plan.md", payload)
    write_powershell(
        output_dir / "run_phase0_v4_hardware_closure_com5.ps1",
        [item["command"] for item in commands if item.get("command") and item.get("action") != "blocked_until_full_route_pass"],
    )
    return payload


def write_hardware_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v4 Hardware Closure Plan",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- runs_hardware: `{payload['runs_hardware']}`",
        "- boundary: COM5 is UART board latency/output sanity evidence, not measured power and not full validation-set board accuracy.",
        "",
        "| arch | action | stage3 op | Vivado | WNS ns | DSP | current COM5 | runs | latency ms | SHA match | command |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['arch_id']}` | {row['action']} | `{row.get('stage3_op', '')}` | "
            f"{row.get('vivado_status', '')} | {fmt(row.get('wns_ns'))} | {fmt(row.get('actual_dsp'))} | "
            f"{row.get('current_com5_status', '')} | {fmt(row.get('current_runs'))} | "
            f"{fmt(row.get('current_com5_latency_ms'))} | {row.get('bitstream_sha256_match', '')} | "
            f"`{row.get('command', '')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_power_guard_artifacts(output_dir: Path) -> dict[str, Any]:
    payload = {
        "generated_at": timestamp(),
        "status": "not measured",
        "measurement_required_before_claim": True,
        "accepted_sources": [
            "external power-meter CSV",
            "confirmed readable board power-monitor CSV",
        ],
        "required_columns_or_fields": [
            "timestamp_s with power_w, or timestamp_s with voltage_v and current_a",
            "active_inference_count",
            "bitstream_sha256",
            "contains_programming=false",
            "same_com5_input=true",
        ],
        "formulas": {
            "idle_power_mean_w": "time-weighted mean(P_idle)",
            "dynamic_power_mean_w": "active_power_mean_w - idle_power_mean_w",
            "dynamic_energy_per_inference_mj": "1000 * integral(P_active(t) - idle_power_mean_w) dt / active_inference_count",
        },
        "claim_boundary": "Vivado/search proxy power_w is not measured board power.",
    }
    write_json(output_dir / "power_energy_measurement_guard.json", payload)
    lines = [
        "# Power/Energy Measurement Guard",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        "- status: `not measured`",
        "- Vivado/search proxy `power_w` must not be reported as measured board power.",
        "- Measured claims require external meter or confirmed board-monitor CSV plus audit PASS.",
        "",
        "| required item | status |",
        "|---|---|",
    ]
    for item in payload["required_columns_or_fields"]:
        lines.append(f"| {item} | required before measured claim |")
    (output_dir / "power_energy_measurement_guard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-manifest", default=str(DEFAULT_V4_MANIFEST))
    parser.add_argument("--v4-output-dir", default=str(DEFAULT_V4_OUTPUT_DIR))
    parser.add_argument("--v4-config", default=str(DEFAULT_V4_CONFIG))
    parser.add_argument("--route-plan", default=str(DEFAULT_ROUTE_PLAN))
    parser.add_argument("--v3-retrain-comparison", default=str(DEFAULT_V3_RETRAIN_COMPARISON))
    parser.add_argument("--retrain-root", default=str(DEFAULT_RETRAIN_ROOT))
    parser.add_argument("--ablation-root", default=str(DEFAULT_ABLATION_ROOT))
    parser.add_argument("--ablation-config-dir", default=str(DEFAULT_ABLATION_CONFIG_DIR))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-retrain", action="store_true")
    parser.add_argument("--run-hardware", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.v4_manifest).expanduser().resolve()
    output_dir = Path(args.v4_output_dir).expanduser().resolve()
    config_path = Path(args.v4_config).expanduser().resolve()
    route_plan = Path(args.route_plan).expanduser().resolve()
    retrain_root = Path(args.retrain_root).expanduser().resolve()
    ablation_root = Path(args.ablation_root).expanduser().resolve()
    ablation_config_dir = Path(args.ablation_config_dir).expanduser().resolve()
    python_path = Path(args.python).expanduser()

    manifest = read_json(manifest_path)
    route_index = load_route_plan_index(route_plan)
    v3_retrain_index = load_v3_retrain_index(Path(args.v3_retrain_comparison).expanduser().resolve())

    retrain_payload = build_retrain_artifacts(
        manifest=manifest,
        route_index=route_index,
        v3_retrain_index=v3_retrain_index,
        output_root=retrain_root,
        config_path=config_path,
        python_path=python_path,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        run_retrain=args.run_retrain,
    )
    evidence_rows = build_evidence_boundary_rows(
        manifest=manifest,
        route_index=route_index,
        retrain_payload=retrain_payload,
        v3_retrain_index=v3_retrain_index,
    )
    write_evidence_artifacts(output_dir, evidence_rows)
    ablation_payload = build_ablation_artifacts(
        base_config_path=config_path,
        output_root=ablation_root,
        config_dir=ablation_config_dir,
        python_path=python_path,
    )
    hardware_payload = build_hardware_artifacts(
        manifest=manifest,
        route_index=route_index,
        output_dir=output_dir,
        run_hardware=args.run_hardware,
    )
    power_guard = write_power_guard_artifacts(output_dir)
    result = {
        "status": "PASS",
        "runs_retrain": args.run_retrain,
        "runs_hardware": args.run_hardware,
        "evidence_boundary_table": rel(output_dir / "evidence_boundary_table.md"),
        "retrain_comparison": rel(retrain_root / "phase0_v4_topk_retrain150_comparison.md"),
        "ablation_summary": rel(ablation_root / "phase0_v4_sonar_ablation_summary.md"),
        "hardware_closure_plan": rel(output_dir / "hardware_closure_plan.md"),
        "power_guard": rel(output_dir / "power_energy_measurement_guard.md"),
        "v4_retrain_completed": retrain_payload["acceptance"]["completed_v4_arches"],
        "ablation_variants": [row["variant"] for row in ablation_payload["rows"]],
        "hardware_actions": {row["arch_id"]: row["action"] for row in hardware_payload["rows"]},
        "power_status": power_guard["status"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
