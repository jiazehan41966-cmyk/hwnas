#!/usr/bin/env python
"""Compare loose/medium/tight timing-profile NAS search runs.

The script reads completed run directories produced by ``run_rl_search.py`` and
writes a small comparison bundle with search time, best metrics, latency, and
resource usage. Missing runs are kept in the output as ``missing`` rows so the
same command can be used before, during, and after the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "timing_profile_comparison_current84_arch84"
DEFAULT_RUN_DIRS = {
    "tight": REPO_ROOT
    / "results"
    / "nas_board_lut_timing_tight_current84_arch84_nksid_full_rl300_eval10_cuda_v1",
    "medium": REPO_ROOT
    / "results"
    / "nas_board_lut_timing_medium_current84_arch84_nksid_full_rl300_eval10_cuda_v1",
    "loose": REPO_ROOT
    / "results"
    / "nas_board_lut_timing_loose_current84_arch84_nksid_full_rl300_eval10_cuda_v1",
}


CSV_FIELDS = (
    "profile",
    "status",
    "run_dir",
    "search_seconds",
    "total_evaluated",
    "feasible",
    "infeasible",
    "arch_id",
    "macro_f1",
    "macro_f1_delta_vs_tight",
    "macro_f1_loss_vs_best",
    "top1",
    "latency_ms",
    "latency_delta_vs_tight_ms",
    "lut",
    "dsp",
    "bram",
    "power_w",
    "lut_true_misses",
    "lut_deferred_hits",
    "strict_formal_lut",
    "strict_lut_label",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _run_seconds(run_info: dict[str, Any]) -> float | None:
    created = _parse_time(run_info.get("created_at"))
    finished = _parse_time(run_info.get("finished_at"))
    if created is None or finished is None:
        return None
    return max(0.0, (finished - created).total_seconds())


def _candidate_from_summary(summary: dict[str, Any], best_candidate: dict[str, Any]) -> dict[str, Any]:
    if isinstance(summary.get("best_candidate"), dict):
        return summary["best_candidate"]
    if isinstance(best_candidate.get("candidate"), dict):
        return best_candidate["candidate"]
    return {}


def _load_profile(profile: str, run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_info = _read_json(run_dir / "run_info.json")
    summary = _read_json(run_dir / "results" / "summary.json")
    best_candidate = _read_json(run_dir / "results" / "best_candidate.json")
    lut_stats = _read_json(run_dir / "results" / "lut_stats.json")
    candidate = _candidate_from_summary(summary, best_candidate)
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}

    status = summary.get("status") or run_info.get("status") or "missing"
    if not run_dir.is_dir():
        status = "missing"

    return {
        "profile": profile,
        "status": status,
        "run_dir": str(run_dir),
        "search_seconds": _run_seconds(run_info),
        "total_evaluated": summary.get("total_evaluated"),
        "feasible": summary.get("feasible"),
        "infeasible": summary.get("infeasible"),
        "arch_id": candidate.get("arch_id"),
        "macro_f1": metrics.get("macro_f1"),
        "top1": metrics.get("top1"),
        "latency_ms": metrics.get("latency_ms"),
        "lut": metrics.get("lut"),
        "dsp": metrics.get("dsp"),
        "bram": metrics.get("bram"),
        "power_w": metrics.get("power_w"),
        "lut_true_misses": lut_stats.get("true_misses"),
        "lut_deferred_hits": lut_stats.get("deferred_hits"),
        "strict_formal_lut": lut_stats.get("strict_formal_lut"),
        "strict_lut_label": lut_stats.get("strict_lut_label"),
    }


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _annotate_deltas(rows: list[dict[str, Any]]) -> None:
    tight = next((row for row in rows if row["profile"] == "tight"), None)
    tight_macro = _to_float(tight.get("macro_f1")) if tight else None
    tight_latency = _to_float(tight.get("latency_ms")) if tight else None
    macros = [_to_float(row.get("macro_f1")) for row in rows]
    best_macro = max([value for value in macros if value is not None], default=None)

    for row in rows:
        macro = _to_float(row.get("macro_f1"))
        latency = _to_float(row.get("latency_ms"))
        row["macro_f1_delta_vs_tight"] = (
            None if macro is None or tight_macro is None else macro - tight_macro
        )
        row["macro_f1_loss_vs_best"] = (
            None if macro is None or best_macro is None else best_macro - macro
        )
        row["latency_delta_vs_tight_ms"] = (
            None if latency is None or tight_latency is None else latency - tight_latency
        )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Timing Profile NAS Comparison",
        "",
        "This table compares completed loose/medium/tight NAS runs. Latency is the",
        "LUT-estimated candidate latency for the timing profile used by that run,",
        "not a new full-network COM5 measurement.",
        "",
        "Only the tight 200 MHz profile is currently backed by the official strict",
        "board LUT. Medium and loose profiles are timing-only derived comparisons",
        "until their harnesses are rebuilt with the relaxed clock constraints and",
        "remeasured over UART.",
        "",
        "| profile | status | search s | macro_f1 | loss vs best | top1 | latency ms | latency delta vs tight | LUT | DSP | BRAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {status} | {search_seconds} | {macro_f1} | "
            "{macro_f1_loss_vs_best} | {top1} | {latency_ms} | "
            "{latency_delta_vs_tight_ms} | {lut} | {dsp} | {bram} |".format(
                **{key: _csv_value(row.get(key)) for key in CSV_FIELDS}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tight-run-dir", type=Path, default=DEFAULT_RUN_DIRS["tight"])
    parser.add_argument("--medium-run-dir", type=Path, default=DEFAULT_RUN_DIRS["medium"])
    parser.add_argument("--loose-run-dir", type=Path, default=DEFAULT_RUN_DIRS["loose"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dirs = {
        "tight": args.tight_run_dir,
        "medium": args.medium_run_dir,
        "loose": args.loose_run_dir,
    }
    rows = [_load_profile(profile, run_dirs[profile]) for profile in ("tight", "medium", "loose")]
    _annotate_deltas(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "rows": rows,
        "notes": [
            "macro_f1_loss_vs_best is computed from completed rows only.",
            "latency_delta_vs_tight_ms is computed against the tight row when available.",
            "LUT latency is an estimator value; final e2e latency still requires COM5 measurement.",
        ],
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "comparison.csv", rows)
    _write_markdown(args.output_dir / "comparison.md", rows)
    print(f"Wrote timing profile comparison to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
