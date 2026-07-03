#!/usr/bin/env python3
"""Generate a paper-ready comparison table from search run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METHOD_ORDER = {
    "random": 0,
    "rl": 1,
    "proxyless": 2,
}

DISPLAY_NAME = {
    "random": "Random",
    "rl": "RL",
    "proxyless": "ProxylessNAS",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_run_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_info.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _load_summary(run_dir: Path, run_info: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_dir / "results" / "summary.json"
    if summary_path.exists():
        payload = _read_json(summary_path)
        if isinstance(payload, dict):
            return payload

    search_state_path = run_dir / "checkpoints" / "search_state.json"
    if search_state_path.exists():
        state = _read_json(search_state_path)
        if isinstance(state, dict):
            return {
                "finished_at": None,
                "status": run_info.get("status", "running"),
                "total_evaluated": state.get("total_evaluated", 0),
                "feasible": state.get("feasible", 0),
                "infeasible": state.get("infeasible", 0),
                "best_candidate": state.get("best_candidate"),
                "extra": {
                    "search_method": run_info.get("search_method", "unknown"),
                    "selection_metric": "macro_f1",
                    "partial": True,
                },
            }
    return {}


def _load_candidate_records(run_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = run_dir / "results" / "candidates.jsonl"
    if jsonl_path.exists():
        rows: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    json_path = run_dir / "results" / "candidates.json"
    if not json_path.exists():
        return []

    payload = _read_json(json_path)
    if not isinstance(payload, list):
        return []

    rows = []
    for candidate in payload:
        if isinstance(candidate, dict):
            rows.append(
                {
                    "feasible": candidate.get("metrics", {}).get("macro_f1") is not None,
                    "candidate": candidate,
                    "history": None,
                    "extra": {},
                }
            )
    return rows


def _metric(candidate: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(candidate, dict):
        return None
    metrics = candidate.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_candidate(summary: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = summary.get("best_candidate")
    if isinstance(best, dict):
        return best

    feasible_candidates = [
        record.get("candidate")
        for record in records
        if record.get("feasible") and isinstance(record.get("candidate"), dict)
    ]
    feasible_candidates = [
        candidate for candidate in feasible_candidates if _metric(candidate, "macro_f1") is not None
    ]
    if not feasible_candidates:
        return None
    return max(
        feasible_candidates,
        key=lambda candidate: float(_metric(candidate, "macro_f1") or float("-inf")),
    )


def _macro_f1_distribution(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    values = [
        float(candidate["metrics"]["macro_f1"])
        for candidate in (
            record.get("candidate")
            for record in records
            if record.get("feasible") and isinstance(record.get("candidate"), dict)
        )
        if isinstance(candidate, dict)
        and isinstance(candidate.get("metrics"), dict)
        and candidate["metrics"].get("macro_f1") is not None
    ]
    if not values:
        return None, None
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), pstdev(values)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    run_info = _load_run_info(run_dir)
    summary = _load_summary(run_dir, run_info)
    records = _load_candidate_records(run_dir)
    best_candidate = _best_candidate(summary, records)
    total_evaluated = int(summary.get("total_evaluated", len(records) or 0))
    feasible = int(summary.get("feasible", 0))
    infeasible = int(summary.get("infeasible", max(0, total_evaluated - feasible)))
    feasible_ratio = (100.0 * feasible / total_evaluated) if total_evaluated else 0.0
    mean_macro_f1, std_macro_f1 = _macro_f1_distribution(records)

    extra = summary.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    method = str(extra.get("search_method") or run_info.get("search_method") or "unknown").lower()
    selection_metric = str(extra.get("selection_metric") or "macro_f1")

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "status": str(summary.get("status") or run_info.get("status") or "unknown"),
        "method": method,
        "selection_metric": selection_metric,
        "total_evaluated": total_evaluated,
        "feasible": feasible,
        "infeasible": infeasible,
        "feasible_ratio": feasible_ratio,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "best_arch_id": None if not isinstance(best_candidate, dict) else best_candidate.get("arch_id"),
        "best_macro_f1": _metric(best_candidate, "macro_f1"),
        "best_top1": _metric(best_candidate, "top1"),
        "best_top5": _metric(best_candidate, "top5"),
        "best_latency_ms": _metric(best_candidate, "latency_ms"),
        "best_dsp": _metric(best_candidate, "dsp"),
        "best_bram": _metric(best_candidate, "bram"),
        "best_lut": _metric(best_candidate, "lut"),
        "best_power_w": _metric(best_candidate, "power_w"),
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (METHOD_ORDER.get(row["method"], 99), row["run_name"]))


def _winners(rows: list[dict[str, Any]], key: str, higher_is_better: bool) -> set[int]:
    values = [
        (index, row.get(key))
        for index, row in enumerate(rows)
        if row.get(key) is not None and not (isinstance(row.get(key), float) and math.isnan(row[key]))
    ]
    if not values:
        return set()
    target = max(value for _index, value in values) if higher_is_better else min(value for _index, value in values)
    return {
        index
        for index, value in values
        if abs(float(value) - float(target)) < 1e-12
    }


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


def _fmt_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "--"
    std_value = 0.0 if std_value is None else std_value
    return f"{mean_value * 100:.2f} $\\pm$ {std_value * 100:.2f}"


def _fmt_float(value: float | None, digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _fmt_int_like(value: float | None) -> str:
    if value is None:
        return "--"
    return str(int(round(value)))


def _method_display_name(method: str) -> str:
    return DISPLAY_NAME.get(method, method)


def _latex_escape(value: str) -> str:
    escaped = value.replace("\\", "\\textbackslash{}")
    for old, new in (
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
    ):
        escaped = escaped.replace(old, new)
    return escaped


def _latex_label(value: str) -> str:
    sanitized = []
    for char in value:
        if char.isalnum() or char in {":", "-", "_"}:
            sanitized.append(char)
        else:
            sanitized.append("-")
    return "".join(sanitized).strip("-") or "tab-search-comparison"


def _maybe_bold(index: int, winners: set[int], text: str) -> str:
    return f"\\textbf{{{text}}}" if index in winners and text != "--" else text


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    headers = [
        "Method",
        "Status",
        "Evals",
        "Feasible",
        "Best Macro-F1 (%)",
        "Mean Macro-F1 (%)",
        "Top-1 (%)",
        "Top-5 (%)",
        "Latency (ms)",
        "DSP",
        "BRAM",
        "LUT",
        "Power (W)",
    ]
    lines = [
        "# Search Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _method_display_name(row["method"]),
                    row["status"],
                    str(row["total_evaluated"]),
                    f"{row['feasible']}/{row['total_evaluated']} ({row['feasible_ratio']:.1f}%)",
                    _fmt_percent(row["best_macro_f1"]),
                    "--" if row["mean_macro_f1"] is None else f"{row['mean_macro_f1'] * 100:.2f} +/- {(row['std_macro_f1'] or 0.0) * 100:.2f}",
                    _fmt_percent(row["best_top1"]),
                    _fmt_percent(row["best_top5"]),
                    _fmt_float(row["best_latency_ms"], 3),
                    _fmt_int_like(row["best_dsp"]),
                    _fmt_int_like(row["best_bram"]),
                    _fmt_int_like(row["best_lut"]),
                    _fmt_float(row["best_power_w"], 2),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict[str, Any]], output_path: Path, caption: str, label: str) -> None:
    feasible_winners = _winners(rows, "feasible_ratio", higher_is_better=True)
    best_macro_winners = _winners(rows, "best_macro_f1", higher_is_better=True)
    mean_macro_winners = _winners(rows, "mean_macro_f1", higher_is_better=True)
    top1_winners = _winners(rows, "best_top1", higher_is_better=True)
    top5_winners = _winners(rows, "best_top5", higher_is_better=True)
    latency_winners = _winners(rows, "best_latency_ms", higher_is_better=False)
    dsp_winners = _winners(rows, "best_dsp", higher_is_better=False)
    bram_winners = _winners(rows, "best_bram", higher_is_better=False)
    lut_winners = _winners(rows, "best_lut", higher_is_better=False)
    power_winners = _winners(rows, "best_power_w", higher_is_better=False)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccccccc}",
        "\\hline",
        "Method & Evals & Feasible & Best Macro-F1 (\\%) & Mean Macro-F1 (\\%) & Top-1 (\\%) & Top-5 (\\%) & Latency (ms) & DSP & BRAM & LUT & Power (W) \\\\",
        "\\hline",
    ]

    for index, row in enumerate(rows):
        lines.append(
            " & ".join(
                [
                    _latex_escape(_method_display_name(row["method"])),
                    str(row["total_evaluated"]),
                    _maybe_bold(
                        index,
                        feasible_winners,
                        f"{row['feasible']}/{row['total_evaluated']} ({row['feasible_ratio']:.1f}\\%)",
                    ),
                    _maybe_bold(index, best_macro_winners, _fmt_percent(row["best_macro_f1"])),
                    _maybe_bold(index, mean_macro_winners, _fmt_mean_std(row["mean_macro_f1"], row["std_macro_f1"])),
                    _maybe_bold(index, top1_winners, _fmt_percent(row["best_top1"])),
                    _maybe_bold(index, top5_winners, _fmt_percent(row["best_top5"])),
                    _maybe_bold(index, latency_winners, _fmt_float(row["best_latency_ms"], 3)),
                    _maybe_bold(index, dsp_winners, _fmt_int_like(row["best_dsp"])),
                    _maybe_bold(index, bram_winners, _fmt_int_like(row["best_bram"])),
                    _maybe_bold(index, lut_winners, _fmt_int_like(row["best_lut"])),
                    _maybe_bold(index, power_winners, _fmt_float(row["best_power_w"], 2)),
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}%",
            "}",
            f"\\label{{{_latex_label(label)}}}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runs": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a paper-ready search comparison table")
    parser.add_argument("run_dirs", nargs="+", help="Run directories under results/")
    parser.add_argument("--output-dir", default="results/paper_search_comparison", help="Directory for generated artifacts")
    parser.add_argument("--basename", default="paper_search_comparison", help="Basename for output files")
    parser.add_argument(
        "--caption",
        default="Comparison of search strategies under the same evaluation budget on NKSID with FPGA constraints.",
        help="LaTeX table caption",
    )
    parser.add_argument("--label", default="tab:search_strategy_comparison", help="LaTeX label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _sort_rows([summarize_run(Path(run_dir)) for run_dir in args.run_dirs])
    if not rows:
        raise SystemExit("No valid run directories were provided.")

    base = args.basename
    write_json(rows, output_dir / f"{base}.json")
    write_markdown(rows, output_dir / f"{base}.md")
    write_latex(rows, output_dir / f"{base}.tex", caption=args.caption, label=args.label)
    print(f"[OK] Wrote comparison artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
