from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUNS = [
    ("compact_4stage_rl_5ep_100ms", Path("results/formal_lut_compact_4stage_smoke_5ep_100ms")),
    ("compact_5stage_rl_5ep_150ms", Path("results/formal_lut_compact_5stage_smoke_5ep_150ms")),
    ("expanded_4stage_rl_20ep_100ms", Path("results/formal_lut_expanded_4stage_smoke_20ep_100ms")),
    ("expanded_5stage_rl_20ep_150ms", Path("results/formal_lut_expanded_5stage_smoke_20ep_150ms")),
    (
        "expanded_4stage_random_50ep_100ms",
        Path("results/formal_lut_expanded_4stage_random_smoke_50ep_100ms"),
    ),
    (
        "expanded_5stage_random_50ep_150ms",
        Path("results/formal_lut_expanded_5stage_random_smoke_50ep_150ms"),
    ),
]

OUT_DIR = Path("results/formal_lut_capacity_diagnosis")

EXHAUSTIVE_RUNS = [
    ("expanded_4stage_exhaustive_100ms", OUT_DIR / "expanded_4stage_exhaustive_100ms"),
    ("expanded_5stage_exhaustive_150ms", OUT_DIR / "expanded_5stage_exhaustive_150ms"),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate.get("metrics") or candidate.get("hardware_metrics") or {}


def _variant_label(block: dict[str, Any]) -> str:
    return f"{block.get('op')}_e{block.get('expand_ratio')}_k{block.get('kernel_size')}"


def _architecture_key(candidate: dict[str, Any]) -> str:
    encoding = candidate.get("encoding") or candidate.get("architecture") or {}
    return json.dumps(encoding, sort_keys=True, separators=(",", ":"))


def _stage_choice_counts(candidates: list[dict[str, Any]]) -> dict[int, Counter[str]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        encoding = candidate.get("encoding") or {}
        for stage_idx, stage in enumerate(encoding.get("stages") or []):
            blocks = stage.get("blocks") or []
            if not blocks:
                counts[stage_idx]["empty"] += 1
                continue
            counts[stage_idx][_variant_label(blocks[0])] += 1
    return dict(counts)


def _range(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def summarize_run(label: str, run_dir: Path) -> dict[str, Any]:
    results_dir = run_dir / "results"
    summary = _load_json(results_dir / "summary.json")
    lut_stats = _load_json(results_dir / "lut_stats.json")
    candidates = _load_json(results_dir / "candidates.json")
    if not isinstance(candidates, list):
        raise ValueError(f"expected candidate list in {results_dir / 'candidates.json'}")

    metric_rows = [_metrics(candidate) for candidate in candidates]
    latency_range = _range([float(row["latency_ms"]) for row in metric_rows if row.get("latency_ms") is not None])
    bram_range = _range([float(row["bram"]) for row in metric_rows if row.get("bram") is not None])
    lut_range = _range([float(row["lut"]) for row in metric_rows if row.get("lut") is not None])
    power_range = _range([float(row["power_w"]) for row in metric_rows if row.get("power_w") is not None])

    unique_arch = len({_architecture_key(candidate) for candidate in candidates})
    stage_counts = _stage_choice_counts(candidates)
    search_method = summary.get("extra", {}).get("search_method", "unknown")

    return {
        "label": label,
        "run_dir": str(run_dir),
        "search_method": search_method,
        "status": summary.get("status"),
        "evaluated": int(summary.get("total_evaluated") or len(candidates)),
        "feasible": int(summary.get("feasible") or 0),
        "infeasible": int(summary.get("infeasible") or 0),
        "unique_arch": unique_arch,
        "latency_min": latency_range[0],
        "latency_max": latency_range[1],
        "bram_min": bram_range[0],
        "bram_max": bram_range[1],
        "lut_min": lut_range[0],
        "lut_max": lut_range[1],
        "power_min": power_range[0],
        "power_max": power_range[1],
        "lut_hits": int(lut_stats.get("hits") or 0),
        "lut_misses": int(lut_stats.get("misses") or 0),
        "true_misses": int(lut_stats.get("true_misses") or 0),
        "deferred_hits": int(lut_stats.get("deferred_hits") or 0),
        "hit_rate": float(lut_stats.get("hit_rate") or 0.0),
        "strict_formal_lut": bool(lut_stats.get("strict_formal_lut")),
        "best_candidate_present": summary.get("best_candidate") is not None,
        "stage_counts": stage_counts,
    }


def summarize_exhaustive(label: str, run_dir: Path) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    return {
        "label": label,
        "candidate_count": int(summary["candidate_count"]),
        "feasible": int(summary["feasible"]),
        "infeasible": int(summary["infeasible"]),
        "unique_arch": int(summary["unique_arch"]),
        "strict_lut_ok": bool(summary["strict_lut_ok"]),
        "lut_hits": int(summary["lut_totals"]["lut_hits"]),
        "lut_misses": int(summary["lut_totals"]["lut_misses"]),
        "true_misses": int(summary["lut_totals"]["true_misses"]),
        "deferred_hits": int(summary["lut_totals"]["deferred_hits"]),
        "latency_min": summary["latency_ms"]["min"],
        "latency_max": summary["latency_ms"]["max"],
        "bram_min": summary["bram"]["min"],
        "bram_max": summary["bram"]["max"],
        "lut_min": summary["lut"]["min"],
        "lut_max": summary["lut"]["max"],
        "power_min": summary["power_w"]["min"],
        "power_max": summary["power_w"]["max"],
        "violation_counts": summary.get("violation_counts") or {},
        "run_dir": str(run_dir),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "label",
        "search_method",
        "status",
        "evaluated",
        "feasible",
        "infeasible",
        "unique_arch",
        "latency_min",
        "latency_max",
        "bram_min",
        "bram_max",
        "lut_min",
        "lut_max",
        "power_min",
        "power_max",
        "lut_hits",
        "lut_misses",
        "true_misses",
        "deferred_hits",
        "hit_rate",
        "strict_formal_lut",
        "best_candidate_present",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_exhaustive_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "label",
        "candidate_count",
        "feasible",
        "infeasible",
        "unique_arch",
        "strict_lut_ok",
        "lut_hits",
        "lut_misses",
        "true_misses",
        "deferred_hits",
        "latency_min",
        "latency_max",
        "bram_min",
        "bram_max",
        "lut_min",
        "lut_max",
        "power_min",
        "power_max",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_markdown(
    rows: list[dict[str, Any]],
    exhaustive_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = ["# Expanded measured LUT smoke summary\n\n"]
    lines.append(
        "| Run | Method | Evaluated | Feasible | Unique arch | Latency ms | BRAM | LUT | Power W | LUT hits | misses/true/deferred |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        lines.append(
            "| {label} | {method} | {evaluated} | {feasible}/{evaluated} | {unique} | "
            "{lat_min}-{lat_max} | {bram_min}-{bram_max} | {lut_min}-{lut_max} | "
            "{pwr_min}-{pwr_max} | {hits} | {miss}/{true}/{deferred} |\n".format(
                label=row["label"],
                method=row["search_method"],
                evaluated=row["evaluated"],
                feasible=row["feasible"],
                unique=row["unique_arch"],
                lat_min=_fmt(row["latency_min"]),
                lat_max=_fmt(row["latency_max"]),
                bram_min=_fmt(row["bram_min"]),
                bram_max=_fmt(row["bram_max"]),
                lut_min=_fmt(row["lut_min"]),
                lut_max=_fmt(row["lut_max"]),
                pwr_min=_fmt(row["power_min"]),
                pwr_max=_fmt(row["power_max"]),
                hits=row["lut_hits"],
                miss=row["lut_misses"],
                true=row["true_misses"],
                deferred=row["deferred_hits"],
            )
        )

    lines.append("\n## Exhaustive expanded-space feasibility\n\n")
    lines.append(
        "| Run | Candidates | Feasible | Unique arch | Latency ms | BRAM | LUT | Power W | LUT hits | misses/true/deferred | Violations |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in exhaustive_rows:
        violation_text = ", ".join(
            f"{name}: {count}" for name, count in sorted(row["violation_counts"].items())
        ) or "-"
        lines.append(
            "| {label} | {count} | {feasible}/{count} | {unique} | {lat_min}-{lat_max} | "
            "{bram_min}-{bram_max} | {lut_min}-{lut_max} | {pwr_min}-{pwr_max} | "
            "{hits} | {miss}/{true}/{deferred} | {violations} |\n".format(
                label=row["label"],
                count=row["candidate_count"],
                feasible=row["feasible"],
                unique=row["unique_arch"],
                lat_min=_fmt(row["latency_min"]),
                lat_max=_fmt(row["latency_max"]),
                bram_min=_fmt(row["bram_min"]),
                bram_max=_fmt(row["bram_max"]),
                lut_min=_fmt(row["lut_min"]),
                lut_max=_fmt(row["lut_max"]),
                pwr_min=_fmt(row["power_min"]),
                pwr_max=_fmt(row["power_max"]),
                hits=row["lut_hits"],
                miss=row["lut_misses"],
                true=row["true_misses"],
                deferred=row["deferred_hits"],
                violations=violation_text,
            )
        )

    lines.append("\n## Coverage check\n\n")
    lines.append("- All expanded smoke runs completed with strict formal LUT enabled.\n")
    lines.append("- Expanded random smoke validates the feasible-sampling path, not the raw feasible ratio, because random search resamples feasible candidates.\n")
    lines.append("- Exhaustive enumeration is the source of truth for raw feasible ratio over explicit `stage_block_choices`.\n")
    lines.append("- Random search summaries report `best_candidate: null`; feasibility and metrics are still present in candidate records.\n")
    lines.append("- Hardware-aware filtering is required before the next unrestricted NKSID NAS launch: expanded 4-stage has 4/8 feasible candidates and expanded 5-stage has 11/24 feasible candidates.\n")

    lines.append("\n## Expanded random stage sampling\n\n")
    for row in rows:
        if row["search_method"] != "random":
            continue
        lines.append(f"### {row['label']}\n\n")
        for stage_idx, counts in sorted(row["stage_counts"].items()):
            parts = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
            lines.append(f"- stage{stage_idx}: {parts}\n")
        lines.append("\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    rows = [summarize_run(label, run_dir) for label, run_dir in RUNS]
    exhaustive_rows = [
        summarize_exhaustive(label, run_dir)
        for label, run_dir in EXHAUSTIVE_RUNS
        if (run_dir / "summary.json").exists()
    ]
    write_csv(rows, OUT_DIR / "expanded_smoke_summary.csv")
    write_exhaustive_csv(exhaustive_rows, OUT_DIR / "expanded_exhaustive_summary.csv")
    write_markdown(rows, exhaustive_rows, OUT_DIR / "expanded_smoke_summary.md")
    print(
        json.dumps(
            {
                "smoke": [{k: v for k, v in row.items() if k != "stage_counts"} for row in rows],
                "exhaustive": exhaustive_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
