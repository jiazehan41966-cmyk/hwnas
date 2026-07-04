#!/usr/bin/env python3
"""Calibrate the analytic hardware surrogate against routed board evidence.

Pairs every routed harness run (post-route Vivado utilization + COM5
measurement) with the search-time estimates stored in its Pareto candidate
artifact, then reports per-metric error statistics and recommended
calibration factors for the analytic cost model.

The output quantifies what the 2026-07-04 assessment found qualitatively:
analytic latency is underestimated ~4-7x for mbconv-heavy networks while
LUT-measured operators land near 1x, and analytic DSP/LUT are systematically
overestimated. Until the analytic model is recalibrated, Pareto fronts built
from raw analytic values are distorted.

Usage:
  python scripts/calibrate_hw_surrogate.py \
      --output-dir artifacts/hw_surrogate_calibration
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOARD_RESULTS = (
    REPO_ROOT / "hls_lut_builder" / "board_harness" / "results"
)
RUN_ROOT = BOARD_RESULTS / "sonar_classifier_rl_best_e2e_board"

RUN_DIR_PATTERN = re.compile(r"^full_rank\d+_(?P<arch_id>.+?)_[0-9a-f]{6,}$")

METRICS = ("latency_ms", "dsp", "lut", "bram")


def parse_utilization_report(path: Path) -> dict[str, float]:
    """Extract post-route Slice LUTs, DSPs, and BRAM tiles from a Vivado report."""
    text = path.read_text(encoding="utf-8", errors="replace")

    def first_value(pattern: str) -> float | None:
        match = re.search(pattern, text)
        if match is None:
            return None
        return float(match.group(1))

    lut = first_value(r"\|\s*(?:Slice|CLB) LUTs\*?\s*\|\s*([\d.]+)\s*\|")
    dsp = first_value(r"\|\s*DSPs\s*\|\s*([\d.]+)\s*\|")
    bram = first_value(r"\|\s*Block RAM Tile\s*\|\s*([\d.]+)\s*\|")
    result: dict[str, float] = {}
    if lut is not None:
        result["lut"] = lut
    if dsp is not None:
        result["dsp"] = dsp
    if bram is not None:
        result["bram"] = bram
    return result


def load_measured_latency_ms(run_dir: Path) -> float | None:
    measurements = sorted((run_dir / "measurements").glob("*.measurement.json"))
    for path in measurements:
        if path.name.endswith(".single_run.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        latency = payload.get("latency_ms")
        if latency is not None:
            return float(latency)
    return None


def index_candidates() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in BOARD_RESULTS.glob("pareto_route_gate_*/candidates/*.candidate.json"):
        match = re.match(r"^\d+_(?P<arch_id>.+)\.candidate$", path.stem)
        if match is None:
            continue
        index.setdefault(match.group("arch_id"), []).append(path)
    return index


def load_candidate_estimates(paths: list[Path]) -> tuple[dict[str, float] | None, str]:
    """Load search-time estimates; refuse silently-ambiguous candidates."""
    loaded: list[tuple[Path, dict[str, float]]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("candidate", {}).get("metrics", {})
        estimates = {
            "latency_ms": metrics.get("latency_ms"),
            "dsp": metrics.get("dsp"),
            "lut": metrics.get("lut"),
            "bram": metrics.get("bram"),
        }
        if any(value is None for value in estimates.values()):
            continue
        loaded.append((path, {key: float(value) for key, value in estimates.items()}))

    if not loaded:
        return None, "no readable candidate metrics"
    reference = loaded[0][1]
    for _, estimates in loaded[1:]:
        if any(
            not math.isclose(estimates[key], reference[key], rel_tol=1e-6)
            for key in METRICS
        ):
            return None, (
                "ambiguous: same arch_id has conflicting estimates in "
                + ", ".join(str(path.relative_to(REPO_ROOT)) for path, _ in loaded)
            )
    return reference, str(loaded[0][0].relative_to(REPO_ROOT))


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_ratios(rows: list[dict], metric: str) -> dict:
    ratios = [
        row["ratio"][metric]
        for row in rows
        if row["ratio"].get(metric) is not None
    ]
    if not ratios:
        return {"n": 0}
    mape = statistics.fmean(abs(1.0 - 1.0 / ratio) for ratio in ratios)
    return {
        "n": len(ratios),
        "geomean_measured_over_estimated": geomean(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "mape_of_estimate": mape,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="artifacts/hw_surrogate_calibration"
    )
    args = parser.parse_args()

    candidate_index = index_candidates()
    rows: list[dict] = []
    skipped: list[dict] = []

    for run_dir in sorted(RUN_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if match is None:
            continue
        arch_id = match.group("arch_id")

        report_path = None
        for candidate_report in (
            run_dir / "vivado_reports" / "utilization.rpt",
            run_dir / "harness_project" / "reports" / "utilization.rpt",
        ):
            if candidate_report.exists():
                report_path = candidate_report
                break

        measured: dict[str, float] = {}
        if report_path is not None:
            measured.update(parse_utilization_report(report_path))
        latency = load_measured_latency_ms(run_dir)
        if latency is not None:
            measured["latency_ms"] = latency

        if not measured:
            skipped.append({"run": run_dir.name, "reason": "no routed report or measurement"})
            continue

        candidate_paths = candidate_index.get(arch_id, [])
        estimates, source = load_candidate_estimates(candidate_paths)
        if estimates is None:
            skipped.append({"run": run_dir.name, "arch_id": arch_id, "reason": source})
            continue

        ratio = {
            metric: (measured[metric] / estimates[metric])
            if metric in measured and estimates.get(metric)
            else None
            for metric in METRICS
        }
        rows.append(
            {
                "run": run_dir.name,
                "arch_id": arch_id,
                "candidate_source": source,
                "estimated": estimates,
                "measured": measured,
                "ratio": ratio,
            }
        )

    if not rows:
        print("No paired routed evidence found; nothing to calibrate.", file=sys.stderr)
        return 1

    summary = {metric: summarize_ratios(rows, metric) for metric in METRICS}

    # Recommended factors, chosen so the calibrated analytic model errs on
    # the safe side for feasibility screening:
    # - latency: scale UP by the worst observed underestimate (max ratio),
    #   so a candidate the calibrated model calls fast really is fast;
    # - resources: scale by the geomean so totals track routed reality,
    #   never below the max observed ratio (avoid new underestimates).
    recommended: dict[str, float] = {}
    latency_stats = summary["latency_ms"]
    if latency_stats.get("n"):
        recommended["latency_scale"] = latency_stats["max"]
    for metric, key in (("dsp", "dsp_scale"), ("lut", "lut_scale"), ("bram", "bram_scale")):
        stats = summary[metric]
        if stats.get("n"):
            # Overestimated resources (max ratio < 1): track routed reality via
            # the geomean. If any run was underestimated, take the worst ratio
            # instead so the calibrated model never under-reports usage.
            recommended[key] = (
                stats["max"]
                if stats["max"] > 1.0
                else stats["geomean_measured_over_estimated"]
            )

    # Residual error after applying the recommended factors: measured vs
    # calibrated estimate. This is the error bound a search may assume.
    factor_by_metric = {
        "latency_ms": recommended.get("latency_scale"),
        "dsp": recommended.get("dsp_scale"),
        "lut": recommended.get("lut_scale"),
        "bram": recommended.get("bram_scale"),
    }
    residuals: dict[str, dict] = {}
    for metric in METRICS:
        factor = factor_by_metric.get(metric)
        if not factor:
            continue
        values = [
            row["ratio"][metric] / factor
            for row in rows
            if row["ratio"].get(metric) is not None
        ]
        if values:
            residuals[metric] = {
                "n": len(values),
                "geomean": geomean(values),
                "min": min(values),
                "max": max(values),
                "max_abs_error": max(abs(1.0 - value) for value in values),
            }

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "post-route Vivado utilization + COM5 deterministic measurements",
        "run_root": str(RUN_ROOT.relative_to(REPO_ROOT)),
        "num_paired_runs": len(rows),
        "pairs": rows,
        "skipped": skipped,
        "summary": summary,
        "post_calibration_residuals": residuals,
        "recommended_analytic_calibration": recommended,
        "note": (
            "ratios are measured/estimated; latency_scale uses the worst "
            "underestimate, resource scales use geomean (or max when the "
            "model also underestimates). Apply only to analytic-fallback "
            "layers, never to LUT-measured entries."
        ),
    }
    json_path = output_dir / "hw_surrogate_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Hardware surrogate calibration",
        "",
        f"Paired routed runs: {len(rows)} (skipped: {len(skipped)})",
        "",
        "| run | arch | est lat ms | meas lat ms | ratio | est DSP | meas DSP | est LUT | meas LUT | est BRAM | meas BRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value: float | None, spec: str = ".2f") -> str:
        return format(value, spec) if value is not None else "-"

    for row in rows:
        est, meas, ratio = row["estimated"], row["measured"], row["ratio"]
        lines.append(
            f"| {row['run']} | {row['arch_id']} "
            f"| {est['latency_ms']:.2f} | {fmt(meas.get('latency_ms'))} "
            f"| {fmt(ratio.get('latency_ms'))} "
            f"| {est['dsp']:.0f} | {fmt(meas.get('dsp'), '.0f')} "
            f"| {est['lut']:.0f} | {fmt(meas.get('lut'), '.0f')} "
            f"| {est['bram']:.0f} | {fmt(meas.get('bram'), '.0f')} |"
        )
    lines += ["", "## Ratio summary (measured / estimated)", ""]
    lines += ["| metric | n | geomean | min | max | MAPE of estimate |", "|---|---:|---:|---:|---:|---:|"]
    for metric in METRICS:
        stats = summary[metric]
        if stats.get("n"):
            lines.append(
                f"| {metric} | {stats['n']} | {stats['geomean_measured_over_estimated']:.3f} "
                f"| {stats['min']:.3f} | {stats['max']:.3f} | {stats['mape_of_estimate']:.1%} |"
            )
    lines += ["", "## Post-calibration residual (measured / calibrated estimate)", ""]
    lines += ["| metric | n | geomean | min | max | worst abs error |", "|---|---:|---:|---:|---:|---:|"]
    for metric, stats in residuals.items():
        lines.append(
            f"| {metric} | {stats['n']} | {stats['geomean']:.3f} "
            f"| {stats['min']:.3f} | {stats['max']:.3f} | {stats['max_abs_error']:.1%} |"
        )
    lines += [
        "",
        "## Recommended analytic calibration",
        "",
        "```json",
        json.dumps(recommended, indent=2),
        "```",
        "",
        "Apply via `hardware.analytic_calibration_path` in a search config;",
        "factors touch only analytic-fallback layers, never LUT-measured entries.",
    ]
    md_path = output_dir / "hw_surrogate_calibration.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Paired {len(rows)} routed runs; skipped {len(skipped)}.")
    for metric in METRICS:
        stats = summary[metric]
        if stats.get("n"):
            print(
                f"  {metric}: geomean ratio {stats['geomean_measured_over_estimated']:.3f} "
                f"(range {stats['min']:.3f}-{stats['max']:.3f}, "
                f"MAPE {stats['mape_of_estimate']:.1%}, n={stats['n']})"
            )
    print(f"Recommended: {json.dumps(recommended)}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
