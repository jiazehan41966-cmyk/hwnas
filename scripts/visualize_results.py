#!/usr/bin/env python3
"""
Visualize HW-NAS run outputs.

This script works with one or more run directories under `results/`.
It reads:
- results/summary.json
- results/candidates.json (preferred) or results/candidates.jsonl (fallback)
and generates:
- accuracy_vs_latency.png
- resource_comparison.png
- search_convergence.png
- pareto_front.png
- pareto_front_by_run.png
- metric_distributions.png
- topk_accuracy_curve.png
- best_metrics_radar.png
- relative_improvement_vs_baseline.png
- results_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    return {}


def load_candidates(results_dir: Path) -> List[Dict[str, Any]]:
    candidates_json = results_dir / "candidates.json"
    if candidates_json.exists():
        with candidates_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    candidates_jsonl = results_dir / "candidates.jsonl"
    if not candidates_jsonl.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with candidates_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate = record.get("candidate")
            if isinstance(candidate, dict):
                rows.append(candidate)
    return rows


def load_experiment(run_dir: Path) -> Dict[str, Any]:
    results_dir = run_dir / "results"
    summary_path = results_dir / "summary.json"
    run_info_path = run_dir / "run_info.json"
    run_info = load_json(run_info_path) if run_info_path.exists() else {}
    summary = load_json(summary_path) if summary_path.exists() else {}
    candidates = load_candidates(results_dir)
    if not summary:
        search_state_path = run_dir / "checkpoints" / "search_state.json"
        if search_state_path.exists():
            state = load_json(search_state_path)
            summary = {
                "total_evaluated": state.get("total_evaluated", len(candidates)),
                "feasible": state.get("feasible", 0),
                "infeasible": state.get("infeasible", 0),
                "best_candidate": state.get("best_candidate", {}),
                "extra": {
                    "search_method": run_info.get("search_method", "unknown"),
                    "status": run_info.get("status", "running"),
                },
            }
    if summary and isinstance(summary, dict):
        extra = summary.setdefault("extra", {})
        if "search_method" not in extra and "search_method" in run_info:
            extra["search_method"] = run_info["search_method"]
    return {
        "name": run_dir.name,
        "run_dir": run_dir,
        "run_info": run_info,
        "summary": summary,
        "candidates": candidates,
    }


def get_method(exp: Dict[str, Any]) -> str:
    return (
        exp.get("summary", {})
        .get("extra", {})
        .get("search_method", "unknown")
    )


def metric_from_best(exp: Dict[str, Any], metric_name: str, default: float = 0.0) -> float:
    best = exp.get("summary", {}).get("best_candidate", {})
    metrics = best.get("metrics", {}) if isinstance(best, dict) else {}
    value = metrics.get(metric_name, default)
    try:
        return float(value)
    except Exception:
        return default


def metric_values_from_candidates(
    exp: Dict[str, Any],
    metric_name: str,
    scale: float = 1.0,
) -> List[float]:
    values: List[float] = []
    for cand in exp.get("candidates", []):
        metrics = cand.get("metrics", {})
        value = metrics.get(metric_name, None)
        try:
            if value is not None:
                values.append(float(value) * scale)
        except Exception:
            continue
    return values


def find_baseline_experiment(experiments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for exp in experiments:
        if get_method(exp).lower() == "random":
            return exp
    return experiments[0] if experiments else None


def print_summary_table(experiments: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print("Experiment Summary")
    print("=" * 88)
    headers = ["Run", "Method", "Candidates", "Best Acc(%)", "Latency(ms)", "DSP", "LUT"]
    widths = [40, 10, 12, 12, 12, 6, 8]
    header = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header)
    print("-" * 88)
    for exp in experiments:
        summary = exp.get("summary", {})
        total = int(summary.get("total_evaluated", len(exp.get("candidates", []))))
        acc = metric_from_best(exp, "accuracy") * 100.0
        latency = metric_from_best(exp, "latency_ms")
        dsp = int(metric_from_best(exp, "dsp"))
        lut = int(metric_from_best(exp, "lut"))
        row = (
            f"{exp['name'][:39]:<40}"
            f"{get_method(exp):<10}"
            f"{total:<12}"
            f"{acc:<12.2f}"
            f"{latency:<12.4f}"
            f"{dsp:<6}"
            f"{lut:<8}"
        )
        print(row)
    print("=" * 88)


def print_topk(experiments: List[Dict[str, Any]], top_k: int) -> None:
    print("\n" + "=" * 88)
    print(f"Top-{top_k} Candidates by Accuracy")
    print("=" * 88)
    for exp in experiments:
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda c: c.get("metrics", {}).get("accuracy", 0.0),
            reverse=True,
        )[:top_k]
        print(f"\n[RUN] {exp['name']}")
        print(f"{'Rank':<6}{'Arch ID':<16}{'Acc(%)':<10}{'Latency(ms)':<13}{'DSP':<7}{'LUT':<8}")
        print("-" * 68)
        for i, cand in enumerate(ranked, start=1):
            metrics = cand.get("metrics", {})
            print(
                f"{i:<6}"
                f"{cand.get('arch_id', '?'):<16}"
                f"{(metrics.get('accuracy', 0.0) * 100):<10.2f}"
                f"{metrics.get('latency_ms', 0.0):<13.4f}"
                f"{int(metrics.get('dsp', 0)):<7}"
                f"{int(metrics.get('lut', 0)):<8}"
            )


def plot_accuracy_vs_latency(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    for idx, exp in enumerate(experiments):
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        xs = [c.get("metrics", {}).get("latency_ms", 0.0) for c in candidates]
        ys = [c.get("metrics", {}).get("accuracy", 0.0) * 100 for c in candidates]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"{get_method(exp)} ({len(candidates)})"
        ax.scatter(xs, ys, c=color, marker=marker, s=45, alpha=0.65, label=label)
        best_x = metric_from_best(exp, "latency_ms")
        best_y = metric_from_best(exp, "accuracy") * 100.0
        ax.scatter([best_x], [best_y], c=color, marker="*", s=220, edgecolors="black", linewidths=1.2)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = output_dir / "accuracy_vs_latency.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_resource_comparison(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    labels = [f"{get_method(exp)}\n{exp['name'][:14]}" for exp in experiments]
    dsp = [metric_from_best(exp, "dsp") for exp in experiments]
    lut = [metric_from_best(exp, "lut") for exp in experiments]
    bram = [metric_from_best(exp, "bram") for exp in experiments]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, dsp, color="#4c78a8")
    axes[0].set_title("DSP")
    axes[1].bar(labels, lut, color="#f58518")
    axes[1].set_title("LUT")
    axes[2].bar(labels, bram, color="#54a24b")
    axes[2].set_title("BRAM")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Best-Candidate Resource Usage")
    fig.tight_layout()
    out = output_dir / "resource_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_search_convergence(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, exp in enumerate(experiments):
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        accs = [c.get("metrics", {}).get("accuracy", 0.0) * 100.0 for c in candidates]
        best_so_far: List[float] = []
        best = 0.0
        for a in accs:
            best = max(best, a)
            best_so_far.append(best)
        x = list(range(1, len(best_so_far) + 1))
        ax.plot(x, best_so_far, color=colors[idx % len(colors)], linewidth=2.0, label=get_method(exp))
    ax.set_xlabel("Evaluated Architectures")
    ax.set_ylabel("Best Accuracy So Far (%)")
    ax.set_title("Search Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "search_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # maximize accuracy (y), minimize latency (x)
    pts = sorted(points, key=lambda p: p[0])  # by latency asc
    frontier: List[Tuple[float, float]] = []
    best_acc = float("-inf")
    for x, y in pts:
        if y > best_acc:
            frontier.append((x, y))
            best_acc = y
    return frontier


def plot_pareto_front(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    all_points: List[Tuple[float, float]] = []
    for idx, exp in enumerate(experiments):
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        xs = [c.get("metrics", {}).get("latency_ms", 0.0) for c in candidates]
        ys = [c.get("metrics", {}).get("accuracy", 0.0) * 100.0 for c in candidates]
        all_points.extend(zip(xs, ys))
        ax.scatter(xs, ys, s=35, alpha=0.55, color=colors[idx % len(colors)], label=get_method(exp))
    if all_points:
        pf = pareto_front(all_points)
        pfx = [p[0] for p in pf]
        pfy = [p[1] for p in pf]
        ax.plot(pfx, pfy, "k--", linewidth=2.0, label="Pareto Front")
        ax.scatter(pfx, pfy, c="red", s=120, marker="*", zorder=5)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pareto Front")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "pareto_front.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_pareto_front_by_run(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    has_curve = False
    for idx, exp in enumerate(experiments):
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        points = [
            (
                c.get("metrics", {}).get("latency_ms", 0.0),
                c.get("metrics", {}).get("accuracy", 0.0) * 100.0,
            )
            for c in candidates
        ]
        pf = pareto_front(points)
        if not pf:
            continue
        has_curve = True
        pfx = [p[0] for p in pf]
        pfy = [p[1] for p in pf]
        color = colors[idx % len(colors)]
        ax.plot(pfx, pfy, color=color, linewidth=2.2, label=f"{get_method(exp)} pareto")
        ax.scatter(pfx, pfy, color=color, s=45, alpha=0.85)

    if not has_curve:
        plt.close(fig)
        return

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pareto Front by Run")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = output_dir / "pareto_front_by_run.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_metric_distributions(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    metric_specs = [
        ("accuracy", "Accuracy (%)", 100.0),
        ("latency_ms", "Latency (ms)", 1.0),
        ("energy_mj", "Energy (mJ)", 1.0),
        ("dsp", "DSP", 1.0),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    has_data = False
    for ax, (metric_name, title, scale) in zip(axes_flat, metric_specs):
        for idx, exp in enumerate(experiments):
            values = metric_values_from_candidates(exp, metric_name, scale=scale)
            if not values:
                continue
            has_data = True
            ax.hist(
                values,
                bins=min(20, max(8, len(values) // 4)),
                alpha=0.42,
                color=colors[idx % len(colors)],
                label=get_method(exp),
                edgecolor="white",
                linewidth=0.7,
            )
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("Count")
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend()

    if not has_data:
        plt.close(fig)
        return

    fig.suptitle("Metric Distributions Across Candidates")
    fig.tight_layout()
    out = output_dir / "metric_distributions.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_topk_accuracy_curve(experiments: List[Dict[str, Any]], output_dir: Path, top_k: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    has_curve = False
    for idx, exp in enumerate(experiments):
        candidates = exp.get("candidates", [])
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda c: c.get("metrics", {}).get("accuracy", 0.0),
            reverse=True,
        )
        top = ranked[:max(1, top_k)]
        ys = [c.get("metrics", {}).get("accuracy", 0.0) * 100.0 for c in top]
        xs = list(range(1, len(ys) + 1))
        has_curve = True
        ax.plot(
            xs,
            ys,
            color=colors[idx % len(colors)],
            marker="o",
            linewidth=2.0,
            label=get_method(exp),
        )

    if not has_curve:
        plt.close(fig)
        return

    ax.set_xlabel("Rank in Top-K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Top-{top_k} Accuracy Curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out = output_dir / "topk_accuracy_curve.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_best_metrics_radar(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    if not experiments:
        return

    categories = [
        ("accuracy", True, "Accuracy"),
        ("latency_ms", False, "Latency"),
        ("energy_mj", False, "Energy"),
        ("dsp", False, "DSP"),
        ("lut", False, "LUT"),
        ("bram", False, "BRAM"),
    ]

    maximize_refs: Dict[str, float] = {}
    minimize_refs: Dict[str, float] = {}
    for metric_name, maximize, _label in categories:
        values = [metric_from_best(exp, metric_name, default=0.0) for exp in experiments]
        positive_values = [v for v in values if v > 0]
        if maximize:
            maximize_refs[metric_name] = max(values) if values else 1.0
        else:
            minimize_refs[metric_name] = min(positive_values) if positive_values else 1.0

    def score(metric_name: str, maximize: bool, value: float) -> float:
        if maximize:
            ref = maximize_refs.get(metric_name, 1.0)
            return 0.0 if ref <= 0 else value / ref
        ref = minimize_refs.get(metric_name, 1.0)
        if value <= 0:
            return 0.0
        return ref / value

    angles = [n / float(len(categories)) * 2.0 * math.pi for n in range(len(categories))]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, exp in enumerate(experiments):
        values = []
        for metric_name, maximize, _label in categories:
            raw = metric_from_best(exp, metric_name, default=0.0)
            values.append(score(metric_name, maximize, raw))
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2.0, label=get_method(exp))
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _metric, _maximize, label in categories])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Best Candidate Radar (Higher Is Better, Normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    out = output_dir / "best_metrics_radar.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def plot_relative_improvement(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    if len(experiments) < 2:
        return

    baseline = find_baseline_experiment(experiments)
    if baseline is None:
        return

    comparisons = [exp for exp in experiments if exp["name"] != baseline["name"]]
    if not comparisons:
        return

    metric_specs = [
        ("accuracy", True, "Accuracy"),
        ("latency_ms", False, "Latency"),
        ("energy_mj", False, "Energy"),
        ("dsp", False, "DSP"),
        ("lut", False, "LUT"),
        ("bram", False, "BRAM"),
    ]
    x_labels = [name for _metric, _maximize, name in metric_specs]
    x = list(range(len(metric_specs)))
    width = 0.75 / max(1, len(comparisons))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    for idx, exp in enumerate(comparisons):
        offsets = [i - 0.375 + width / 2.0 + idx * width for i in x]
        deltas: List[float] = []
        for metric_name, maximize, _label in metric_specs:
            base = metric_from_best(baseline, metric_name, default=0.0)
            cur = metric_from_best(exp, metric_name, default=0.0)
            if abs(base) < 1e-12:
                deltas.append(0.0)
                continue
            if maximize:
                delta = (cur - base) / abs(base) * 100.0
            else:
                delta = (base - cur) / abs(base) * 100.0
            deltas.append(delta)
        ax.bar(
            offsets,
            deltas,
            width=width,
            color=colors[idx % len(colors)],
            alpha=0.85,
            label=f"{get_method(exp)}:{exp['name'][:12]}",
        )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Improvement (%)")
    ax.set_title(f"Relative Improvement vs Baseline ({get_method(baseline)})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = output_dir / "relative_improvement_vs_baseline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def generate_latex_table(experiments: List[Dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{HW-NAS Search Results}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & Candidates & Acc (\\%) & Latency (ms) & DSP & LUT & BRAM \\\\",
        "\\hline",
    ]
    for exp in experiments:
        summary = exp.get("summary", {})
        total = int(summary.get("total_evaluated", len(exp.get("candidates", []))))
        lines.append(
            f"{get_method(exp)} & {total} & "
            f"{metric_from_best(exp, 'accuracy') * 100:.2f} & "
            f"{metric_from_best(exp, 'latency_ms'):.4f} & "
            f"{int(metric_from_best(exp, 'dsp'))} & "
            f"{int(metric_from_best(exp, 'lut'))} & "
            f"{int(metric_from_best(exp, 'bram'))} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\label{tab:hwnas_results}", "\\end{table}"])
    out = output_dir / "results_table.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize HW-NAS run outputs")
    parser.add_argument("result_dirs", nargs="+", help="Run directories (e.g., results/run_xxx)")
    parser.add_argument("--output", "-o", default="results/comparison", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K entries to print")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments: List[Dict[str, Any]] = []
    for rd in args.result_dirs:
        run_dir = Path(rd)
        if not run_dir.exists():
            print(f"[WARN] Not found: {run_dir}")
            continue
        experiments.append(load_experiment(run_dir))
        print(f"[OK] Loaded: {run_dir}")

    if not experiments:
        print("[ERROR] No valid run directories found")
        return 1

    print_summary_table(experiments)
    print_topk(experiments, args.top_k)

    if HAS_MATPLOTLIB and not args.no_plots:
        print("\n[INFO] Generating plots...")
        plot_accuracy_vs_latency(experiments, output_dir)
        plot_resource_comparison(experiments, output_dir)
        plot_search_convergence(experiments, output_dir)
        plot_pareto_front(experiments, output_dir)
        plot_pareto_front_by_run(experiments, output_dir)
        plot_metric_distributions(experiments, output_dir)
        plot_topk_accuracy_curve(experiments, output_dir, top_k=max(5, args.top_k))
        plot_best_metrics_radar(experiments, output_dir)
        plot_relative_improvement(experiments, output_dir)
    elif not HAS_MATPLOTLIB:
        print("[WARN] matplotlib unavailable, skipped figure generation")

    generate_latex_table(experiments, output_dir)
    print(f"\n[OK] All outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
