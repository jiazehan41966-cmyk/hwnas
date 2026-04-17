#!/usr/bin/env python3
"""Plot a clean heavy-budget comparison for RL vs Proxyless runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RL_RUN = "results/full_av7k325_formal_rl_eval200_noes_20260329"
DEFAULT_PROXYLESS_OFFICIAL_RUN = "results/strategy_compare_budget200_proxyless_official_av7k325_full"
DEFAULT_RANDOM_RUN = "results/strategy_compare_budget200_random_av7k325_full"
DEFAULT_OUTPUT_DIR = "results/rl_vs_proxyless_heavy_budget_viz"


@dataclass
class RunMetric:
    label: str
    run_dir: str
    best_arch: str
    accuracy: float
    latency_ms: float
    dsp: int
    lut: int
    total_evaluated: int
    feasible: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a heavy-budget RL vs Proxyless comparison figure.",
    )
    parser.add_argument("--rl-run", default=DEFAULT_RL_RUN, help="RL run directory.")
    parser.add_argument(
        "--proxyless-official-run",
        default=DEFAULT_PROXYLESS_OFFICIAL_RUN,
        help="Proxyless official run directory.",
    )
    parser.add_argument(
        "--random-run",
        default=DEFAULT_RANDOM_RUN,
        help="Random baseline run directory used as a reference line.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated figure and summary.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_metric(run_dir: Path, label: str) -> RunMetric:
    summary = _load_json(run_dir / "results" / "summary.json")
    best_payload = _load_json(run_dir / "results" / "best_candidate.json")

    candidate = best_payload.get("candidate") or {}
    metrics = candidate.get("metrics") or {}

    return RunMetric(
        label=label,
        run_dir=str(run_dir),
        best_arch=str(candidate.get("arch_id") or "n/a"),
        accuracy=float(metrics.get("accuracy") or 0.0),
        latency_ms=float(metrics.get("latency_ms") or 0.0),
        dsp=int(metrics.get("dsp") or 0),
        lut=int(metrics.get("lut") or 0),
        total_evaluated=int(summary.get("total_evaluated") or 0),
        feasible=int(summary.get("feasible") or 0),
    )


def save_summary(
    rl: RunMetric,
    proxyless_official: RunMetric,
    random_ref: RunMetric,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "rl_vs_proxyless_official_heavy_budget_summary.json"
    payload = {
        "setting": "heavy_budget_or_retraining",
        "bars": [
            asdict(rl),
            asdict(proxyless_official),
        ],
        "random_reference": asdict(random_ref),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def plot_figure(
    rl: RunMetric,
    proxyless_official: RunMetric,
    random_ref: RunMetric,
    output_dir: Path,
) -> Path:
    labels = [
        "RL",
        "Proxyless official",
    ]
    values = [
        rl.accuracy * 100.0,
        proxyless_official.accuracy * 100.0,
    ]
    colors = ["#C44E52", "#54A24B"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.62, edgecolor="black", linewidth=0.8)

    random_value = random_ref.accuracy * 100.0
    ax.axhline(
        random_value,
        color="#4C78A8",
        linestyle="--",
        linewidth=2.0,
        label=f"Random reference: {random_value:.2f}%",
    )

    for bar, value, metric in zip(
        bars,
        values,
        [rl, proxyless_official],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            3.5,
            f"lat {metric.latency_ms:.3f} ms\nDSP {metric.dsp} | LUT {metric.lut}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Best accuracy (%)")
    ax.set_title("Heavy-Budget Search Comparison: RL vs Proxyless", fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    note = (
        "Bars show the best candidate from the heavy-budget setting. "
        "RL is compared against the stronger Proxyless run; "
        "Random is shown only as a reference line."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color="#444444")

    fig.tight_layout(rect=(0.02, 0.05, 0.98, 1.0))
    output_path = output_dir / "rl_vs_proxyless_official_heavy_budget.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rl = load_run_metric(Path(args.rl_run), "RL")
    proxyless_official = load_run_metric(
        Path(args.proxyless_official_run),
        "Proxyless official",
    )
    random_ref = load_run_metric(Path(args.random_run), "Random")

    figure_path = plot_figure(
        rl,
        proxyless_official,
        random_ref,
        output_dir,
    )
    summary_path = save_summary(
        rl,
        proxyless_official,
        random_ref,
        output_dir,
    )

    print(f"[OK] Figure saved to {figure_path}")
    print(f"[OK] Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
