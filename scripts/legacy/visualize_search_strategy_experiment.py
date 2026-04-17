#!/usr/bin/env python3
"""Visualize the search-strategy selection experiments."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


DEFAULT_RUN_DIRS = [
    "results/seq_random_baseline_20260328_021931",
    "results/seq_rl_baseline_20260328_004201",
    "results/seq_proxyless_official_20260328_003048",
    "results/seq_proxyless_mit_aligned_20260328_035621",
]

LABEL_OVERRIDES = {
    "seq_random_baseline_20260328_021931": "Random baseline",
    "seq_rl_baseline_20260328_004201": "RL baseline",
    "seq_proxyless_official_20260328_003048": "Proxyless official",
    "seq_proxyless_mit_aligned_20260328_035621": "Proxyless aligned",
    "full_av7k325_formal_rl_eval200_noes_20260329": "RL 200ep AV7K325 Full",
    "strategy_compare_budget200_random_av7k325_full": "Random 200ep AV7K325 Full",
    "strategy_compare_budget200_proxyless_official_av7k325_full": "Proxyless official 200ep AV7K325 Full",
    "strategy_compare_budget200_proxyless_aligned_av7k325_full": "Proxyless aligned 200ep AV7K325 Full",
    "strategy_compare_budget200_random_av7k325_small": "Random 200ep AV7K325",
    "strategy_compare_budget200_rl_av7k325_small": "RL 200ep AV7K325",
    "strategy_compare_budget200_proxyless_official_av7k325_small": "Proxyless official 200ep AV7K325",
    "strategy_compare_budget200_proxyless_aligned_av7k325_small": "Proxyless aligned 200ep AV7K325",
    "strategy_compare_budget200_random_av7k325_small64": "Random 200ep AV7K325 64",
    "strategy_compare_budget200_rl_av7k325_small64": "RL 200ep AV7K325 64",
    "strategy_compare_budget200_proxyless_official_av7k325_small64": "Proxyless official 200ep AV7K325 64",
    "strategy_compare_budget200_proxyless_aligned_av7k325_small64": "Proxyless aligned 200ep AV7K325 64",
}

COLOR_OVERRIDES = {
    "random": "#4C78A8",
    "rl": "#F58518",
    "proxyless": "#54A24B",
}

RUN_COLORS = {
    "Random baseline": "#4C78A8",
    "RL baseline": "#F58518",
    "Proxyless official": "#54A24B",
    "Proxyless aligned": "#E45756",
    "RL 200ep AV7K325 Full": "#C44E52",
    "Random 200ep AV7K325 Full": "#4C78A8",
    "Proxyless official 200ep AV7K325 Full": "#54A24B",
    "Proxyless aligned 200ep AV7K325 Full": "#E45756",
    "Random 200ep AV7K325": "#4C78A8",
    "RL 200ep AV7K325": "#F58518",
    "Proxyless official 200ep AV7K325": "#54A24B",
    "Proxyless aligned 200ep AV7K325": "#E45756",
    "Random 200ep AV7K325 64": "#4C78A8",
    "RL 200ep AV7K325 64": "#F58518",
    "Proxyless official 200ep AV7K325 64": "#54A24B",
    "Proxyless aligned 200ep AV7K325 64": "#E45756",
}


@dataclass
class RoundSummary:
    type: str
    configured_eval_epochs: int | None = None
    configured_warmup_epochs: int | None = None
    configured_search_epochs: int | None = None
    configured_total_epochs: int | None = None
    actual_search_epochs: int | None = None
    actual_total_epochs: int | None = None
    actual_min_epochs: int | None = None
    actual_avg_epochs: float | None = None
    actual_median_epochs: float | None = None
    actual_max_epochs: int | None = None
    best_candidate_epochs: int | None = None


@dataclass
class ExperimentRecord:
    run_name: str
    label: str
    path: str
    method: str
    status: str
    total_evaluated: int
    feasible: int
    infeasible: int
    best_arch: str
    best_accuracy: float
    best_latency_ms: float
    best_dsp: int
    best_lut: int
    accuracies: list[float]
    latencies: list[float]
    progress_steps: list[int]
    progress_best_acc: list[float]
    budget_steps: list[int]
    budget_best_acc: list[float]
    total_epoch_budget: int
    actual_epoch_lengths: list[int]
    round_summary: RoundSummary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize search-strategy experiment runs.",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="Experiment directories to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/search_strategy_selection_viz",
        help="Directory for generated figures and summaries.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _running_best(values: list[float]) -> list[float]:
    best = float("-inf")
    output: list[float] = []
    for value in values:
        best = max(best, float(value))
        output.append(best)
    return output


def _format_round_summary(record: ExperimentRecord) -> str:
    summary = record.round_summary
    if summary.type == "per_candidate":
        return (
            f"{record.label}: cfg {summary.configured_eval_epochs} ep/candidate | "
            f"actual min/avg/median/max "
            f"{summary.actual_min_epochs}/{summary.actual_avg_epochs:.2f}/"
            f"{summary.actual_median_epochs:.1f}/{summary.actual_max_epochs} | "
            f"best {summary.best_candidate_epochs}"
        )
    return (
        f"{record.label}: warmup {summary.configured_warmup_epochs} + "
        f"search {summary.configured_search_epochs} = {summary.configured_total_epochs} | "
        f"actual search {summary.actual_search_epochs}, total {summary.actual_total_epochs}"
    )


def _record_label(run_dir: Path, method: str) -> str:
    if run_dir.name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[run_dir.name]
    return f"{method}:{run_dir.name}"


def load_experiment(run_dir: Path) -> ExperimentRecord:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config = _load_yaml(run_dir / "config.yaml")
    summary = _load_json(run_dir / "results" / "summary.json")
    best_payload = _load_json(run_dir / "results" / "best_candidate.json")
    candidates = _load_json(run_dir / "results" / "candidates.json")

    search_cfg = config.get("search", {})
    proxyless_cfg = search_cfg.get("proxyless", {})
    method = str(search_cfg.get("method", "unknown"))
    label = _record_label(run_dir, method)

    accuracies = [
        float((candidate.get("metrics") or {}).get("accuracy") or 0.0)
        for candidate in candidates
    ]
    latencies = [
        float((candidate.get("metrics") or {}).get("latency_ms") or 0.0)
        for candidate in candidates
    ]

    jsonl_records: list[dict[str, Any]] = []
    epoch_lengths: list[int] = []
    with (run_dir / "results" / "candidates.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            jsonl_records.append(payload)
            history = payload.get("history") or {}
            train_loss = history.get("train_loss")
            if isinstance(train_loss, list):
                epoch_lengths.append(len(train_loss))

    best_candidate = best_payload.get("candidate") or {}
    best_metrics = best_candidate.get("metrics") or {}
    best_history = best_payload.get("history") or {}

    if method in {"random", "rl"}:
        if not epoch_lengths:
            raise ValueError(f"No per-candidate training history found for {run_dir}")
        progress_steps = list(range(1, len(accuracies) + 1))
        progress_best_acc = _running_best(accuracies)
        budget_steps: list[int] = []
        budget_accuracies: list[float] = []
        consumed_epochs = 0
        for payload in jsonl_records:
            history = payload.get("history") or {}
            candidate = payload.get("candidate") or {}
            train_loss = history.get("train_loss")
            if not isinstance(train_loss, list):
                continue
            consumed_epochs += len(train_loss)
            budget_steps.append(consumed_epochs)
            budget_accuracies.append(float(((candidate.get("metrics") or {}).get("accuracy")) or 0.0))
        budget_best_acc = _running_best(budget_accuracies)
        round_summary = RoundSummary(
            type="per_candidate",
            configured_eval_epochs=int(search_cfg.get("eval_epochs", 0)),
            actual_min_epochs=min(epoch_lengths),
            actual_avg_epochs=float(statistics.mean(epoch_lengths)),
            actual_median_epochs=float(statistics.median(epoch_lengths)),
            actual_max_epochs=max(epoch_lengths),
            best_candidate_epochs=len(best_history.get("train_loss") or []),
        )
    elif method == "proxyless":
        epoch_history = best_history.get("epochs") or []
        if not isinstance(epoch_history, list) or not epoch_history:
            raise ValueError(f"No proxyless epoch history found for {run_dir}")
        progress_steps = [int(item.get("epoch", index + 1)) for index, item in enumerate(epoch_history)]
        progress_best_acc = _running_best(
            [float(item.get("val_acc") or 0.0) for item in epoch_history]
        )
        warmup_epochs = int(proxyless_cfg.get("warmup_epochs", 0))
        search_epochs = int(proxyless_cfg.get("search_epochs", search_cfg.get("eval_epochs", 0)))
        budget_steps = []
        budget_accuracies = []
        for payload in jsonl_records:
            history = payload.get("history") or {}
            epoch_summary = history.get("epoch_summary") or {}
            candidate = payload.get("candidate") or {}
            epoch_index = int(epoch_summary.get("epoch", len(budget_steps) + 1))
            budget_steps.append(warmup_epochs + epoch_index)
            budget_accuracies.append(float(((candidate.get("metrics") or {}).get("accuracy")) or 0.0))
        budget_best_acc = _running_best(budget_accuracies)
        round_summary = RoundSummary(
            type="search_epochs",
            configured_eval_epochs=int(search_cfg.get("eval_epochs", 0)),
            configured_warmup_epochs=warmup_epochs,
            configured_search_epochs=search_epochs,
            configured_total_epochs=warmup_epochs + search_epochs,
            actual_search_epochs=len(epoch_history),
            actual_total_epochs=warmup_epochs + len(epoch_history),
            best_candidate_epochs=len(epoch_history),
        )
    else:
        raise ValueError(f"Unsupported search method in {run_dir}: {method}")

    return ExperimentRecord(
        run_name=run_dir.name,
        label=label,
        path=str(run_dir),
        method=method,
        status=str(summary.get("status", "unknown")),
        total_evaluated=int(summary.get("total_evaluated", 0)),
        feasible=int(summary.get("feasible", 0)),
        infeasible=int(summary.get("infeasible", 0)),
        best_arch=str(best_candidate.get("arch_id", "n/a")),
        best_accuracy=float(best_metrics.get("accuracy") or 0.0),
        best_latency_ms=float(best_metrics.get("latency_ms") or 0.0),
        best_dsp=int(best_metrics.get("dsp") or 0),
        best_lut=int(best_metrics.get("lut") or 0),
        accuracies=accuracies,
        latencies=latencies,
        progress_steps=progress_steps,
        progress_best_acc=progress_best_acc,
        budget_steps=budget_steps,
        budget_best_acc=budget_best_acc,
        total_epoch_budget=budget_steps[-1],
        actual_epoch_lengths=epoch_lengths,
        round_summary=round_summary,
    )


def save_summary(records: list[ExperimentRecord], output_dir: Path) -> Path:
    output_path = output_dir / "training_rounds_summary.json"
    payload = []
    for record in records:
        item = {
            "run_name": record.run_name,
            "label": record.label,
            "path": record.path,
            "method": record.method,
            "status": record.status,
            "total_evaluated": record.total_evaluated,
            "feasible": record.feasible,
            "infeasible": record.infeasible,
            "best_arch": record.best_arch,
            "best_accuracy": record.best_accuracy,
            "best_latency_ms": record.best_latency_ms,
            "best_dsp": record.best_dsp,
            "best_lut": record.best_lut,
            "total_epoch_budget": record.total_epoch_budget,
            "round_summary": asdict(record.round_summary),
        }
        payload.append(item)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def plot_overview(records: list[ExperimentRecord], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Search Strategy Selection Overview", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    for record in records:
        ax.plot(
            record.progress_steps,
            [value * 100.0 for value in record.progress_best_acc],
            linewidth=2.2,
            label=record.label,
            color=RUN_COLORS.get(record.label, COLOR_OVERRIDES.get(record.method, "#777777")),
        )
    ax.set_title("Running Best Accuracy")
    ax.set_xlabel("Search step (candidate or search epoch)")
    ax.set_ylabel("Best accuracy so far (%)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for record in records:
        color = RUN_COLORS.get(record.label, COLOR_OVERRIDES.get(record.method, "#777777"))
        ax.scatter(
            record.latencies,
            [value * 100.0 for value in record.accuracies],
            s=28,
            alpha=0.50,
            color=color,
            label=record.label,
        )
        ax.scatter(
            [record.best_latency_ms],
            [record.best_accuracy * 100.0],
            s=140,
            color=color,
            edgecolors="black",
            linewidths=0.8,
        )
    ax.set_title("Accuracy vs Latency")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    labels = [record.label for record in records]
    values = [record.best_accuracy * 100.0 for record in records]
    colors = [RUN_COLORS.get(label, "#777777") for label in labels]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("Best Candidate Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.4,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1, 1]
    ax.axis("off")
    text_lines = ["Training rounds summary", ""]
    for record in records:
        text_lines.append(_format_round_summary(record))
    ax.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    output_path = output_dir / "strategy_selection_overview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_round_details(records: list[ExperimentRecord], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Round Details", fontsize=14, fontweight="bold")

    random_rl_records = [record for record in records if record.method in {"random", "rl"}]
    ax = axes[0]
    if random_rl_records:
        box_data = [record.actual_epoch_lengths for record in random_rl_records]
        box_labels = [record.label for record in random_rl_records]
        box = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels)
        for patch, record in zip(box["boxes"], random_rl_records):
            patch.set_facecolor(RUN_COLORS.get(record.label, "#777777"))
            patch.set_alpha(0.75)
        ax.set_title("Actual Epochs per Evaluated Candidate")
        ax.set_ylabel("Epochs")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No random/RL runs found.", ha="center", va="center")

    proxyless_records = [record for record in records if record.method == "proxyless"]
    ax = axes[1]
    if proxyless_records:
        labels = [record.label for record in proxyless_records]
        warmup = [record.round_summary.configured_warmup_epochs or 0 for record in proxyless_records]
        search = [record.round_summary.actual_search_epochs or 0 for record in proxyless_records]
        x_positions = list(range(len(proxyless_records)))
        ax.bar(x_positions, warmup, color="#B8B8B8", label="Warmup epochs")
        ax.bar(
            x_positions,
            search,
            bottom=warmup,
            color=[RUN_COLORS.get(label, "#777777") for label in labels],
            label="Search epochs",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel("Epochs")
        ax.set_title("Proxyless Warmup + Search Epochs")
        ax.legend(fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No proxyless runs found.", ha="center", va="center")

    fig.tight_layout()
    output_path = output_dir / "strategy_selection_round_details.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _best_accuracy_within_budget(record: ExperimentRecord, epoch_budget: int) -> float:
    best_accuracy = 0.0
    for step, accuracy in zip(record.budget_steps, record.budget_best_acc):
        if step <= epoch_budget:
            best_accuracy = float(accuracy)
        else:
            break
    return best_accuracy


def save_equal_budget_summary(records: list[ExperimentRecord], output_dir: Path) -> tuple[Path, int]:
    common_budget = min(record.total_epoch_budget for record in records)
    payload = {
        "comparison_budget_epochs": common_budget,
        "results": [
            {
                "label": record.label,
                "method": record.method,
                "total_epoch_budget": record.total_epoch_budget,
                "best_accuracy_within_budget": _best_accuracy_within_budget(record, common_budget),
                "final_best_accuracy": record.best_accuracy,
            }
            for record in records
        ],
    }
    output_path = output_dir / "equal_budget_summary.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, common_budget


def plot_equal_budget(records: list[ExperimentRecord], output_dir: Path) -> tuple[Path, int]:
    common_budget = min(record.total_epoch_budget for record in records)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        f"Equal-Budget Comparison (same total training budget = {common_budget} epochs)",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    for record in records:
        color = RUN_COLORS.get(record.label, COLOR_OVERRIDES.get(record.method, "#777777"))
        ax.step(
            record.budget_steps,
            [value * 100.0 for value in record.budget_best_acc],
            where="post",
            linewidth=2.2,
            label=f"{record.label} (total {record.total_epoch_budget})",
            color=color,
        )
    ax.axvline(common_budget, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Consumed training epochs")
    ax.set_ylabel("Running best accuracy (%)")
    ax.set_title("Running Best Accuracy under Equal Budget")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [record.label for record in records]
    values = [_best_accuracy_within_budget(record, common_budget) * 100.0 for record in records]
    colors = [RUN_COLORS.get(label, "#777777") for label in labels]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Best accuracy within equal budget (%)")
    ax.set_title(f"Best Accuracy within {common_budget} Epoch Budget")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.4,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path = output_dir / "strategy_selection_equal_budget.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, common_budget


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [load_experiment(Path(run_dir).expanduser().resolve()) for run_dir in args.run_dirs]

    summary_path = save_summary(records, output_dir)
    overview_path = plot_overview(records, output_dir)
    details_path = plot_round_details(records, output_dir)
    equal_budget_summary_path, common_budget = save_equal_budget_summary(records, output_dir)
    equal_budget_path, _ = plot_equal_budget(records, output_dir)

    print("Search strategy training-round summary:")
    for record in records:
        print(f"- {_format_round_summary(record)}")
    print(f"Equal-budget comparison budget: {common_budget} epochs")
    for record in records:
        best_within_budget = _best_accuracy_within_budget(record, common_budget)
        print(
            f"- {record.label}: best within {common_budget} epochs = "
            f"{best_within_budget:.4f} (final best = {record.best_accuracy:.4f})"
        )
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {overview_path}")
    print(f"Saved figure: {details_path}")
    print(f"Saved summary: {equal_budget_summary_path}")
    print(f"Saved figure: {equal_budget_path}")


if __name__ == "__main__":
    main()
