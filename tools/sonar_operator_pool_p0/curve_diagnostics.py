"""Read-only diagnostics of the three frozen D learning curves.

No model, dataset, optimizer, or checkpoint is loaded. Epochs are temporally
dependent observations: all summaries are descriptive, not significance tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


METRICS = ("train_loss", "valid_loss", "valid_mAP50_95")


def first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def number(value: Any, location: str, *, required: bool = False) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"Missing required metric: {location}")
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {location}={result}")
    return result


def normalize_history(path: Path, seed: int) -> list[dict[str, Any]]:
    """Normalize formal_runner and P0 histories without inventing missing data."""
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8-sig"))
    if isinstance(payload, dict):
        payload = first_value(payload.get("history"), payload.get("rows"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected non-empty history list: {path}")
    digest = hashlib.sha256(content).hexdigest()
    normalized = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError(f"History row must be a dictionary: {path}")
        epoch = raw.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError(f"Expected integer epoch in {path}: {epoch!r}")
        valid = raw.get("valid", {})
        parts = raw.get("losses", raw.get("loss_parts", {}))
        if not isinstance(valid, dict) or not isinstance(parts, dict):
            raise ValueError(f"Invalid valid/losses structure at {path}:{epoch}")
        where = f"{path.name}:epoch{epoch}"
        row = dict(
            architecture="D", seed=seed, epoch=epoch,
            history_path=str(path.resolve()), history_sha256=digest,
            train_loss=number(first_value(raw.get("train_loss"), parts.get("loss")),
                              where + ":train_loss", required=True),
            valid_loss=number(valid.get("validation_loss"), where + ":valid_loss"),
            valid_mAP50_95=number(first_value(valid.get("mAP50_95"), valid.get("mAP50:95")),
                                   where + ":valid_mAP50_95", required=True),
            valid_AP50=number(first_value(valid.get("AP50"), valid.get("mAP50")), where + ":AP50"),
            valid_AP75=number(valid.get("AP75"), where + ":AP75"),
            valid_Precision=number(valid.get("Precision"), where + ":Precision"),
            valid_Recall=number(valid.get("Recall"), where + ":Recall"),
            wall_seconds=number(first_value(raw.get("wall_seconds"), raw.get("epoch_seconds")),
                                where + ":wall_seconds"),
            train_elapsed_seconds=number(raw.get("elapsed_seconds"), where + ":elapsed_seconds"),
        )
        for part in ("box_loss", "object_loss", "class_loss"):
            row["train_" + part] = number(parts.get(part), where + ":" + part)
        if not 0 <= row["valid_mAP50_95"] <= 1:
            raise ValueError(f"Expected mAP as fraction in [0,1], got {where}")
        normalized.append(row)
    if [row["epoch"] for row in normalized] != list(range(1, 81)):
        raise ValueError(f"History must contain exactly ordered epochs 1..80: {path}")
    return normalized


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [row[key] for row in rows if row.get(key) is not None]


def descriptive(values_: list[float]) -> dict[str, float | int | None]:
    return dict(n=len(values_), mean=mean(values_) if values_ else None,
                sample_std=stdev(values_) if len(values_) > 1 else None,
                median=median(values_) if values_ else None,
                minimum=min(values_) if values_ else None,
                maximum=max(values_) if values_ else None)


def linear_slope(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = [(row["epoch"], row[key]) for row in rows if row.get(key) is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in pairs) / sum((x - mx) ** 2 for x in xs)


def summarize_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use predeclared 41..60 and 61..80 windows; never pick favorable windows."""
    window_a = [row for row in rows if 41 <= row["epoch"] <= 60]
    window_b = [row for row in rows if 61 <= row["epoch"] <= 80]
    last10 = [row for row in rows if 71 <= row["epoch"] <= 80]
    best = max(rows, key=lambda row: row["valid_mAP50_95"])
    end = rows[-1]
    summary = dict(seed=rows[0]["seed"], n_epochs=len(rows), best_epoch=best["epoch"],
                   best_mAP50_95=best["valid_mAP50_95"],
                   endpoint_mAP50_95=end["valid_mAP50_95"],
                   best_endpoint_gap=best["valid_mAP50_95"] - end["valid_mAP50_95"])
    for key in METRICS:
        for label, window in (("epoch41_60", window_a), ("epoch61_80", window_b), ("epoch71_80", last10)):
            for stat, value in descriptive(values(window, key)).items():
                summary[f"{label}_{key}_{stat}"] = value
        early = summary[f"epoch41_60_{key}_mean"]
        late = summary[f"epoch61_80_{key}_mean"]
        summary[f"late_minus_earlier_{key}_mean"] = late - early if late is not None and early is not None else None
        summary[f"epoch61_80_{key}_slope_per_epoch"] = linear_slope(window_b, key)
    summary["best_last10median_gap"] = best["valid_mAP50_95"] - median(values(last10, "valid_mAP50_95"))
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty analysis CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    labels = ("Training loss (augmented / sampled)", "Validation loss (fixed evaluation)",
              "Validation mAP50:95 (fraction)")
    seeds = sorted({row["seed"] for row in rows})
    for ax, key, label in zip(axes, METRICS, labels):
        for seed in seeds:
            seed_rows = [row for row in rows if row["seed"] == seed and row.get(key) is not None]
            if seed_rows:
                ax.plot([row["epoch"] for row in seed_rows], [row[key] for row in seed_rows],
                        label=f"seed {seed}", linewidth=1.2, alpha=0.85)
        if not values(rows, key):
            ax.text(0.5, 0.5, "Not recorded in source histories", ha="center", transform=ax.transAxes)
        ax.axvspan(40.5, 60.5, alpha=0.08, color="gray")
        ax.axvspan(60.5, 80.5, alpha=0.08, color="orange")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.set_xlim(1, 80)
    axes[0].legend(loc="upper right", ncol=len(seeds))
    axes[0].set_title("Frozen D: existing 80-epoch histories (no new training)")
    axes[-1].set_xlabel("Epoch; shaded windows: 41-60 and 61-80")
    figure.savefig(output / "learning_curves.png", dpi=180)
    figure.savefig(output / "learning_curves.svg")
    plt.close(figure)
    write_csv(output / "learning_curves.csv", rows)


def analyze(input_dir: Path, output_dir: Path, seeds: list[int]) -> dict[str, Any]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    histories = input_dir / "histories" if (input_dir / "histories").is_dir() else input_dir
    rows: list[dict[str, Any]] = []
    summaries = []
    bindings = []
    for seed in seeds:
        source = histories / f"D_seed{seed}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen D history: {source}")
        history = normalize_history(source, seed)
        rows.extend(history)
        summaries.append(summarize_seed(history))
        bindings.append(dict(seed=seed, path=str(source.resolve()), sha256=history[0]["history_sha256"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    write_csv(output_dir / "training_curves.csv", rows)
    write_csv(output_dir / "late_training_summary.csv", summaries)
    across = [dict(metric=key, **descriptive(values(summaries, key)))
              for key in summaries[0] if key not in ("seed", "n_epochs")]
    write_csv(output_dir / "across_seed_curve_summary.csv", across)
    plot_curves(rows, figures)
    metadata = dict(history_bindings=bindings, seeds=seeds, curve_rows=len(rows),
                    epochs_per_seed=80, test_accessed=False, new_training=False,
                    window_a=[41, 60], window_b=[61, 80], last10_window=[71, 80],
                    missing_valid_loss_rows=sum(row["valid_loss"] is None for row in rows),
                    inference="descriptive_only_no_epoch_independence_assumption",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir / "CURVE_DIAGNOSTICS_MANIFEST.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# D learning-curve diagnostics", "",
             "Existing frozen histories only; no new training or test access.", "",
             "| Seed | Best epoch | Best mAP | Epoch80 mAP | Gap | Last10 median | Last10 temporal SD |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        lines.append(f"| {item['seed']} | {item['best_epoch']} | {item['best_mAP50_95']:.6f} | "
                     f"{item['endpoint_mAP50_95']:.6f} | {item['best_endpoint_gap']:.6f} | "
                     f"{item['epoch71_80_valid_mAP50_95_median']:.6f} | "
                     f"{item['epoch71_80_valid_mAP50_95_sample_std']:.6f} |")
    lines += ["", "## Interpretation boundaries", "",
              "- A best-to-endpoint decrease alone does not prove overfitting: the best epoch is selected as a maximum.",
              "- Compare the fixed 41-60 and 61-80 windows and the 61-80 descriptive slopes; no window was selected using its outcome.",
              "- The last-ten-epoch standard deviation describes temporal fluctuation, not a confidence interval from independent samples.",
              "- Across-seed summaries use the seed as the unit. n=3 is small; no cross-epoch significance test is performed.",
              "- Training loss is measured during weight updates with the training sampler/augmentation; validation loss is evaluated separately. Their absolute gap alone is not diagnostic.",
              "- These histories cannot by themselves distinguish overfitting, batch-normalization instability, stochastic optimization, or validation sampling uncertainty.",
              "", "## Figure", "",
              "`figures/learning_curves.png` / `.svg` plot the unmodified per-epoch train loss, valid loss and valid mAP for each seed.",
              "The corresponding data are `figures/learning_curves.csv` (also `training_curves.csv`).",
              "Use this figure to inspect sustained trends versus isolated peaks; fixed shaded windows match `late_training_summary.csv`.",
              f"Validation loss missing in {metadata['missing_valid_loss_rows']} of {len(rows)} rows; missing values are never imputed.", ""]
    (output_dir / "LEARNING_CURVE_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing histories/D_seed{42,43,44}.json (or these files directly)")
    parser.add_argument("--output", type=Path, required=True, help="Generated CSV/figure/report directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    metadata = analyze(args.input, args.output, args.seeds)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
