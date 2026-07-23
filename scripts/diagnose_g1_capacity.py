#!/usr/bin/env python3
"""Re-audit the G1 NAS-vs-scratch training-curve mechanism.

This script is deliberately read-only with respect to experiment records.  It
parses complete per-run curves, keeps fold/seed pairing, and labels the logged
``train_acc`` correctly: it is an online, training-mode accuracy measured on
randomly augmented batches.  It can diagnose underfitting *under the frozen
recipe*, but it cannot by itself isolate expressive capacity from optimisation,
regularisation, augmentation sensitivity, or train/eval-mode behaviour.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"===\s*fold\s+(\d+)\s+seed\s+(\d+)\s*===", re.IGNORECASE)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+train_loss=([\d.eE+-]+)\s+"
    r"train_acc=([\d.eE+-]+)\s+inner_macro_f1=([\d.eE+-]+)"
)

METHODS = {
    "nas_rl_arch_135": {
        "records": "results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected",
        "log": "results/protocol/g1_rl_arch_135_legacy_selected.launcher.log",
        "role": "primary",
    },
    "mnv2_scratch": {
        "records": "results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2",
        "log": "logs/g1_clean_20260718/scratch_v2_stdout.log",
        "role": "primary",
    },
    "mnv2_pretrained": {
        "records": "results/protocol/g1_clean_20260711/g1_mobilenet_v2_grayscale_imagenet",
        "log": "results/protocol/g1_mobilenet_v2_grayscale_imagenet.launcher.log",
        "role": "context_only_incomplete_curve_log",
    },
}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mixed_encoding(path: Path) -> str:
    """Decode launcher logs containing UTF-8 headers and UTF-16-like NULs."""
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")


def parse_log(path: Path) -> dict:
    curves: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    current: tuple[int, int] | None = None
    orphan_epoch_lines = 0
    duplicate_epochs = 0
    for line in read_mixed_encoding(path).splitlines():
        header = RUN_RE.search(line)
        if header:
            current = (int(header.group(1)), int(header.group(2)))
            continue
        epoch = EPOCH_RE.search(line)
        if not epoch:
            continue
        if current is None:
            orphan_epoch_lines += 1
            continue
        epoch_number = int(epoch.group(1))
        if epoch_number in curves[current]:
            duplicate_epochs += 1
        curves[current][epoch_number] = {
            "epoch": epoch_number,
            "epoch_cap": int(epoch.group(2)),
            "train_loss": float(epoch.group(3)),
            "train_acc": float(epoch.group(4)),
            "inner_macro_f1": float(epoch.group(5)),
        }

    runs = {}
    for key, by_epoch in sorted(curves.items()):
        rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
        caps = sorted({row["epoch_cap"] for row in rows})
        expected = caps[-1] if len(caps) == 1 else None
        complete = expected is not None and sorted(by_epoch) == list(range(1, expected + 1))
        late = [row for row in rows if 140 <= row["epoch"] <= 150]
        runs[f"fold{key[0]}_seed{key[1]}"] = {
            "fold": key[0],
            "seed": key[1],
            "n_epoch_lines": len(rows),
            "epoch_caps": caps,
            "first_epoch": rows[0]["epoch"] if rows else None,
            "last_epoch": rows[-1]["epoch"] if rows else None,
            "complete": complete,
            "late_ep_140_150": summarize_rows(late),
            "curve": rows,
        }
    return {
        "source": relative(path),
        "sha256": sha256(path),
        "n_run_headers": len(runs),
        "n_complete_runs": sum(run["complete"] for run in runs.values()),
        "n_epoch_lines": sum(run["n_epoch_lines"] for run in runs.values()),
        "orphan_epoch_lines": orphan_epoch_lines,
        "duplicate_epochs_overwritten": duplicate_epochs,
        "runs": runs,
    }


def summarize_rows(rows: Iterable[dict]) -> dict | None:
    rows = list(rows)
    if not rows:
        return None
    return {
        "n_epochs": len(rows),
        "train_acc": statistics.fmean(row["train_acc"] for row in rows),
        "train_loss": statistics.fmean(row["train_loss"] for row in rows),
        "inner_macro_f1": statistics.fmean(row["inner_macro_f1"] for row in rows),
    }


def load_records(path: Path) -> dict:
    records = []
    for record_path in sorted(path.glob("run_fold*_seed*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        match = re.search(r"run_fold(\d+)_seed(\d+)\.json$", record_path.name)
        if match is None:
            continue
        records.append(
            {
                "fold": int(match.group(1)),
                "seed": int(match.group(2)),
                "best_epoch": int(record["best_epoch"]),
                "inner_macro_f1": float(record["inner_val"]["macro_f1"]),
                "outer_macro_f1": float(record["outer_val"]["macro_f1"]),
            }
        )
    best_epochs = [record["best_epoch"] for record in records]
    return {
        "source": relative(path),
        "n_runs": len(records),
        "best_epoch_median": statistics.median(best_epochs) if best_epochs else None,
        "best_epoch_sorted": sorted(best_epochs),
        "best_epoch_exactly_150": sum(epoch == 150 for epoch in best_epochs),
        "best_epoch_145_150": sum(145 <= epoch <= 150 for epoch in best_epochs),
        "inner_macro_f1_mean": (
            statistics.fmean(record["inner_macro_f1"] for record in records)
            if records
            else None
        ),
        "outer_macro_f1_mean": (
            statistics.fmean(record["outer_macro_f1"] for record in records)
            if records
            else None
        ),
        "runs": records,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260722) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += statistic >= observed - 1e-15
        total += 1
    return exceed / total


def primary_pair_stats(methods: dict) -> dict:
    nas_runs = methods["nas_rl_arch_135"]["training_curve"]["runs"]
    scratch_runs = methods["mnv2_scratch"]["training_curve"]["runs"]
    paired_keys = sorted(set(nas_runs) & set(scratch_runs))
    usable = [
        key
        for key in paired_keys
        if nas_runs[key]["complete"]
        and scratch_runs[key]["complete"]
        and nas_runs[key]["late_ep_140_150"]
        and scratch_runs[key]["late_ep_140_150"]
    ]
    nas = np.asarray(
        [nas_runs[key]["late_ep_140_150"]["train_acc"] for key in usable], dtype=float
    )
    scratch = np.asarray(
        [scratch_runs[key]["late_ep_140_150"]["train_acc"] for key in usable], dtype=float
    )
    delta = scratch - nas
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
    return {
        "metric": "mean online augmented training-mode train_acc over epochs 140-150",
        "paired_run_keys": usable,
        "n_pairs": len(usable),
        "nas_mean": float(nas.mean()),
        "scratch_mean": float(scratch.mean()),
        "paired_delta_scratch_minus_nas": {
            "mean": float(delta.mean()),
            "bootstrap_95_ci": bootstrap_mean_ci(delta),
            "paired_cohens_dz": float(delta.mean() / sd) if sd > 0 else None,
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(delta),
            "per_pair": [
                {
                    "run": key,
                    "nas": float(nas[index]),
                    "scratch": float(scratch[index]),
                    "delta": float(delta[index]),
                }
                for index, key in enumerate(usable)
            ],
        },
    }


def epoch_matrix(log: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    complete = [run for run in log["runs"].values() if run["complete"]]
    epochs = np.arange(1, 151)
    matrix = np.asarray(
        [
            [next(row[metric] for row in run["curve"] if row["epoch"] == epoch) for epoch in epochs]
            for run in complete
        ],
        dtype=float,
    )
    return epochs, matrix


def make_figures(payload: dict, output_dir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    colors = {"nas_rl_arch_135": "#D55E00", "mnv2_scratch": "#0072B2"}
    labels = {"nas_rl_arch_135": "rl_arch_135", "mnv2_scratch": "MNV2 scratch"}
    for method in ("nas_rl_arch_135", "mnv2_scratch"):
        log = payload["methods"][method]["training_curve"]
        epochs, acc = epoch_matrix(log, "train_acc")
        _, f1 = epoch_matrix(log, "inner_macro_f1")
        for axis, matrix, ylabel in (
            (axes[0], acc, "Online augmented train accuracy"),
            (axes[1], f1, "Inner-validation macro F1"),
        ):
            mean = matrix.mean(axis=0)
            half = 2.1448 * matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
            axis.plot(epochs, mean, color=colors[method], label=labels[method], linewidth=1.8)
            axis.fill_between(epochs, mean - half, mean + half, color=colors[method], alpha=0.16)
            axis.set(xlabel="Epoch", ylabel=ylabel, xlim=(1, 150), ylim=(0, 1.03))
            axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right")
    figure1 = figures_dir / "figure-01-online-training-dynamics.png"
    fig.savefig(figure1, dpi=220)
    plt.close(fig)

    pairs = payload["primary_paired_analysis"]["paired_delta_scratch_minus_nas"]["per_pair"]
    fig, ax = plt.subplots(figsize=(6.2, 4.5), constrained_layout=True)
    for index, pair in enumerate(pairs):
        ax.plot([0, 1], [pair["nas"], pair["scratch"]], color="#999999", alpha=0.55, linewidth=0.9)
        ax.scatter([0, 1], [pair["nas"], pair["scratch"]], color=["#D55E00", "#0072B2"], s=24)
    ax.set(
        xticks=[0, 1],
        xticklabels=["rl_arch_135", "MNV2 scratch"],
        ylabel="Epoch 140-150 mean online augmented train accuracy",
        ylim=(0, 1.03),
    )
    ax.grid(axis="y", alpha=0.2)
    figure2 = figures_dir / "figure-02-late-online-train-accuracy-paired.png"
    fig.savefig(figure2, dpi=220)
    plt.close(fig)
    return [
        {
            "id": "Figure 1",
            "path": relative(figure1),
            "caption": (
                "Mean learning dynamics with pointwise 95% t intervals across 15 complete "
                "fold/seed runs. Training accuracy is online and augmentation-affected."
            ),
        },
        {
            "id": "Figure 2",
            "path": relative(figure2),
            "caption": (
                "Paired late-epoch online augmented training accuracy for the same 15 "
                "fold/seed runs. Lines preserve the pairing."
            ),
        },
    ]


def write_bundle(payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = make_figures(payload, output_dir)
    stats = payload["primary_paired_analysis"]
    delta = stats["paired_delta_scratch_minus_nas"]
    report = f"""# G1 capacity attribution re-audit

## Result

The frozen `rl_arch_135` candidate shows a large **underfitting signal under the current
recipe**, but the existing curves do **not** uniquely identify parameter capacity as the
mechanism. Across {stats['n_pairs']} complete paired fold/seed runs, its mean online,
augmentation-affected training accuracy over epochs 140-150 was {stats['nas_mean']:.4f},
versus {stats['scratch_mean']:.4f} for scratch MobileNetV2 (paired delta
{delta['mean']:+.4f}, bootstrap 95% CI [{delta['bootstrap_95_ci'][0]:.4f},
{delta['bootstrap_95_ci'][1]:.4f}]).

## What this establishes

- The gap is present during training under the frozen recipe, not only on outer validation.
- The classic pattern “near-perfect training accuracy plus weak validation” is absent from
  this logged metric.
- The NAS inner-validation optimum is usually not pinned to epoch 150, so blindly extending
  the same schedule is not the highest-information next action.

## What this does not establish

`train_acc` is computed in `model.train()` on randomly flipped, rotated, affinely transformed,
brightness/contrast-jittered, and sometimes speckle-corrupted batches. It is therefore not a
deterministic accuracy on the unaugmented training set. A shared recipe also does not control
architecture-specific optimisation difficulty. Capacity, optimiser/schedule mismatch,
regularisation strength, augmentation sensitivity, and train/eval-mode behaviour remain
confounded.

## Decision

Do not launch a full capacity sweep or distillation comparison yet. First run an inference-only
evaluation of every saved best checkpoint on its own deterministic, no-augmentation training
indices. If the large gap remains, run one micro-overfit/LR triage before selecting the capacity
sweep. The four-arm preprocessing campaign stays queued but unlaunched until a source-grouped
split contract exists.
"""
    appendix = f"""# Statistical appendix

## Analysis unit and pairing

- Unit: one `(outer fold, seed)` run.
- Primary methods: `rl_arch_135` and the formal scratch-v2 MobileNetV2 rerun.
- Included pairs: {stats['n_pairs']} complete 150-epoch curves with the same fold/seed key.
- Late window: arithmetic mean over epochs 140-150 within each run; inference is across runs,
  not across epoch lines.

## Primary result

| Quantity | Estimate |
|---|---:|
| NAS mean | {stats['nas_mean']:.6f} |
| Scratch MNV2 mean | {stats['scratch_mean']:.6f} |
| Paired mean delta (scratch − NAS) | {delta['mean']:.6f} |
| Paired bootstrap 95% CI | [{delta['bootstrap_95_ci'][0]:.6f}, {delta['bootstrap_95_ci'][1]:.6f}] |
| Paired Cohen's dz | {delta['paired_cohens_dz']:.3f} |
| Exact two-sided sign-flip p | {delta['exact_two_sided_sign_flip_p']:.8f} |

The exact p-value is descriptive mechanism-triage evidence, not a license to claim that capacity
has been isolated. The bootstrap uses 50,000 paired resamples with seed 20260722.

## Coverage and exclusions

The primary NAS and scratch-v2 logs each contain 15 complete × 150-epoch curves. The historical
root pretrained log is retained as context only because it does not contain a balanced complete
15-run curve set. It is excluded from primary curve inference.
"""
    catalog_lines = ["# Figure catalog", ""]
    for figure in figures:
        catalog_lines.extend([f"## {figure['id']}", "", f"- File: `{figure['path']}`", f"- {figure['caption']}", ""])
    (output_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    (output_dir / "stats-appendix.md").write_text(appendix, encoding="utf-8")
    (output_dir / "figure-catalog.md").write_text("\n".join(catalog_lines), encoding="utf-8")


def compact_for_record(payload: dict) -> dict:
    """Drop duplicated epoch rows while retaining source hashes and run summaries."""
    compact = copy.deepcopy(payload)
    for method in compact["methods"].values():
        for run in method["training_curve"]["runs"].values():
            run.pop("curve", None)
    compact["record_compaction"] = {
        "omitted": "per-epoch rows duplicated from launcher logs",
        "reproduction": "rerun python scripts/diagnose_g1_capacity.py",
        "integrity": "each source log path and SHA256 is retained under methods.*.training_curve",
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/g1_capacity_diagnostic.json",
    )
    parser.add_argument(
        "--bundle-dir",
        default=(
            "artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/"
            "g1_capacity_reaudit_v2"
        ),
    )
    args = parser.parse_args()

    methods = {}
    for name, spec in METHODS.items():
        records_path = resolve(spec["records"])
        log_path = resolve(spec["log"])
        if not records_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Missing evidence for {name}: {records_path} / {log_path}")
        methods[name] = {
            "role": spec["role"],
            "records": load_records(records_path),
            "training_curve": parse_log(log_path),
        }

    primary = primary_pair_stats(methods)
    payload = {
        "schema_version": 2,
        "diagnostic": "G1 current-recipe underfit signal; mechanism re-audit",
        "method": "read-only paired analysis of existing per-run records and launcher logs",
        "metric_semantics": (
            "Logged train_acc is online training-mode accuracy on randomly augmented batches; "
            "it is not clean eval-mode training-set accuracy."
        ),
        "methods": methods,
        "primary_paired_analysis": primary,
        "verdict": {
            "status": "CURRENT_RECIPE_UNDERFIT_SIGNAL__MECHANISM_UNRESOLVED",
            "supported": [
                "rl_arch_135 has a large training-side gap versus scratch MobileNetV2 under the frozen recipe",
                "classic high-train/low-validation overfitting is not visible in the logged metric",
                "inner-validation best epochs are usually not pinned to the 150-epoch cap",
            ],
            "not_isolated": [
                "parameter or expressive capacity",
                "optimiser or schedule compatibility",
                "regularisation and augmentation strength",
                "train-mode versus eval-mode behaviour",
            ],
            "next_gate": (
                "deterministic no-augmentation eval-mode evaluation of each best checkpoint on "
                "its own split.train_indices, followed if needed by a micro-overfit/LR triage"
            ),
            "campaign_decision": (
                "Do not launch the full capacity sweep, distillation comparison, or four-arm "
                "preprocessing campaign until the mechanism and grouped-split gates close."
            ),
        },
    }

    output_path = resolve(args.output)
    bundle_dir = resolve(args.bundle_dir)
    write_bundle(payload, bundle_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(compact_for_record(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["verdict"], indent=2, ensure_ascii=False))
    print(f"written: {output_path}")
    print(f"bundle:  {bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
