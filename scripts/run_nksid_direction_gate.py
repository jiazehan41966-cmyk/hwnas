#!/usr/bin/env python3
"""Run the preregistered outer-train-only NKSID directional-basis gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.data.dataset import FrozenGeometryTransform, NKSIDDataset  # noqa: E402
from hwnas_fpga.data.protocol import build_protocol_split  # noqa: E402
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.models import build_model  # noqa: E402
from hwnas_fpga.search_space import ArchitectureSpec  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


EPSILON = 1e-12


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    extreme = 0
    total = 1 << array.size
    for mask in range(total):
        signs = np.fromiter(
            (1.0 if mask & (1 << index) else -1.0 for index in range(array.size)),
            dtype=float,
            count=array.size,
        )
        if abs(float(np.mean(array * signs))) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def bootstrap_fold_mean_ci(
    fold_values: Mapping[int, float],
    *,
    seed: int,
    iterations: int = 20_000,
) -> tuple[float, float]:
    folds = np.asarray(sorted(fold_values), dtype=int)
    values = np.asarray([fold_values[int(fold)] for fold in folds], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        chosen = rng.choice(values, size=values.size, replace=True)
        draws[index] = chosen.mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def pearson_permutation(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict[str, float]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.size != right.size or left.size < 3:
        raise ValueError("association requires equal arrays with at least 3 samples")
    if np.std(left) <= EPSILON or np.std(right) <= EPSILON:
        return {"r": 0.0, "permutation_p_two_sided": 1.0}
    observed = float(np.corrcoef(left, right)[0, 1])
    rng = np.random.default_rng(seed)
    extreme = 1
    for _ in range(iterations):
        permuted = rng.permutation(right)
        value = float(np.corrcoef(left, permuted)[0, 1])
        if abs(value) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "r": observed,
        "permutation_p_two_sided": extreme / (iterations + 1),
        "iterations": iterations,
    }


def class_permutation_test(
    values: Sequence[float],
    labels: Sequence[int],
    *,
    seed: int,
    iterations: int = 10_000,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    classes = np.asarray(labels, dtype=int)
    grand = float(array.mean())
    total_ss = float(np.sum((array - grand) ** 2))

    def between(current: np.ndarray) -> float:
        return float(
            sum(
                np.sum(current == label)
                * (float(array[current == label].mean()) - grand) ** 2
                for label in np.unique(current)
            )
        )

    observed = between(classes)
    rng = np.random.default_rng(seed)
    extreme = 1
    for _ in range(iterations):
        if between(rng.permutation(classes)) >= observed - 1e-15:
            extreme += 1
    return {
        "eta_squared": observed / total_ss if total_ss > EPSILON else 0.0,
        "permutation_p": extreme / (iterations + 1),
        "iterations": iterations,
    }


def geometry_values(contract: Mapping[str, Any], geometry: str) -> dict[str, Any]:
    prereg = contract["geometry_gate"]["candidates"][geometry]
    return {
        "image_size": 224,
        "mode": geometry,
        "fixed_scale_factor": (
            float(prereg["fixed_scale_factor"])
            if geometry == "fixed_scale_pad_224"
            else None
        ),
        "padding_value": int(prereg.get("padding_value_uint8", 0)),
    }


def sobel_features(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gx = (
        array[:-2, 2:]
        + 2.0 * array[1:-1, 2:]
        + array[2:, 2:]
        - array[:-2, :-2]
        - 2.0 * array[1:-1, :-2]
        - array[2:, :-2]
    )
    gy = (
        array[2:, :-2]
        + 2.0 * array[2:, 1:-1]
        + array[2:, 2:]
        - array[:-2, :-2]
        - 2.0 * array[:-2, 1:-1]
        - array[:-2, 2:]
    )
    eh = float(np.mean(np.abs(gx)))
    ev = float(np.mean(np.abs(gy)))
    signed = (eh - ev) / (eh + ev + EPSILON)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    angle = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
    coherence = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
        jxx + jyy + EPSILON
    )
    return {
        "Eh_mean_abs_Gx": eh,
        "Ev_mean_abs_Gy": ev,
        "signed_direction_ratio": signed,
        "unsigned_anisotropy": abs(signed),
        "structure_tensor_angle_rad_axial": angle,
        "structure_tensor_coherence": coherence,
    }


def macro_f1(targets: Sequence[int], predictions: Sequence[int]) -> float:
    scores = []
    for label in range(8):
        tp = sum(t == label and p == label for t, p in zip(targets, predictions))
        fp = sum(t != label and p == label for t, p in zip(targets, predictions))
        fn = sum(t == label and p != label for t, p in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return statistics.fmean(scores)


class RotatedInnerDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        indices: Sequence[int],
        transform: FrozenGeometryTransform,
        angle: int,
    ) -> None:
        self.samples = samples
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform
        self.angle = int(angle)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        path, label = self.samples[index]
        with Image.open(path) as handle:
            image = self.transform(handle.convert("L"))
        if self.angle:
            image = image.rotate(
                self.angle,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=0,
            )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, int(label), int(index)


@torch.no_grad()
def predict_model(model, loader, device: str):
    targets = []
    predictions = []
    indices = []
    model.eval()
    for inputs, labels, sample_indices in loader:
        outputs = model(inputs.to(device))
        targets.extend(int(value) for value in labels.tolist())
        predictions.extend(int(value) for value in outputs.argmax(1).cpu().tolist())
        indices.extend(int(value) for value in sample_indices.tolist())
    return targets, predictions, indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--prereg",
        default="configs/evaluation/nksid_protocol_v2_selection_preregistered.yaml",
    )
    parser.add_argument(
        "--protocol-selection-dir",
        default="results/sonar_fourstage_operator_v2/protocol_selection",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sonar_fourstage_operator_v2/direction_gate",
    )
    parser.add_argument(
        "--summary",
        default="artifacts/sonar_fourstage_operator_v2/direction_gate_summary.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    prereg_path = Path(args.prereg).resolve()
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    data_dir = Path(args.data_dir).resolve()
    selection_root = Path(args.protocol_selection_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_decision = json.loads(
        (selection_root / "bundle_decision.json").read_text(encoding="utf-8")
    )
    geometry_decision = json.loads(
        (selection_root / "geometry_decision.json").read_text(encoding="utf-8")
    )
    selected_bundle = str(bundle_decision["selected_bundle"])
    selected_geometry = str(geometry_decision["selected_geometry"])

    view = NKSIDDataset(
        data_dir=str(data_dir),
        image_size=224,
        is_training=False,
        use_kfold=False,
        split="full",
        augmentation_profile="none",
    )
    samples = list(view.samples)
    feature_rows = []
    feature_by_geometry: dict[str, dict[int, dict[str, float]]] = {}
    for geometry in prereg["direction_gate"]["geometries"]:
        transform = FrozenGeometryTransform(**geometry_values(prereg, geometry))
        geometry_features = {}
        for index, (path, label) in enumerate(samples):
            with Image.open(path) as handle:
                transformed = transform(handle.convert("L"))
            features = sobel_features(transformed)
            geometry_features[index] = features
            feature_rows.append(
                {
                    "sample_index": index,
                    "sample_path": str(Path(path).resolve()),
                    "label": int(label),
                    "geometry": geometry,
                    **features,
                }
            )
        feature_by_geometry[geometry] = geometry_features
    feature_path = output_dir / "direction_features.jsonl"
    with feature_path.open("w", encoding="utf-8") as handle:
        for row in feature_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    image_reports = {}
    signed_pvalues = {}
    class_pvalues = {}
    common_signs = []
    for geometry_index, geometry in enumerate(prereg["direction_gate"]["geometries"]):
        features = feature_by_geometry[geometry]
        fold_rows = {}
        fold_signed = {}
        fold_unsigned = {}
        all_outer_train = set()
        for fold in range(5):
            historical = build_protocol_split(
                samples,
                data_dir,
                fold_index=fold,
                seed=42,
                inner_val_fraction=0.15,
                num_classes=8,
            )
            indices = sorted(
                set(historical.train_indices) | set(historical.inner_val_indices)
            )
            all_outer_train.update(indices)
            signed = [features[index]["signed_direction_ratio"] for index in indices]
            unsigned = [features[index]["unsigned_anisotropy"] for index in indices]
            angles = [
                features[index]["structure_tensor_angle_rad_axial"]
                for index in indices
            ]
            resultant = math.sqrt(
                statistics.fmean(math.cos(2 * value) for value in angles) ** 2
                + statistics.fmean(math.sin(2 * value) for value in angles) ** 2
            )
            fold_signed[fold] = statistics.fmean(signed)
            fold_unsigned[fold] = statistics.fmean(unsigned)
            fold_rows[str(fold)] = {
                "sample_count": len(indices),
                "signed_direction_ratio_mean": fold_signed[fold],
                "unsigned_anisotropy_mean": fold_unsigned[fold],
                "structure_tensor_axial_resultant_length": resultant,
            }
        unsigned_ci = bootstrap_fold_mean_ci(
            fold_unsigned, seed=20260729 + geometry_index
        )
        unique_indices = sorted(all_outer_train)
        unique_signed = [
            features[index]["signed_direction_ratio"] for index in unique_indices
        ]
        labels = [int(samples[index][1]) for index in unique_indices]
        class_report = class_permutation_test(
            unique_signed,
            labels,
            seed=20260740 + geometry_index,
        )
        per_class = {}
        for label in range(8):
            values = [
                features[index]["signed_direction_ratio"]
                for index in unique_indices
                if int(samples[index][1]) == label
            ]
            per_class[str(label)] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        all_angles = [
            features[index]["structure_tensor_angle_rad_axial"]
            for index in unique_indices
        ]
        axial_resultant = math.sqrt(
            statistics.fmean(math.cos(2 * value) for value in all_angles) ** 2
            + statistics.fmean(math.sin(2 * value) for value in all_angles) ** 2
        )
        dominant_sign = int(np.sign(statistics.fmean(fold_signed.values())))
        consistent_folds = sum(
            int(np.sign(value)) == dominant_sign for value in fold_signed.values()
        )
        common_signs.append(dominant_sign)
        signed_pvalues[geometry] = exact_sign_flip_pvalue(list(fold_signed.values()))
        class_pvalues[geometry] = class_report["permutation_p"]
        image_reports[geometry] = {
            "folds": fold_rows,
            "signed_direction_ratio_fold_mean": statistics.fmean(fold_signed.values()),
            "signed_direction_exact_fold_sign_flip_p": signed_pvalues[geometry],
            "dominant_sign": dominant_sign,
            "consistent_fold_direction_count": consistent_folds,
            "unsigned_anisotropy_fold_mean": statistics.fmean(fold_unsigned.values()),
            "unsigned_anisotropy_fold_bootstrap_ci95": list(unsigned_ci),
            "structure_tensor_axial_resultant_length": axial_resultant,
            "per_class_signed_direction": per_class,
            "within_class_mean_std": statistics.fmean(
                row["std"] for row in per_class.values()
            ),
            "between_class": class_report,
        }
    signed_adjusted = holm_adjust(signed_pvalues)
    class_adjusted = holm_adjust(class_pvalues)
    for geometry in image_reports:
        image_reports[geometry]["signed_direction_holm_p"] = signed_adjusted[geometry]
        image_reports[geometry]["between_class"]["holm_p"] = class_adjusted[geometry]

    rotations = [int(value) for value in prereg["direction_gate"]["rotations_degrees"]]
    angles = [0] + rotations
    rotation_units = []
    sample_losses: dict[int, dict[int, list[float]]] = {
        angle: defaultdict(list) for angle in rotations
    }
    transform = FrozenGeometryTransform(
        **geometry_values(prereg, selected_geometry)
    )
    for fold in range(5):
        historical = build_protocol_split(
            samples,
            data_dir,
            fold_index=fold,
            seed=42,
            inner_val_fraction=0.15,
            num_classes=8,
        )
        outer_val = set(historical.outer_val_indices)
        for seed in (42, 43, 44):
            run_dir = (
                selection_root
                / "runs"
                / "historical_inner"
                / selected_geometry
                / selected_bundle
            )
            record_path = run_dir / f"run_fold{fold}_seed{seed}.json"
            checkpoint_path = run_dir / f"best_fold{fold}_seed{seed}.pt"
            prediction_path = run_dir / f"inner_predictions_fold{fold}_seed{seed}.jsonl"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["outer_validation_accessed"] is not False:
                raise AssertionError("direction gate checkpoint consumed outer validation")
            inner_indices = [
                int(json.loads(line)["sample_index"])
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if set(inner_indices) & outer_val:
                raise AssertionError("direction gate inner indices overlap outer validation")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            architecture = ArchitectureSpec.from_dict(checkpoint["architecture"])
            model = build_model(architecture=architecture, num_classes=8)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(args.device)
            angle_results = {}
            clean_correct = {}
            for angle in angles:
                loader = DataLoader(
                    RotatedInnerDataset(samples, inner_indices, transform, angle),
                    batch_size=32,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                targets, predictions, indices = predict_model(
                    model, loader, args.device
                )
                angle_results[str(angle)] = {
                    "macro_f1": macro_f1(targets, predictions),
                    "sample_count": len(targets),
                }
                correct = {
                    index: float(target == prediction)
                    for index, target, prediction in zip(indices, targets, predictions)
                }
                if angle == 0:
                    clean_correct = correct
                else:
                    for index in indices:
                        sample_losses[angle][index].append(
                            clean_correct[index] - correct[index]
                        )
            rotation_units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "outer_validation_accessed": False,
                    "record_path": str(record_path.resolve()),
                    "record_sha256": sha256_file(record_path),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "angles": angle_results,
                }
            )
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    rotation_path = output_dir / "rotation_inner_only_units.json"
    rotation_path.write_text(
        json.dumps(rotation_units, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rotation_report = {}
    association_raw = {}
    for angle_index, angle in enumerate(rotations):
        by_fold = defaultdict(list)
        for unit in rotation_units:
            drop = (
                float(unit["angles"][str(angle)]["macro_f1"])
                - float(unit["angles"]["0"]["macro_f1"])
            )
            by_fold[int(unit["fold"])].append(drop)
        fold_means = {fold: statistics.fmean(values) for fold, values in by_fold.items()}
        loss_by_sample = {
            index: statistics.fmean(values)
            for index, values in sample_losses[angle].items()
        }
        ordered_indices = sorted(loss_by_sample)
        association = pearson_permutation(
            [
                feature_by_geometry[selected_geometry][index][
                    "unsigned_anisotropy"
                ]
                for index in ordered_indices
            ],
            [loss_by_sample[index] for index in ordered_indices],
            seed=20260800 + angle_index,
        )
        association_raw[str(angle)] = association["permutation_p_two_sided"]
        rotation_report[str(angle)] = {
            "rotated_minus_clean_macro_f1_mean": statistics.fmean(
                value for values in by_fold.values() for value in values
            ),
            "per_fold_mean": {str(key): value for key, value in fold_means.items()},
            "negative_fold_means": sum(value < 0 for value in fold_means.values()),
            "anisotropy_vs_correctness_loss": association,
        }
    association_adjusted = holm_adjust(association_raw)
    for angle in rotations:
        rotation_report[str(angle)]["anisotropy_vs_correctness_loss"][
            "holm_p"
        ] = association_adjusted[str(angle)]

    thresholds = prereg["direction_gate"]["tests"]
    image_pass = all(
        report["unsigned_anisotropy_fold_bootstrap_ci95"][0]
        > float(thresholds["unsigned_anisotropy_ci95_low_gt"])
        and report["structure_tensor_axial_resultant_length"]
        > float(thresholds["axial_resultant_length_gt"])
        and report["consistent_fold_direction_count"]
        >= int(thresholds["consistent_fold_direction_at_least"])
        for report in image_reports.values()
    ) and len(set(common_signs)) == 1 and common_signs[0] != 0
    rotation90 = rotation_report["90"]
    rotation_pass = (
        -float(rotation90["rotated_minus_clean_macro_f1_mean"])
        > float(thresholds["rotation90_macro_f1_mean_drop_gt"])
        and int(rotation90["negative_fold_means"])
        >= int(thresholds["rotation90_negative_fold_means_at_least"])
    )
    association90 = rotation90["anisotropy_vs_correctness_loss"]
    association_pass = (
        float(association90["r"]) > 0.0
        and float(association90["holm_p"])
        < float(thresholds["anisotropy_loss_association_holm_alpha"])
    )
    passed = image_pass and rotation_pass and association_pass
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "DIRECTIONAL_BASIS_PASS" if passed else "NO_DIRECTIONAL_BASIS",
        "outer_validation_accessed": False,
        "selected_bundle": selected_bundle,
        "selected_geometry": selected_geometry,
        "sobel_definition": prereg["direction_gate"]["sobel_definition"],
        "image_only_analysis": image_reports,
        "rotation_inner_only_analysis": rotation_report,
        "gate_components": {
            "image_direction_consistency": image_pass,
            "rotation90_performance_relation": rotation_pass,
            "anisotropy_loss_association": association_pass,
        },
        "preregistration": {
            "path": str(prereg_path),
            "sha256": sha256_file(prereg_path),
        },
        "raw_evidence": {
            "direction_features": {
                "path": str(feature_path.resolve()),
                "sha256": sha256_file(feature_path),
                "rows": len(feature_rows),
            },
            "rotation_units": {
                "path": str(rotation_path.resolve()),
                "sha256": sha256_file(rotation_path),
                "units": len(rotation_units),
            },
        },
        "claim_boundary": (
            "All diagnostic images and checkpoint evaluations come from each "
            "fold's outer-train/inner split. Inferred hash clusters are not "
            "real acquisition groups. A failed gate terminates Dir development."
        ),
    }
    summary["payload_sha256"] = canonical_sha256(summary)
    target = Path(args.summary).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    random.seed(20260729)
    raise SystemExit(main())
