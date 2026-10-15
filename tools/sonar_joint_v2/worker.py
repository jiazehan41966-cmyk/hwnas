"""Real budget-specific training and train-calibrated validation PTQ workers.

Training and source/data binding reuse the frozen V1 implementation.  Each PTQ
result is isolated by architecture, seed, and budget, so the 40/60/80 promotion
stages cannot accidentally reuse a later checkpoint or its quantized metrics.
This worker never measures hardware and never evaluates the sealed test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from sonar_search import worker as legacy
from phase5a_quantization.calibrate import (
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_IMAGES,
    CALIBRATION_SEED,
    build_calibration,
    calibration_tensor_batches,
)

# Keep the real training implementation, including resumable optimizer/RNG and
# earliest-best checkpoint selection, identical to the established worker.
assert_binding = legacy.assert_binding
train_candidate = legacy.train_candidate
architectures = legacy.architectures
PROFILES = legacy.PROFILES
torch = legacy.torch
frozen = legacy.frozen
build_sonar_candidate = legacy.build_sonar_candidate
SPEC_PATH = legacy.SPEC_PATH
prepare_h2_ptq = legacy.prepare_h2_ptq
DenseAnchorDetectionLoss = legacy.DenseAnchorDetectionLoss
sha = legacy.sha


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _digest(payload, *, sort_keys=False):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _check_task(aid, seed, budget, freeze):
    protocol = freeze["protocol"]
    if protocol["budgets"] != [40, 60, 80]:
        raise ValueError("Frozen worker requires the 40/60/80 training budgets")
    if (
        aid not in architectures()
        or seed not in protocol["confirmation_seeds"]
        or budget not in protocol["budgets"]
    ):
        raise ValueError("Task outside frozen candidate/seed/budget")
    # The existing training contract admits extra confirmation seeds at the
    # final budget only.  PTQ follows the same seed/budget domain.
    if seed != 42 and budget != 80:
        raise ValueError("Confirmation seeds only at full training budget")
    profiles = protocol["quant_profiles"]
    if len(profiles) != len(PROFILES) or set(profiles) != set(PROFILES):
        raise ValueError("All three measured precision profiles, including W8A8, are required")
    calibration = protocol["calibration"]
    if (
        protocol.get("test_policy") != "sealed"
        or protocol.get("evaluation_split") != "valid"
        or calibration.get("split") != "train"
        or calibration.get("images") != CALIBRATION_IMAGES
        or calibration.get("augmentation") is not False
    ):
        raise ValueError("Worker requires sealed test, valid evaluation, and fixed train-only calibration")
    if freeze.get("test_evaluated") is not False:
        raise ValueError("Frozen campaign must keep test_evaluated=false")
    if set(freeze["dataset"]) != {"train", "valid"}:
        raise ValueError("Only frozen train/valid manifests are admitted")


def validate_calibration(manifest, freeze, baseline_config):
    """Bind the fixed 64 train images to frozen identities, hashes, and order."""
    expected_fields = {
        "split": "train",
        "calibration_seed": CALIBRATION_SEED,
        "number_of_calibration_images": CALIBRATION_IMAGES,
        "batch_size": CALIBRATION_BATCH_SIZE,
        "order": "frozen_loader_sorted_first_64",
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError("Calibration contract mismatch: " + field)
    for field in ("valid_used_for_selection", "test_evaluated", "augmentation", "shuffle"):
        if manifest.get(field) is not False:
            raise ValueError("Calibration split/selection violation: " + field)

    frozen_train = freeze["dataset"]["train"]
    if len(frozen_train) < CALIBRATION_IMAGES:
        raise ValueError("Frozen train split has fewer than 64 calibration images")
    # Freeze order is significant: sorting here would hide an identity or
    # loader-order mismatch and permit a different calibration subset.
    selected = frozen_train[:CALIBRATION_IMAGES]
    if manifest.get("entries") != selected:
        raise ValueError("Calibration identity/hash/order differs from frozen first 64 train entries")
    train_listing = [entry["image"] for entry in frozen_train]
    if manifest.get("train_image_count") != len(frozen_train):
        raise ValueError("Calibration train population drift")
    if manifest.get("split_hash") != _digest(train_listing):
        raise ValueError("Calibration full-train order hash mismatch")
    if manifest.get("dataset_hash") != _digest(selected, sort_keys=True):
        raise ValueError("Calibration selected-image/annotation hash mismatch")

    data = ROOT / baseline_config["dataset"]["data_dir"]
    loader = frozen.make_loader(
        baseline_config, data, "train", CALIBRATION_SEED, 0, train=False
    )
    dataset = loader.dataset
    actual_listing = [path.resolve().relative_to(dataset.root.resolve()).as_posix() for path in dataset.samples]
    if actual_listing != train_listing:
        raise ValueError("Live train-loader identity/order differs from frozen train manifest")
    for entry in selected:
        for identity_key, hash_key in (("image", "image_sha256"), ("annotation", "annotation_sha256")):
            relative = Path(entry[identity_key])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train":
                raise ValueError("Calibration entry is outside the train split")
            if sha(data / relative) != entry[hash_key]:
                raise ValueError("Calibration file content changed: " + entry[identity_key])


def _check_profile(profile, name):
    """Verify that the generated model profile represents its claimed bits."""
    head_weight, head_activation = PROFILES[name]
    layers = profile.get("layers", [])
    if not layers or not any(row["name"].startswith("head.") for row in layers):
        raise ValueError("PTQ profile has no bound detection-head layers")
    for row in layers:
        weight, activation = (head_weight, head_activation) if row["name"].startswith("head.") else (8, 8)
        if (row["weight_bits"], row["input_bits"], row["output_bits"]) != (weight, activation, activation):
            raise ValueError("Generated PTQ layer precision differs from requested profile")
        if row.get("fp32_weight") is not False or row.get("fp32_activation") is not False:
            raise ValueError("PTQ profile unexpectedly bypasses quantization")
    bypass = profile["fp32_bypass"]
    if bypass.get("software_diagnostic_only") is not False or bypass.get("neck_adds_quantized") is not True:
        raise ValueError("PTQ profile unexpectedly bypasses FPN integer quantization")


def quant_candidate(out, aid, seed, budget, device, freeze, freeze_sha, baseline_config):
    _check_task(aid, seed, budget, freeze)
    checkpoint = out / "training" / aid / f"seed_{seed}" / f"best_{budget}.pt"
    checkpoint_sha = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_binding = {
        "freeze_sha256": freeze_sha,
        "architecture_id": aid,
        "seed": seed,
        "budget": budget,
    }
    for field, expected in expected_binding.items():
        if state.get(field) != expected:
            raise ValueError("Quantized checkpoint binding mismatch: " + field)
    if state.get("architecture") != architectures()[aid]:
        raise ValueError("Checkpoint architecture configuration mismatch")
    if state.get("test_evaluated") is not False:
        raise ValueError("Checkpoint test split violation")
    if not isinstance(state.get("epoch"), int) or not 1 <= state["epoch"] <= budget:
        raise ValueError("Checkpoint epoch outside requested training budget")

    folder = out / "quantization" / aid / f"seed_{seed}" / f"budget_{budget}"
    folder.mkdir(parents=True, exist_ok=True)
    calibration_dir = out / "calibration"
    calibration_path = calibration_dir / "calibration_manifest.json"
    manifest = read(calibration_path) if calibration_path.exists() else build_calibration(calibration_dir)
    validate_calibration(manifest, freeze, baseline_config)
    calibration_sha = sha(calibration_path)
    batches = list(calibration_tensor_batches(manifest, device))
    if sum(int(batch.shape[0]) for batch in batches) != CALIBRATION_IMAGES:
        raise ValueError("Calibration tensor population is not exactly 64 train images")

    model = build_sonar_candidate(SPEC_PATH, architectures()[aid]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    data = ROOT / baseline_config["dataset"]["data_dir"]
    valid = frozen.make_loader(baseline_config, data, "valid", seed, 0)
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline_config)).to(device)

    def evaluate(candidate):
        metrics, _ = frozen.evaluate(candidate, valid, criterion, baseline_config, device)
        result = {key: float(metrics[key]) for key in ("mAP50_95", "AP50", "AP75", "Precision", "Recall")}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result.values()):
            raise ValueError("Validation evaluation produced invalid accuracy metrics")
        return result

    binding = dict(
        architecture_id=aid, seed=seed, budget=budget,
        checkpoint=checkpoint.relative_to(out).as_posix(),
        checkpoint_sha256=checkpoint_sha, freeze_sha256=freeze_sha,
        calibration_sha256=calibration_sha, split="valid", test_evaluated=False,
    )
    result = dict(binding, fp32=evaluate(model), profiles={})
    artifacts = [checkpoint, calibration_path]
    for name in freeze["protocol"]["quant_profiles"]:
        weight_bits, activation_bits = PROFILES[name]
        folded, quantized, extrema, profile = prepare_h2_ptq(
            model, batches, device=device,
            weight_granularity="per_tensor", activation_granularity="per_tensor",
            scheme="symmetric", weight_bits_by_prefix={"head": weight_bits},
            activation_bits_by_prefix={"head": activation_bits}, quantize_neck_adds=True,
        )
        _check_profile(profile, name)
        metrics = evaluate(quantized)
        profile.update(binding=binding, precision_profile=name)
        profile_path = folder / (name + "_profile.json")
        save(profile_path, profile)
        artifacts.append(profile_path)
        result["profiles"][name] = dict(
            metrics, profile=profile_path.relative_to(out).as_posix(),
            profile_sha256=sha(profile_path),
        )
        del folded, quantized, extrema
    if sha(checkpoint) != checkpoint_sha or sha(calibration_path) != calibration_sha:
        raise ValueError("Checkpoint/calibration changed while quantization was running")
    result_path = folder / "result.json"
    save(result_path, result)
    artifacts.append(result_path)
    return result, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "quant"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output.resolve()
    # Reject any accidental test manifest before the legacy data-hash verifier
    # can read it.  Normal V2 freezes contain only train and valid.
    preview = read(out / "freeze.json")
    _check_task(args.architecture, args.seed, args.budget, preview)
    freeze, freeze_sha, baseline_config = assert_binding(out)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    function = train_candidate if args.mode == "train" else quant_candidate
    result, outputs = function(
        out, args.architecture, args.seed, args.budget, args.device,
        freeze, freeze_sha, baseline_config,
    )
    receipt = out / "tasks" / f"{args.mode}_{args.architecture}_{args.seed}_{args.budget}.json"
    save(receipt, dict(
        mode=args.mode, architecture_id=args.architecture, seed=args.seed, budget=args.budget,
        freeze_sha256=freeze_sha, result=result,
        outputs={path.relative_to(out).as_posix(): sha(path) for path in outputs},
    ))


if __name__ == "__main__":
    main()
