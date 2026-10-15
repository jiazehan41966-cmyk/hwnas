"""Independent runtime-evidence checker for the bounded operator software pilot.

The manual team source review remains separate from this deterministic check.
No runtime PASS is emitted until the complete new source-bound real smoke exists.
This checker trains no model, installs nothing, and starts no queue or GPU job.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
METHOD = "TEAM_STATIC_REVIEW_PLUS_INDEPENDENT_RUNTIME_CHECKER"
STATIC_STATUS = "STATIC_REVIEW_PASS_RUNTIME_PENDING"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_pairs, parse_constant=invalid_constant)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite numerical receipt")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def digest(value, name):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            "Missing or invalid SHA256: " + name)


def inside(root, relative):
    entry = Path(relative)
    require(not entry.is_absolute() and ".." not in entry.parts,
            "Source path must be relative and bounded: " + str(relative))
    resolved = (root / entry).resolve()
    require(resolved.is_relative_to(root.resolve()), "Source escaped the checkout")
    return resolved


def validate_source_contract(root, manifest, review, review_relative):
    require(review.get("status") == STATIC_STATUS and review.get("method") == METHOD,
            "Exact new team static review is required; runtime status must be pending")
    require(review.get("runtime_gate_status") == "PENDING"
            and review.get("formal_eligible") is False
            and review.get("test_evaluated") is False,
            "Static review must not pretend that runtime or formal gates passed")
    sources = manifest.get("source_sha256", {})
    reviewed = review.get("reviewed_source_sha256", {})
    require(isinstance(sources, dict) and bool(sources)
            and isinstance(reviewed, dict) and bool(reviewed), "Missing source closure")
    require(review_relative in sources, "Static review must be included in the new source freeze")
    expected = {name: value for name, value in sources.items() if name != review_relative}
    require(reviewed == expected, "Manifest source closure differs from independently reviewed snapshot")
    for name, value in sources.items():
        digest(value, name)
        require(sha(inside(root, name)) == value, "Reviewed source bytes drifted: " + name)


def validate_receipt(report, info, initialization, rng, records, expected_binding, targets):
    finite_tree(report)
    finite_tree(info)
    finite_tree(records)
    require(report.get("status") == "PASS" and report.get("binding") == expected_binding,
            "Smoke report binding mismatch")
    require(report.get("formal_eligible") is False and report.get("test_evaluated") is False,
            "Smoke must remain non-formal and train/valid-only")
    train = report.get("training", {})
    require(train.get("batches") == 2 and train.get("images") == 8
            and train.get("gradient_finite") is True, "Actual two-batch/eight-image finite-gradient smoke required")
    losses = train.get("losses", {})
    require(set(losses) == {"loss", "box_loss", "class_loss", "dfl_loss"}
            and losses.get("dfl_loss") == 0.0
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in losses.values()), "Finite TAL/GIoU losses and exactly zero DFL required")
    for key in ("image_order_sha256", "augmented_batches_sha256"):
        digest(train.get(key), key)
    batches = train.get("batch_sha256", [])
    require(isinstance(batches, list) and len(batches) == 2, "Two actual augmented batch hashes required")
    for value in batches:
        digest(value, "batch_sha256")
    require(canonical_sha(batches) == train["augmented_batches_sha256"], "Augmented-batch digest does not reproduce")
    digest(report.get("valid_batches_sha256"), "valid_batches_sha256")
    for key in ("initial_state_sha256", "final_state_sha256"):
        digest(report.get(key), key)
    require(report["initial_state_sha256"] != report["final_state_sha256"], "Smoke state did not change")
    require(report.get("seconds", 0) > 0 and report.get("peak_cuda_allocated_bytes", 0) > 0,
            "Real CUDA smoke timing and allocation evidence required")
    require(report.get("initialization") == initialization
            and info.get("initialization") == initialization,
            "Smoke/model/initialization receipts disagree")
    require(initialization.get("seed") == 42 and initialization.get("variant") == expected_binding["variant"],
            "Initialization seed or variant mismatch")
    require(initialization.get("unchanged_tensor_sha256") == initialization.get("control_unchanged_tensor_sha256"),
            "Paired unchanged tensors differ")
    digest(initialization.get("unchanged_tensor_sha256"), "unchanged tensor identity")
    shared = initialization.get("shared_tensor_sha256", {})
    require(isinstance(shared, dict) and bool(shared), "Actual per-tensor pairing witnesses required")
    for name, value in shared.items():
        digest(value, name)
    require(rng.get("seed") == 42 and rng.get("policy") == "reset_after_construction_before_loaders",
            "Post-construction training RNG receipt required")
    for key in ("torch_cpu_rng_sha256", "numpy_rng_sha256", "python_rng_sha256"):
        digest(rng.get(key), key)
    head = info.get("head_contract", {})
    require(head == {"reg_max": 1, "raw_channels": 6, "measured_channels": [6] * 4},
            "Measured C01b four-scale raw6 head required")
    criterion = info.get("criterion", {})
    require(criterion.get("box") == "GIoU" and criterion.get("distribution") == "disabled"
            and criterion.get("reg_max") == 1 and criterion.get("gains", {}).get("dfl") == 0.0,
            "C01b GIoU/noDFL criterion receipt required")
    require(report.get("valid", {}).get("num_images") == 4
            and isinstance(records, list) and len(records) == 4, "Actual four-image validation records required")
    seen = set()
    for record in records:
        name = record.get("image", "")
        require(isinstance(name, str) and name.startswith("valid/") and ".." not in Path(name).parts
                and name not in seen, "Unique train/valid-only validation image paths required")
        seen.add(name)
        image = Path(name).name
        require(image in targets and record.get("target") == targets[image], "Smoke GT differs from frozen full99")
        pred = record.get("prediction", {})
        boxes, scores, labels = (pred.get(key, []) for key in ("boxes", "scores", "labels"))
        require(len(boxes) == len(scores) == len(labels) and len(scores) <= 100,
                "Frozen prediction lengths or post-NMS limit drift")
        require(all(isinstance(box, list) and len(box) == 4 for box in boxes), "Malformed predicted box")
        require(all(isinstance(score, (int, float)) and 0 <= score <= 1 for score in scores)
                and all(isinstance(label, int) and not isinstance(label, bool) and label in (0, 1)
                        for label in labels), "Nonfinite/out-of-range prediction scores or labels")
    return {"augmented_batches_sha256": train["augmented_batches_sha256"],
            "image_order_sha256": train["image_order_sha256"],
            "valid_batches_sha256": report["valid_batches_sha256"]}


def validate(output, config, static_review=None, verify_actual_environment=True):
    # Heavy/runtime imports occur only for a real invocation, never semantic self-tests.
    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
    from tools.sonar_operator_probe import train as runner
    import torch
    torch.set_num_threads(2)
    cfg, baseline = runner.verify_manifest(output, config)
    source_root = runner.ROOT.resolve()
    output = Path(output).resolve()
    require(output.is_relative_to(source_root / "results"), "Readiness output must remain in isolated results namespace")
    runtime = cfg.get("runtime_readiness", {})
    require(runtime.get("method") == METHOD, "Frozen runtime readiness method differs")
    static_relative = runtime.get("static_review")
    require(isinstance(static_relative, str), "Config must name the new static-review source snapshot")
    static_path = inside(source_root, static_relative)
    if static_review is not None:
        require(Path(static_review).resolve() == static_path, "Static-review override differs from frozen config")
    manifest = read_json(output / "execution_manifest.json")
    review = read_json(static_path)
    validate_source_contract(source_root, manifest, review, static_relative)
    require(cfg["variants"] == review.get("expected_variants") and len(cfg["variants"]) == 21,
            "Exact independently reviewed 21-variant scope required")
    require(review.get("config_sha256") == sha(config)
            and review.get("preregistration_sha256") == manifest["preregistration_sha256"],
            "Reviewed config or preregistration differs")
    lock = read_json(output / "environment_lock.json")
    require(lock.get("fingerprint") == canonical_sha(lock.get("environment")), "Environment lock digest drift")
    env = lock["environment"]
    require(env.get("device") == "cuda" and env.get("workers") == 0
            and env.get("torch_threads") == 2 and env.get("amp") is False,
            "Actual registered CUDA/workers0 execution environment required")
    if verify_actual_environment:
        runner.p.seed_all(42)
        require(runner.t.environment("cuda", cfg) == env, "Checker execution environment differs from real smoke")
    gate = read_json(output / "smoke_gate.json")
    require(gate.get("status") == "PASS" and gate.get("variants") == cfg["variants"]
            and gate.get("execution_manifest_sha256") == sha(output / "execution_manifest.json")
            and gate.get("environment_fingerprint") == lock["fingerprint"]
            and gate.get("formal_eligible") is False and gate.get("test_evaluated") is False,
            "Complete new source/environment-bound smoke gate required")
    require(all(gate.get(key) is True for key in ("actual_two_train_batches_each", "actual_one_valid_batch_each",
            "identical_augmented_batches", "paired_unchanged_initial_tensors")), "Actual smoke assertions required")
    admission = gate.get("admission", {})
    require(admission.get("admitted") is True and admission.get("device") == "cuda"
            and admission.get("physical_gpu") == 1 and admission.get("other_compute_pids") == [],
            "GPU1-exclusive smoke admission evidence required")
    preflight = read_json(output / "preflight_supervisor_status.json")
    commands = preflight.get("commands", [])
    require(preflight.get("status") == "COMPLETE" and preflight.get("completed") == commands
            and len(commands) == 4 and all(isinstance(command, list) for command in commands)
            and "prepare" in commands[0] and "verify" in commands[1]
            and "pytest" in commands[2] and "smoke" in commands[3],
            "Actual prepare/verify/tests/smoke preflight completion receipt required")
    targets = {row["image"]: row["target"] for row in manifest["validation_targets"]}
    require(len(targets) == 99, "Frozen full99 GT closure required")
    rows, witnesses, rngs, evidence = [], [], [], {}
    for variant in cfg["variants"]:
        folder = runner.result_dir(output, variant, 42, "smoke")
        names = ("smoke_report.json", "model_info.json", "initialization_identity.json",
                 "training_rng_identity.json", "valid_records.json")
        report, info, initialization, rng, records = [read_json(folder / name) for name in names]
        require(not (folder / "smoke_failure.json").exists(), "Conflicting smoke failure receipt: " + variant)
        expected = runner.binding(output, variant, 42, "smoke")
        witnesses.append(validate_receipt(report, info, initialization, rng, records, expected, targets))
        require(runner.t.enrich_metrics(runner.p.metrics_from_records(records, baseline)) == report["valid"],
                "Saved validation predictions do not reproduce smoke metrics: " + variant)
        rows.append(report)
        rngs.append(rng)
        for name in names:
            relative = (folder / name).relative_to(output).as_posix()
            evidence[relative] = sha(folder / name)
    require(all(value == witnesses[0] for value in witnesses), "Actual augmented pixels/order/validation witnesses do not pair")
    require(all(value == rngs[0] for value in rngs), "Training RNG receipts do not pair")
    runner.verify_pairing(rows, smoke=True)
    require(all(row["initialization"]["head_sha256"] == rows[0]["initialization"]["head_sha256"] for row in rows),
            "C01b head initialization differs across arms")
    result = dict(schema_version=1, status="PASS", method=METHOD,
        scope="exploratory_operator_screen_runtime_readiness",
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_manifest_sha256=sha(output / "execution_manifest.json"),
        smoke_gate_sha256=sha(output / "smoke_gate.json"),
        environment_lock_sha256=sha(output / "environment_lock.json"),
        static_review_sha256=sha(static_path), static_review=static_relative,
        checker_source_sha256=sha(Path(__file__)), config_sha256=sha(config),
        preregistration_sha256=manifest["preregistration_sha256"],
        preflight_supervisor_status_sha256=sha(output / "preflight_supervisor_status.json"),
        source_files_checked=len(manifest["source_sha256"]), variants=cfg["variants"],
        actual_smoke_reports_checked=len(rows), smoke_seed=42,
        smoke_evidence_sha256=evidence, data_witnesses=witnesses[0],
        all_real_smokes_two_train_batches_eight_images_valid4=True,
        finite_losses_gradients_scores=True, measured_head_raw6_noDFL_GIoU=True,
        weights_changed=True, paired_initialization_and_data=True,
        limitations=["Team static review plus independent deterministic checker; not cross-model review",
                    "Smoke is interface/numeric evidence, not 80-epoch performance",
                    "R0 original-label development protocol; no test or hardware admission"],
        formal_eligible=False, test_evaluated=False,
        automatic_search=False, automatic_hardware=False)
    target = output / "readiness_review.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


class SemanticTests(unittest.TestCase):
    """Synthetic invalid-receipt tests only; never manufacture runtime evidence."""
    def fixture(self):
        code = "a" * 64
        batches = ["b" * 64, "c" * 64]
        binding = {"variant": "control"}
        initialization = {"seed": 42, "variant": "control", "unchanged_tensor_sha256": code,
                          "control_unchanged_tensor_sha256": code, "shared_tensor_sha256": {"head.weight": code}}
        info = {"initialization": initialization,
                "head_contract": {"reg_max": 1, "raw_channels": 6, "measured_channels": [6] * 4},
                "criterion": {"box": "GIoU", "distribution": "disabled", "reg_max": 1, "gains": {"dfl": 0.0}}}
        rng = {"seed": 42, "policy": "reset_after_construction_before_loaders",
               "torch_cpu_rng_sha256": code, "numpy_rng_sha256": code, "python_rng_sha256": code}
        target = {"boxes": [[1., 2., 3., 4.]], "labels": [0]}
        records = [{"image": f"valid/images/synthetic_{index}.png", "target": deepcopy(target),
                    "prediction": {"boxes": [[1., 2., 3., 4.]], "scores": [.5], "labels": [0]}}
                   for index in range(4)]
        targets = {f"synthetic_{index}.png": deepcopy(target) for index in range(4)}
        report = {"status": "PASS", "binding": binding, "formal_eligible": False, "test_evaluated": False,
                  "training": {"batches": 2, "images": 8, "gradient_finite": True,
                               "losses": {"loss": 1., "box_loss": .5, "class_loss": .5, "dfl_loss": 0.0},
                               "batch_sha256": batches, "augmented_batches_sha256": canonical_sha(batches),
                               "image_order_sha256": code},
                  "valid": {"num_images": 4}, "valid_batches_sha256": code,
                  "initial_state_sha256": code, "final_state_sha256": "d" * 64,
                  "initialization": initialization, "seconds": 1., "peak_cuda_allocated_bytes": 1}
        return deepcopy((report, info, initialization, rng, records, binding, targets))

    def test_complete_semantic_fixture_returns_only_witnesses(self):
        witnesses = validate_receipt(*self.fixture())
        self.assertEqual(set(witnesses), {"augmented_batches_sha256", "image_order_sha256", "valid_batches_sha256"})

    def test_source_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.py").write_text("original", encoding="utf-8")
            (root / "static.json").write_text("static", encoding="utf-8")
            sources = {name: sha(root / name) for name in ("candidate.py", "static.json")}
            review = {"status": STATIC_STATUS, "method": METHOD, "runtime_gate_status": "PENDING",
                      "formal_eligible": False, "test_evaluated": False,
                      "reviewed_source_sha256": {"candidate.py": sources["candidate.py"]}}
            validate_source_contract(root, {"source_sha256": sources}, review, "static.json")
            (root / "candidate.py").write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes drifted"):
                validate_source_contract(root, {"source_sha256": sources}, review, "static.json")

    def test_measured_head_and_changed_weights_required(self):
        args = self.fixture()
        args[1]["head_contract"]["measured_channels"] = [21] * 4
        with self.assertRaisesRegex(ValueError, "raw6"):
            validate_receipt(*args)
        args = self.fixture()
        args[0]["final_state_sha256"] = args[0]["initial_state_sha256"]
        with self.assertRaisesRegex(ValueError, "did not change"):
            validate_receipt(*args)

    def test_actual_gt_and_scores_required(self):
        args = self.fixture()
        args[4][0]["prediction"]["scores"] = [1.1]
        with self.assertRaisesRegex(ValueError, "prediction scores"):
            validate_receipt(*args)
        args = self.fixture()
        args[4][0]["target"]["labels"] = [1]
        with self.assertRaisesRegex(ValueError, "GT differs"):
            validate_receipt(*args)

    def test_augmented_hash_and_test_split_guards(self):
        args = self.fixture()
        args[0]["training"]["batch_sha256"][0] = "e" * 64
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            validate_receipt(*args)
        args = self.fixture()
        args[4][0]["image"] = "test/images/synthetic_0.png"
        with self.assertRaisesRegex(ValueError, "validation image"):
            validate_receipt(*args)

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            finite_tree({"score": float("nan")})

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _pairs([("status", "PASS"), ("status", "FAIL")])

    def test_source_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            inside(ROOT, "../outside.py")

    def test_fake_runtime_static_pass_rejected(self):
        with self.assertRaisesRegex(ValueError, "runtime status"):
            validate_source_contract(ROOT, {}, {"status": "PASS", "method": METHOD}, "static.json")

    def test_missing_actual_batches_rejected(self):
        report = {"status": "PASS", "binding": {}, "formal_eligible": False, "test_evaluated": False,
                  "training": {"batches": 1, "images": 4, "gradient_finite": True}}
        with self.assertRaisesRegex(ValueError, "two-batch"):
            validate_receipt(report, {}, {}, {}, [], {}, {})

    def test_dfl_nonzero_rejected(self):
        report = {"status": "PASS", "binding": {}, "formal_eligible": False, "test_evaluated": False,
                  "training": {"batches": 2, "images": 8, "gradient_finite": True,
                  "losses": {"loss": 1., "box_loss": 1., "class_loss": 0., "dfl_loss": .1}}}
        with self.assertRaisesRegex(ValueError, "zero DFL"):
            validate_receipt(report, {}, {}, {}, [], {}, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--static-review", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SemanticTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.config is None or args.output is None:
        parser.error("--config and --output are required for a real runtime check")
    try:
        result = validate(args.output, args.config.resolve(), args.static_review)
    except Exception as error:
        print(json.dumps({"status": "FAILED_RUNTIME_READINESS", "error_type": type(error).__name__,
                          "error": str(error), "method": METHOD, "formal_eligible": False}), flush=True)
        raise SystemExit(1)
    print(json.dumps({key: result[key] for key in ("status", "method", "source_files_checked",
                      "actual_smoke_reports_checked", "execution_manifest_sha256", "smoke_gate_sha256",
                      "formal_eligible", "test_evaluated")}), flush=True)


if __name__ == "__main__":
    main()
