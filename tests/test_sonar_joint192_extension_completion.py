"""Exercise completion metadata on disk without data loading or training.

The regression fixture includes ``test_evaluated`` in the actual run binding:
the original completion failure only appeared after a full training run when
that field was supplied twice to ``dict``. Recovery checkpoint fixtures contain
synthetic metadata, not trained weights or performance evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sonar_joint192_extension_20261013 import queue_runner, recovery, train


@pytest.fixture
def completion(tmp_path, monkeypatch):
    architecture = json.loads(train.CONFIG.read_text(encoding="utf-8"))["arms"]["A00_reference"]["architecture"]
    config = {"reference": "A00_reference", "arms": {"A00_reference": {"architecture": architecture}}}
    result_root = tmp_path / "results"
    config_path = tmp_path / "config.json"
    train.dump(config_path, config)
    manifest = {"config": config, "test_evaluated": False, "test_mounted": False}
    train.dump(result_root / "execution_manifest.json", manifest)
    monkeypatch.setattr(train, "CONFIG", config_path)
    monkeypatch.setattr(train, "RESULT", result_root)
    # Source/data verification is separate from this metadata transaction test.
    monkeypatch.setattr(train, "manifest", lambda: deepcopy(manifest))
    binding = dict(arm="A00_reference", seed=42, config_sha256=train.sha(config_path),
                   execution_manifest_sha256=train.sha(result_root / "execution_manifest.json"),
                   architecture_sha256=train.architecture_hash(architecture),
                   training_epochs=80, fresh_initialization=True, test_evaluated=False)
    folder = result_root / "runs" / binding["arm"] / "seed_42"
    folder.mkdir(parents=True)
    for name in ("best.pt", "latest.pt"):
        (folder / name).write_bytes(("metadata artifact: " + name).encode())
    train.dump(folder / "epoch_log.json", [{"epoch": n} for n in range(1, 81)])
    train.dump(folder / "run_binding.json", binding)
    train.dump(folder / "status.json", dict(status="RUNNING", binding=binding, epoch=80))
    metrics = {key: (i + 1) / 100 for i, key in enumerate(train.METRICS)}
    payload = dict(best_epoch=7, best_metrics=metrics, endpoint_metrics=metrics,
                   best_domain={"FP_per_image": 1.0}, endpoint_domain={"FP_per_image": 1.1},
                   benchmark={"claimable_efficiency_comparison": False}, info={"fixture": True},
                   best_checkpoint_sha256=train.sha(folder / "best.pt"),
                   latest_checkpoint_sha256=train.sha(folder / "latest.pt"),
                   history_sha256=train.sha(folder / "epoch_log.json"), wall_seconds=80.0)
    return folder, binding, payload


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _recover_result(completion):
    folder, finalization, payload = completion
    original = deepcopy(finalization)
    original["execution_manifest_sha256"] = "a" * 64
    origin = dict(status="VERIFIED", origin_execution_manifest_sha256="a" * 64,
                  optimizer_steps_during_recovery=0, training_epochs_repeated=0)
    train.dump(folder / "run_binding.json", original)
    result = train.pack_completed_result(original, **payload,
                                         finalization_binding=finalization, recovery_origin=origin)
    train.dump(folder / "recovery_receipt.json",
               dict(status="PASS", origin=origin, finalization_binding=finalization))
    return result, original


def test_complete_binding_pack_commit_and_collect_round_trip(completion, monkeypatch):
    folder, binding, payload = completion
    original_binding = deepcopy(binding)
    train.dump(folder / "failure.json", {"status": "OLD_FAILURE_PRESERVED"})
    failure_bytes = (folder / "failure.json").read_bytes()
    result = train.pack_completed_result(binding, **payload)
    assert binding == original_binding
    assert result["test_evaluated"] is False
    assert result["full_epochs"] == 80 and result["global_step"] == 6640
    assert "test_evaluated" in result
    assert len([key for key in result if key == "test_evaluated"]) == 1
    observed_between_writes = []
    real_dump = train.dump

    def observe_result_before_complete(path, value):
        real_dump(path, value)
        if Path(path).name == "result.json":
            observed_between_writes.append(train.collect_results(verify_artifacts=True)[1])

    monkeypatch.setattr(train, "dump", observe_result_before_complete)
    train.commit_completed_result(folder, result)
    assert observed_between_writes == [{}]
    assert _read(folder / "result.json") == result
    assert _read(folder / "status.json")["binding"] == original_binding
    assert train.collect_results(verify_artifacts=True)[1] == {(binding["arm"], 42): result}
    assert (folder / "failure.json").read_bytes() == failure_bytes


@pytest.mark.parametrize("change", [{"training_epochs": 79}, {"test_evaluated": True},
                                    {"unexpected": 1}])
def test_pack_rejects_invalid_completion_binding(completion, change):
    _, binding, payload = completion
    with pytest.raises(ValueError, match="Invalid completion binding"):
        train.pack_completed_result(dict(binding, **change), **payload)


@pytest.mark.parametrize("epoch", [0, 81, 7.0])
def test_pack_rejects_invalid_best_epoch(completion, epoch):
    _, binding, payload = completion
    payload["best_epoch"] = epoch
    with pytest.raises(ValueError, match="Invalid completed best epoch"):
        train.pack_completed_result(binding, **payload)


@pytest.mark.parametrize("phase", ["best_metrics", "endpoint_metrics"])
def test_pack_rejects_nonfinite_metrics(completion, phase):
    _, binding, payload = completion
    payload = deepcopy(payload)
    payload[phase]["mAP50_95"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite completed metrics"):
        train.pack_completed_result(binding, **payload)


@pytest.mark.parametrize("field,value", [("config_sha256", "wrong"),
                                         ("execution_manifest_sha256", "wrong"),
                                         ("architecture_sha256", "wrong"),
                                         ("seed", 43), ("test_evaluated", True),
                                         ("full_epochs", 79), ("global_step", 6557)])
def test_collector_rejects_wrong_identity_or_incomplete_result(completion, field, value):
    folder, binding, payload = completion
    result = train.pack_completed_result(binding, **payload)
    train.commit_completed_result(folder, result)
    result[field] = value
    train.dump(folder / "result.json", result)
    with pytest.raises(ValueError, match="identity or completion"):
        train.collect_results()


@pytest.mark.parametrize("artifact", ["best.pt", "latest.pt", "epoch_log.json"])
def test_collector_verifies_saved_artifact_hashes(completion, artifact):
    folder, binding, payload = completion
    train.commit_completed_result(folder, train.pack_completed_result(binding, **payload))
    (folder / artifact).write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="artifact hash drift"):
        train.collect_results(verify_artifacts=True)


def test_duplicate_completion_cannot_overwrite_result_or_status(completion):
    folder, binding, payload = completion
    result = train.pack_completed_result(binding, **payload)
    train.commit_completed_result(folder, result)
    before = {name: (folder / name).read_bytes() for name in ("result.json", "status.json")}
    result["best_epoch"] = 8
    with pytest.raises(FileExistsError, match="already exists"):
        train.commit_completed_result(folder, result)
    assert {name: (folder / name).read_bytes() for name in before} == before


@pytest.mark.parametrize("present", ["finalization_binding", "recovery_origin"])
def test_recovery_requires_origin_and_finalization_together(completion, present):
    _, binding, payload = completion
    one = binding if present == "finalization_binding" else {"status": "VERIFIED"}
    with pytest.raises(ValueError, match="origin and finalization bindings together"):
        train.pack_completed_result(binding, **payload, **{present: one})


@pytest.mark.parametrize("seed", [41, 44, "42", 42.0, True, None])
def test_recovery_refuses_unapproved_seeds_before_reading_or_writing(monkeypatch, seed):
    def forbidden():
        pytest.fail("Invalid recovery seed must fail before accessing the manifest")

    monkeypatch.setattr(train, "manifest", forbidden)
    with pytest.raises(ValueError, match="Only the two completed reference seeds 42/43"):
        recovery.recover_reference(seed)


def test_recovered_completion_keeps_original_training_identity(completion):
    folder, finalization, _ = completion
    before = deepcopy(finalization)
    result, original = _recover_result(completion)
    train.commit_completed_result(folder, result)
    assert finalization == before
    assert result["origin_run_binding"] == original
    assert result["execution_manifest_sha256"] == original["execution_manifest_sha256"]
    assert result["finalization_binding"] == finalization
    assert result["execution_manifest_sha256"] != finalization["execution_manifest_sha256"]
    assert _read(folder / "run_binding.json") == original
    assert train.collect_results()[1] == {(original["arm"], 42): result}
    # Recovery forces artifact checks even when the caller uses the default.
    (folder / "best.pt").write_bytes(b"corrupted recovered checkpoint")
    with pytest.raises(ValueError, match="artifact hash drift"):
        train.collect_results()


@pytest.mark.parametrize("target", ["origin_run_binding", "finalization_binding", "receipt"])
def test_recovery_collector_rejects_changed_provenance(completion, target):
    folder, _, _ = completion
    result, _ = _recover_result(completion)
    train.commit_completed_result(folder, result)
    if target == "receipt":
        receipt = _read(folder / "recovery_receipt.json")
        receipt["origin"]["origin_execution_manifest_sha256"] = "b" * 64
        train.dump(folder / "recovery_receipt.json", receipt)
    else:
        result[target]["execution_manifest_sha256"] = "b" * 64
        train.dump(folder / "result.json", result)
    with pytest.raises(ValueError, match="binding|receipt"):
        train.collect_results()


@pytest.mark.parametrize("recovered_count", [0, 1])
def test_queue_requires_both_recovered_references_before_any_launch(completion, monkeypatch, recovered_count):
    folder, _, _ = completion
    if recovered_count:
        result, _ = _recover_result(completion)
        train.commit_completed_result(folder, result)
    manifest = train.manifest()
    manifest["config"].update(gpu=1, concurrency=2)
    monkeypatch.setattr(train, "manifest", lambda: deepcopy(manifest))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    train.dump(train.RESULT / "smoke_gate.json",
               dict(status="PASS", execution_manifest_sha256=train.sha(train.RESULT / "execution_manifest.json")))
    train.dump(train.RESULT / "launch.json", dict(actual_concurrency=2))
    # The queue runs only on Linux; replace its OS lock on the CPU test host.
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(
        LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None))

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete inherited reference set must never start a training child")

    monkeypatch.setattr(queue_runner.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="Both verified recovered reference runs are required"):
        queue_runner.main(1)
    assert not (train.RESULT / "queue_status.json").exists()


@pytest.fixture
def completed_checkpoint_metadata(completion):
    folder, binding, _ = completion
    history = [dict(epoch=n, batches=83, gradient_finite=True,
                    valid={"mAP50_95": 0.1 if n in (3, 5) else 0.01}) for n in range(1, 81)]
    initialization = {"fixture": "synthetic metadata without trained model weights"}
    train.dump(folder / "initialization.json", initialization)
    train.dump(folder / "epoch_log.json", history)
    latest = dict(binding=binding, epoch=80, global_step=6640, history=history,
                  initialization=initialization)
    best = dict(binding=binding, epoch=3, global_step=249, history=history[:3],
                initialization=initialization)
    torch.save(latest, folder / "latest.pt")
    torch.save(best, folder / "best.pt")
    return folder, binding, latest, best, history


def test_recovery_validates_full_coverage_and_earliest_tie_without_training(completed_checkpoint_metadata, monkeypatch):
    folder, binding, latest, best, history = completed_checkpoint_metadata

    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint metadata validation must not train or take optimizer steps")

    monkeypatch.setattr(train, "train_epoch", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    actual = recovery.validate_completed_checkpoints(
        folder, binding["execution_manifest_sha256"], binding["arm"], binding["seed"])
    assert actual == (binding, latest, best, history, history[2])


@pytest.mark.parametrize("fault", ["epoch79", "wrong_manifest", "missing_epoch",
                                   "nonfinite_gradient", "later_tied_best", "initialization"])
def test_recovery_rejects_incomplete_or_inconsistent_checkpoint_metadata(completed_checkpoint_metadata, fault):
    folder, binding, latest, best, history = completed_checkpoint_metadata
    manifest_sha = binding["execution_manifest_sha256"]
    if fault == "epoch79":
        latest["epoch"] = 79
    elif fault == "wrong_manifest":
        manifest_sha = "wrong"
    elif fault == "missing_epoch":
        history.pop(20)
        train.dump(folder / "epoch_log.json", history)
    elif fault == "nonfinite_gradient":
        history[60]["gradient_finite"] = False
        train.dump(folder / "epoch_log.json", history)
    elif fault == "later_tied_best":
        best["epoch"] = 5
        best["history"] = history[:5]
    elif fault == "initialization":
        best["initialization"] = {"fixture": "wrong initialization"}
    torch.save(latest, folder / "latest.pt")
    torch.save(best, folder / "best.pt")
    with pytest.raises(ValueError, match="Original"):
        recovery.validate_completed_checkpoints(folder, manifest_sha, binding["arm"], binding["seed"])
