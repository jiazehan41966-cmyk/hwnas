"""Bounded rule-based continuation of six historical 60-epoch candidates.

No surrogate, no new architecture, no test, no changes to the frozen trainer.
Run this file in an isolated copy of the admitted Phase2E source tree.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
D_ID = "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3"
VERSION = "phase6b-rule-loop-v1"
METRICS = {"mAP50_95": "map50_95", "AP50": "map50", "AP75": "map75",
           "Precision": "precision", "Recall": "recall"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@contextmanager
def supervisor_lock(path):
    """An OS-held nonblocking lock, released on process exit without stale-PID logic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError(f"PHASE6B_SUPERVISOR_ALREADY_RUNNING: {path}") from exc
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def reference_provenance():
    return dict(source_checkpoint_role="epoch80 archive containing the complete training/valid history",
                primary_metric_role="best valid mAP50_95 within epochs1-80 and Recall at that same best epoch",
                endpoint_weight_metric=False,
                limitation="The archive SHA does not bind the quoted best metric to epoch80 endpoint weights")


def select_candidates(preflight, stage2_rows, final_rows):
    """Four highest observed 60e scores, two uniform exploratory IDs; seed42."""
    if preflight.get("source_protocol_binding") != "PASS" or preflight.get("test_evaluated") is not False:
        raise ValueError("Preflight source binding/test seal has not passed")
    by_id = {r["architecture_id"]: r for r in stage2_rows}
    if len(by_id) != len(stage2_rows):
        raise ValueError("Duplicate architecture in Stage2 records")
    final_ids = {r["architecture_id"] for r in final_rows}
    eligible = []
    seen = set()
    for entry in preflight["remaining"]:
        aid = entry["architecture_id"]
        if aid in seen:
            raise ValueError("Duplicate architecture in preflight")
        seen.add(aid)
        if not entry.get("eligible_resume60to80"):
            continue
        if aid in final_ids or entry.get("is_existing_80_witness") or entry.get("budget80_exists"):
            raise ValueError("Completed 80-epoch architecture cannot be rerun")
        if entry.get("latest_epoch") != 60 or entry.get("history_epochs") != list(range(1, 61)):
            raise ValueError("Parent must be a complete 60-epoch checkpoint")
        if not all(entry.get(k) for k in ("budget60_hash_matches", "architecture_matches", "source_binding_matches", "optimizer_present")):
            raise ValueError("Parent checkpoint admission failed")
        source = by_id[aid]
        if int(source["budget"]) != 60 or int(source["seed"]) != 42:
            raise ValueError("Only historical seed42 budget60 is admissible")
        score = float(source["mAP50_95"])
        if not math.isfinite(score) or score != float(entry["mAP60_best"]):
            raise ValueError("Preflight/Stage2 score mismatch")
        if source["checkpoint_sha256"] != entry["latest_sha256"]:
            raise ValueError("Preflight/Stage2 parent hash mismatch")
        eligible.append(dict(architecture_id=aid, mAP60_best=score,
                             parent_checkpoint_sha256=entry["latest_sha256"],
                             Params=int(source["Params"]), MACs=int(source["MACs"])))
    if len(eligible) < 6:
        raise ValueError("At least six eligible uncompleted candidates are required")
    ranked = sorted(eligible, key=lambda r: (-r["mAP60_best"], r["architecture_id"]))
    chosen = [dict(r, selection_reason="top4_60e_best") for r in ranked[:4]]
    pool = sorted(ranked[4:], key=lambda r: r["architecture_id"])
    chosen.extend(dict(r, selection_reason="uniform_exploration_seed42") for r in random.Random(42).sample(pool, 2))
    return chosen, len(eligible)


def prepare(preflight_path, stage2_path, final_path, output):
    output = Path(output)
    preflight = read(preflight_path)
    final_rows = table(final_path)
    selected, pool_size = select_candidates(preflight, table(stage2_path), final_rows)
    refs = [r for r in final_rows if r["architecture_id"] == D_ID and int(r["seed"]) == 42 and int(r["budget"]) == 80]
    if len(refs) != 1:
        raise ValueError("Unique D seed42 80e historical reference is required")
    ref = refs[0]
    protocol = dict(version=VERSION, seed=42, maximum_new_candidates=6, parent_budget=60,
        final_budget=80, selection="4 highest 60e best valid mAP + 2 uniform seed42 exploration",
        eligible_population="historical Stage2 minus FINAL72, witnesses, audited completed80",
        stage2_population_conditioning=True, no_global_optimum_claim=True,
        optimization="frozen Phase2E own checkpoint optimizer+all RNG resume",
        stop_metric="best-valid checkpoint mAP50_95 AND its Recall >= D seed42 80e",
        stop_semantics="post-batch screening only; no unsafe early-epoch futility test",
        deployment_gate="separate three-seed same-precision D gate remains required",
        automatic_next_batch=False, automatic_multiseed=False, automatic_quantization=False,
        automatic_hardware=False, test_evaluated=False,
        metrics_policy="epoch sidecar mirrored to append-only ledger; missing not fabricated")
    manifest = dict(version=VERSION, protocol=protocol, source_binding=preflight["binding"],
        sources={str(Path(p)): sha(p) for p in (preflight_path, stage2_path, final_path)},
        eligible_count=pool_size, selected=selected,
        reference=dict(architecture_id=D_ID, seed=42, budget=80, best_epoch=int(ref["best_epoch"]),
                       mAP50_95=float(ref["mAP50_95"]), Recall=float(ref["Recall"]),
                       source_checkpoint_sha256=ref["checkpoint_sha256"],
                       source_history_sha256=ref["source_history_sha256"]), test_evaluated=False)
    destination = output / "selection_manifest.json"
    if destination.exists() and read(destination) != manifest:
        raise ValueError("Frozen selection manifest already exists with different content")
    write(destination, manifest)
    write(output / "RULE_PROTOCOL.json", protocol)
    write(output / "selection_manifest.sha256.json", {"sha256": sha(destination)})
    return manifest


def screening_decision(rows, reference):
    if len(rows) != 6 or len({r["architecture_id"] for r in rows}) != 6:
        return dict(status="INCOMPLETE", automatic_next_batch=False)
    passed = []
    for row in rows:
        if row.get("epoch") != 80 or row.get("seed") != 42 or row.get("test_evaluated") is not False:
            raise ValueError("Screening requires matched completed seed42 budget80 valid-only results")
        if any(not math.isfinite(float(row[k])) for k in ("mAP50_95", "Recall")):
            raise ValueError("Non-finite screening metric")
        if row["mAP50_95"] >= reference["mAP50_95"] and row["Recall"] >= reference["Recall"]:
            passed.append(row["architecture_id"])
    return dict(status="SCREENING_PASS_REVIEW_REQUIRED" if passed else "STOP_NO_SCREENING_CANDIDATE",
                passing_architectures=passed, automatic_next_batch=False,
                automatic_quantization=False, automatic_hardware=False, test_evaluated=False,
                note="seed42 screening is not the three-seed deployment admission gate")


def check_state(state, selected, binding, epoch):
    if (state.get("architecture_id") != selected["architecture_id"] or state.get("seed") != 42
            or state.get("epoch") != epoch or state.get("binding") != binding
            or state.get("smoke") is not False or state.get("test_evaluated") is not False):
        raise ValueError("Checkpoint identity/budget/binding/test-policy mismatch")
    if [r["epoch"] for r in state["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Incomplete checkpoint history")
    if "optimizer_state_dict" not in state or "rng" not in state:
        raise ValueError("Checkpoint optimizer/RNG is missing")
    required_rng = {"python", "numpy", "torch", "cuda", "loader_generator", "sampler_generator"}
    if not isinstance(state["rng"], dict) or not required_rng.issubset(state["rng"]):
        raise ValueError("Checkpoint RNG must include Python, NumPy, Torch, CUDA, loader and sampler states")
    if not isinstance(state["rng"]["cuda"], (list, tuple)) or len(state["rng"]["cuda"]) != 1:
        raise ValueError("Single-GPU continuation requires exactly one saved CUDA RNG state")


def mirror_history(folder, selected, manifest_hash, ledger, expected_binding, strict=False):
    """Persist committed epoch values; exact raw loss/valid dictionaries also retained."""
    sys.path.insert(0, str(ROOT / "src"))
    from hwnas_fpga.metrics_ledger import append_metric_record, build_metric_record, require_measured_metrics
    path = folder / "epoch_log.json"
    if not path.exists():
        if strict:
            raise ValueError("Completed run has no epoch history")
        return
    try:
        history = read(path)
    except json.JSONDecodeError:
        if strict:
            raise ValueError("Completed run history is not readable")
        return  # Legacy sidecar write is non-atomic; next poll/mandatory final read recovers.
    if strict and [r["epoch"] for r in history] != list(range(1, 81)):
        raise ValueError("Completed run must retain all 80 epoch records")
    for row in history:
        if row["epoch"] <= 60:
            continue
        epoch = int(row["epoch"])
        provenance = {"source": str(path), "selection_manifest_sha256": manifest_hash,
                      "metric_scope": "valid", "epoch": epoch}
        measured = {dst: {"value": row["valid"][src], "provenance": provenance}
                    for src, dst in METRICS.items()}
        measured.update({k: {"value": v, "provenance": provenance} for k, v in {
            "params": selected["Params"], "macs": selected["MACs"], "flops": 2 * selected["MACs"],
            "search_time_seconds": float(row["wall_seconds"]),
            "gpu_hours": float(row["wall_seconds"]) / 3600}.items()})
        record = build_metric_record(task_type="detection", dataset="Cylinder2", network=selected["architecture_id"],
            seed=42, stage="phase6b_continuation_epoch", identity=dict(
                run_id=f"{VERSION}:{selected['architecture_id']}:seed42", epoch=epoch,
                selection_manifest_sha256=manifest_hash, parent_checkpoint_sha256=selected["parent_checkpoint_sha256"],
                source_binding=expected_binding, test_evaluated=False, clean_reference_available=False,
                raw_epoch_record=row,
                gpu_time_scope="training plus valid wall time only; excludes checkpoint saving and process startup"), measured=measured)
        require_measured_metrics(record, METRICS.values())
        append_metric_record(ledger, record)


def run(manifest_path, parent_root, gpus):
    with supervisor_lock(ROOT / "results/phase6b_rule_loop_supervisor.lock"):
        return _run_locked(manifest_path, parent_root, gpus)


def _run_locked(manifest_path, parent_root, gpus):
    import torch
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    digest = sha(manifest_path)
    if digest != read(manifest_path.with_name("selection_manifest.sha256.json"))["sha256"]:
        raise ValueError("Selection manifest hash drift")
    if manifest["version"] != VERSION or len(manifest["selected"]) != 6:
        raise ValueError("Only this bounded six-candidate protocol is permitted")
    parent_root = Path(parent_root).resolve()
    if ROOT.resolve() == parent_root:
        raise ValueError("Run requires an isolated root; historical results cannot be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_ENVIRONMENT_NOT_READY; no CPU formal fallback")
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Exactly two distinct GPU workers are required")
    sys.path.insert(0, str(ROOT / "tools/sonar_spec"))
    import phase2e_runner as trainer
    trainer.load()
    if trainer.binding() != manifest["source_binding"]:
        raise ValueError("Frozen trainer source/protocol drift")
    out = manifest_path.parent
    staging = []
    # Validate and copy ALL parents before any GPU job. Never hardlink mutable artifacts.
    for selected in manifest["selected"]:
        aid = selected["architecture_id"]
        source = parent_root / "results/phase2e_independent" / aid
        target = ROOT / "results/phase2e_independent" / aid
        if sha(source / "latest.pt") != selected["parent_checkpoint_sha256"]:
            raise ValueError("Historical parent latest checkpoint changed")
        state = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
        check_state(state, selected, manifest["source_binding"], 60)
        expected_optimizer_hash = state.get("optimizer_state_sha256")
        if expected_optimizer_hash is not None and trainer.state_hash(state["optimizer_state_dict"]) != expected_optimizer_hash:
            raise ValueError("Parent optimizer state checksum mismatch")
        if read(source / "epoch_log.json") != state["history"]:
            raise ValueError("Parent history sidecar disagrees with committed checkpoint")
        best = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
        best_row = max(state["history"], key=lambda r: r["valid"]["mAP50_95"])
        check_state(best, selected, manifest["source_binding"], best_row["epoch"])
        if best["history"][-1]["valid"] != best_row["valid"]:
            raise ValueError("Historical best checkpoint does not match parent best metrics")
        copied = {name: sha(source / name) for name in ("latest.pt", "best.pt", "epoch_log.json", "model_info.json")}
        marker = target / "phase6b_parent.json"
        expected_marker = dict(selection_manifest_sha256=digest, parent_files=copied)
        if target.exists():
            if not marker.exists() or read(marker) != expected_marker:
                raise ValueError("Existing target has no matching isolated-run identity")
        else:
            target.mkdir(parents=True)
            for name, checksum in copied.items():
                shutil.copy2(source / name, target / name)
                if sha(target / name) != checksum:
                    raise ValueError("Parent copy checksum mismatch")
            write(marker, expected_marker)
        staging.append(dict(architecture_id=aid, parent_root=str(parent_root), parent_files=copied))
    write(out / "execution_manifest.json", dict(selection_manifest_sha256=digest, parents=staging,
        wrapper_sha256=sha(__file__), ledger_sha256=sha(ROOT / "src/hwnas_fpga/metrics_ledger.py"),
        source_binding=manifest["source_binding"], gpu_workers=gpus, started_at=now(),
        reference_provenance=reference_provenance(), test_evaluated=False))
    jobs = queue.Queue()
    for selected in manifest["selected"]:
        jobs.put(selected)
    failed = threading.Event()

    def worker(gpu):
        while not failed.is_set():
            try:
                selected = jobs.get_nowait()
            except queue.Empty:
                return
            aid = selected["architecture_id"]
            folder = ROOT / "results/phase2e_independent" / aid
            command = [sys.executable, "-u", str(ROOT / "tools/sonar_spec/phase2e_runner.py"),
                       "train", "--architecture-id", aid, "--budget", "80", "--device", "cuda"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
            tick = time.monotonic()
            log_path = out / "logs" / f"{aid}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            process = None
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        mirror_history(folder, selected, digest, out / "METRICS.jsonl", manifest["source_binding"])
                        time.sleep(5)
                    if process.returncode:
                        raise RuntimeError(f"Frozen trainer exit={process.returncode}; see {log_path}")
                mirror_history(folder, selected, digest, out / "METRICS.jsonl", manifest["source_binding"], strict=True)
                state = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
                check_state(state, selected, manifest["source_binding"], 80)
                best = max(state["history"], key=lambda r: r["valid"]["mAP50_95"])
                elapsed = time.monotonic() - tick
                result = dict(architecture_id=aid, epoch=80, seed=42, best_epoch=best["epoch"],
                    **{k: best["valid"][k] for k in METRICS}, endpoint_metrics=state["history"][-1]["valid"],
                    Params=selected["Params"], MACs=selected["MACs"], FLOPs=2 * selected["MACs"],
                    checkpoint_sha256=sha(folder / "latest.pt"), best_checkpoint_sha256=sha(folder / "best.pt"),
                    history_sha256=sha(folder / "epoch_log.json"), parent_checkpoint_sha256=selected["parent_checkpoint_sha256"],
                    selection_manifest_sha256=digest, elapsed_invocation_seconds=elapsed,
                    total_reserved_gpu_hours=elapsed / 3600,
                    total_reserved_gpu_hours_scope="this worker invocation wall time including startup, training, validation, checkpoint saving and final record collection",
                    training_validation_gpu_hours=sum(r["wall_seconds"] for r in state["history"][60:]) / 3600,
                    training_validation_gpu_hours_scope="all epochs61-80 training/valid wall time; excludes checkpoint saving/startup; may span resumed invocations",
                    source_binding=manifest["source_binding"], gpu=gpu, test_evaluated=False)
                write(out / "completed" / f"{aid}.json", result)
            except BaseException as exc:
                failed.set()
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=60)
                write(out / "failures" / f"{aid}.json", dict(error=str(exc), at=now(), test_evaluated=False))
                raise
            finally:
                jobs.task_done()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    return summarize(manifest_path)


def summarize(manifest_path):
    path = Path(manifest_path)
    manifest = read(path)
    rows = []
    for selected in manifest["selected"]:
        result = path.parent / "completed" / (selected["architecture_id"] + ".json")
        if result.exists():
            row = read(result)
            if row["selection_manifest_sha256"] != sha(path) or row["architecture_id"] != selected["architecture_id"]:
                raise ValueError("Result manifest binding drift")
            rows.append(row)
    decision = screening_decision(rows, manifest["reference"])
    decision["completed"] = len(rows)
    decision["reference_provenance"] = reference_provenance()
    write(path.parent / "SCREENING_GATE.json", decision)
    write(path.parent / "RESULTS.json", rows)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--preflight", type=Path, required=True)
    prep.add_argument("--stage2", type=Path, default=ROOT / "reports/phase2e_multifidelity/stage2_results.csv")
    prep.add_argument("--final", type=Path, default=ROOT / "reports/phase2e_multifidelity/final_results.csv")
    prep.add_argument("--output", type=Path, default=ROOT / "reports/phase6b_rule_loop")
    execute = subs.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--parent-root", type=Path, required=True)
    execute.add_argument("--gpus", default="0,1")
    summary = subs.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.preflight, args.stage2, args.final, args.output)
    elif args.mode == "run":
        result = run(args.manifest, args.parent_root, [int(x) for x in args.gpus.split(",")])
    else:
        result = summarize(args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
