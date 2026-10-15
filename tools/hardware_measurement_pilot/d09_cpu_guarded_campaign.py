"""Quiet-v3 D09 orchestration: new-boot ownership guards; unchanged measurement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from tools.hardware_measurement_pilot import board_runner, d09_cpu_guarded_transport as transfer

RESULT = Path("results/d09_cpu_freeze_20261013/quiet_v3").resolve()
PILOT = Path("results/hardware_measurement_pilot_20261012/attempt05/board_packages").resolve()
CASES = {
    "pw1x1_int8_reference_high": "20261012T150909489320Z",
    "dw5_int8_reference_low": "20261012T145119529880Z",
    "body8_to_head16_reference_high": "20261012T141130951567Z",
}
BOARD_ID = "machine-id:bd081b0be1fd457093e07237b0dee576"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def event(directory, name, **details):
    record = dict(event=name, created_utc=datetime.now(timezone.utc).isoformat(), **details)
    with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def prepare(case_id, phase, run_id):
    if case_id not in CASES or phase not in ("calibration", "validation"):
        raise ValueError("campaign scope differs")
    transfer.valid_job(run_id)
    source = PILOT / case_id / CASES[case_id] / "case"
    _, old_bundle, case, _, _ = board_runner.validate_bundle(source)
    target = RESULT / "packages" / run_id / "case"
    target.mkdir(parents=True, exist_ok=False)
    for name, digest in old_bundle["files"].items():
        src, dst = transfer.child(source, name), transfer.child(target, name)
        if sha(src) != digest:
            raise ValueError("original pilot package changed")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    extras = ["d09_protocol.json", "pl_baseline.json", "user_authorization.json", "external_conditions.json", "board_ownership.json"]
    for name in extras:
        shutil.copyfile(RESULT / name, target / name)
    for name in ("d09_cpu_runner.py", "cpu_conditions.py", "d09_board_ownership_v1.py"):
        shutil.copyfile(Path(__file__).parent / name, target / name)
    protocol = load(target / "d09_protocol.json")
    ownership_ref = dict(path="board_ownership.json", sha256=sha(target / "board_ownership.json"))
    if protocol.get("board_ownership") != ownership_ref:
        raise ValueError("quiet-v3 protocol does not bind the exact ownership config")
    for name in ("d09_cpu_runner.py", "cpu_conditions.py", "d09_board_ownership_v1.py",
                 "d09_cpu_guarded_transport.py", "d09_cpu_guarded_campaign.py"):
        if protocol.get("source_files", {}).get(name) != sha(Path(__file__).parent / name):
            raise ValueError("quiet-v3 source differs from the pre-sampling protocol: " + name)
    auth = load(target / "user_authorization.json")
    baseline = load(target / "pl_baseline.json")["cases"][case_id]
    config = dict(schema="d09_cpu_conditions_v1", campaign_id="d09_cpu_freeze_20261013",
        run_id=run_id, experiment_phase=phase, cpu_index=3, policy_name="policy0", mode="userspace",
        frequency_khz=1199999, batches=3, warmup_calls=50, measured_calls=1000, group_size=1,
        userspace_authorization=dict(authorized=auth["authorized"], source=auth["source"]),
        protocol=dict(path="d09_protocol.json", sha256=sha(target / "d09_protocol.json")),
        pl_baseline=dict(path="pl_baseline.json", sha256=sha(target / "pl_baseline.json")),
        pl_compute_expected_cycles=baseline["compute_cycles"],
        external_conditions_evidence=dict(path="external_conditions.json", sha256=sha(target / "external_conditions.json")),
        host_median_relative_limit=None, system_conditions=protocol.get("system_conditions", {}),
        expected_versions=protocol.get("expected_versions", {}), runtime_files=protocol.get("runtime_files", {}), case_id=case_id,
        board_ownership=ownership_ref)
    if phase == "validation":
        frozen = load(RESULT / "host_thresholds.json")
        if frozen.get("status") != "PASS_CALIBRATION_THRESHOLDS_LOCKED":
            raise ValueError("calibration not passed; validation forbidden")
        shutil.copyfile(RESULT / "host_thresholds.json", target / "host_thresholds.json")
        extras.append("host_thresholds.json")
        config["calibration_freeze"] = dict(path="host_thresholds.json", sha256=sha(target / "host_thresholds.json"))
        config["host_median_relative_limit"] = frozen["host_median_relative_limit"]
    save(target / "cpu_conditions.json", config)
    bundle = dict(old_bundle, files=dict(old_bundle["files"]))
    for name in extras + ["d09_cpu_runner.py", "cpu_conditions.py", "cpu_conditions.json", "d09_board_ownership_v1.py"]:
        bundle["files"][name] = sha(target / name)
    save(target / "board_bundle.json", bundle)
    board_runner.validate_bundle(target)
    record = dict(case_id=case_id, case_sha256=case["case_sha256"], phase=phase, run_id=run_id,
        source_case=str(source), source_bundle_sha256=sha(source / "board_bundle.json"),
        new_bundle_sha256=sha(target / "board_bundle.json"), case_dir=str(target))
    save(target.parent / "package_provenance.json", record)
    return record


def run(package_dir, timeout_seconds=600):
    root = Path(package_dir).resolve()
    if not root.is_relative_to(RESULT / "packages"):
        raise ValueError("package outside new campaign")
    config = load(root / "cpu_conditions.json")
    if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 3600):
        raise ValueError("finite bounded timeout in (0,3600] required")
    if config.get("case_id") not in CASES or config.get("experiment_phase") not in ("calibration", "validation"):
        raise ValueError("case/phase outside authorized campaign")
    run_id = transfer.valid_job(config["run_id"])
    provenance = load(root.parent / "package_provenance.json")
    if (root.parent.name != run_id or provenance.get("run_id") != run_id
            or provenance.get("case_id") != config["case_id"]
            or provenance.get("phase") != config["experiment_phase"]
            or provenance.get("new_bundle_sha256") != sha(root / "board_bundle.json")):
        raise ValueError("package provenance differs from requested run")
    execution = RESULT / "executions" / run_id
    execution.mkdir(parents=True, exist_ok=False)
    event(execution, "TRANSPORT_PREPARE", case_id=config.get("case_id"), phase=config["experiment_phase"])
    transport_dir = execution / "transport"
    transfer.prepare(root, transport_dir, mode="timing", chunk_bytes=4 * 1024 * 1024,
                     bridge=str(Path(__file__).with_name("board_bridge_stdin.py").resolve()))
    event(execution, "UPLOAD_START")
    transfer.upload(transport_dir)
    folder, transport_config = transfer.config_for(transport_dir)
    event(execution, "LAUNCH_START", run_id=run_id)
    receipt = transfer.call(folder, transport_config, "launch", job_id=run_id,
                            board_id=BOARD_ID, timeout_seconds=30.0)
    save(execution / "launch.json", receipt)
    deadline = time.monotonic() + timeout_seconds
    terminal = {"PASS", "FAIL", "CPU_CONDITIONS_UNKNOWN", "DEVICE_STATE_UNKNOWN", "WORKER_FAILED_LOCK_RETAINED", "DMA_QUARANTINED"}
    while True:
        observed = transfer.call(folder, transport_config, "status", job_id=run_id)
        state = observed["state"]
        event(execution, "STATUS", state=state)
        if state in terminal:
            break
        if time.monotonic() >= deadline:
            save(execution / "bounded_wait_expired.json", dict(state=state, no_reset_or_kill=True))
            raise RuntimeError("bounded wait expired; inspect job, never relaunch")
        time.sleep(180)
    event(execution, "FETCH_START", state=state)
    fetched = transfer.fetch(transport_dir, run_id, execution / "snapshot")
    event(execution, "FETCH_DONE", state=state, **fetched)
    save(execution / "result.json", dict(state=state, snapshot=fetched))
    if state != "PASS":
        raise RuntimeError("retained nonpassing run: " + str(state))
    return dict(status="PASS_FETCHED_NOT_YET_INDEPENDENTLY_VALIDATED", run_id=run_id, execution=str(execution))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--phase", choices=("calibration", "validation"), required=True)
    p.add_argument("--run-id", required=True)
    p = sub.add_parser("run")
    p.add_argument("--case-dir", required=True)
    p.add_argument("--timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    result = prepare(args.case, args.phase, args.run_id) if args.action == "prepare" else run(args.case_dir, args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
