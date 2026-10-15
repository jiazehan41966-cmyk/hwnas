"""Launch only the frozen COCO seed42 run after independent Linux CUDA gates.

The frozen baseline code and inputs remain unchanged. Remote measurements use
the same protocol with a separately bound environment; they are not a paired
estimate of the pretraining effect against local scratch measurements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = ROOT / "results/sonar_yolov9_baseline_20261005"
DEFAULT_RESULT = CAMPAIGN_ROOT / "remote_coco"
ENV_SPEC_PATH = CAMPAIGN_ROOT / "remote_environment_spec.json"
ENV_LOCK_PATH = CAMPAIGN_ROOT / "remote_environment_lock.json"
GROUPS = ("L1_scratch", "YOLOv9_T_scratch", "YOLOv9_T_COCO")
FORMAL_GROUP = "YOLOv9_T_COCO"
INTERPRETATION = "same_protocol_different_environment_not_paired_pretraining_effect"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _option(args: list[str], name: str) -> str | None:
    """Read a real argv option, never substring-match shell command text."""
    for index, argument in enumerate(args):
        if argument == name and index + 1 < len(args):
            return args[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def duplicate_coco_processes(source_root: Path, proc_root: Path = Path("/proc")) -> list[dict]:
    """Find real train.py COCO invocations of this checkout, excluding self."""
    script = (source_root / "tools/sonar_yolov9_baseline/train.py").resolve()
    duplicates = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = [part.decode("utf-8", errors="surrogateescape")
                         for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
            if not arguments:
                continue
            # Script arguments are separate /proc tokens. A shell -c command
            # or this launcher's own source text can never match this test.
            for index, argument in enumerate(arguments):
                if Path(argument).name != script.name:
                    continue
                candidate = Path(argument)
                if not candidate.is_absolute():
                    candidate = (directory / "cwd").resolve() / candidate
                if candidate.resolve() != script:
                    continue
                tail = arguments[index + 1:]
                if not tail or tail[0] != "train" or _option(tail, "--group") != FORMAL_GROUP:
                    continue
                duplicates.append({
                    "pid": int(directory.name), "argv": arguments,
                    "result_root_argument": _option(tail, "--result-root"),
                })
                break
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited while enumerating.
        except PermissionError as exc:
            raise RuntimeError(f"Cannot inspect /proc process {directory.name}; duplicate gate unavailable") from exc
    return duplicates


def query_gpu(gpu_index: int) -> dict:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total,utilization.gpu,uuid,name",
        "--format=csv,noheader,nounits",
    ], text=True)
    matches = []
    for row in csv.reader(raw.splitlines()):
        values = [value.strip() for value in row]
        if len(values) != 7:
            raise ValueError("Unexpected nvidia-smi GPU query schema")
        if int(values[0]) == gpu_index:
            matches.append(dict(index=int(values[0]), free_mib=int(values[1]),
                                used_mib=int(values[2]), total_mib=int(values[3]),
                                utilization_percent=int(values[4]), uuid=values[5], name=values[6]))
    if len(matches) != 1:
        raise ValueError(f"GPU {gpu_index} absent or ambiguous")
    return matches[0]


def verify_smokes_and_review(result: Path, manifest_sha: str, environment: dict) -> dict:
    spec = read_json(ENV_SPEC_PATH)
    spec_sha = canonical_sha(spec)
    lock = read_json(ENV_LOCK_PATH)
    lock_sha = sha(ENV_LOCK_PATH)
    if lock.get("remote_environment_spec_canonical_sha256") != spec_sha:
        raise ValueError("Remote environment lock does not bind the exact canonical specification")
    if lock.get("environment") != environment:
        raise ValueError("Current Linux CUDA environment differs from the remote environment lock")
    if lock.get("test_evaluated") is not False or lock.get("local_frozen_spec_is_provenance_only") is not True:
        raise ValueError("Remote environment evidence boundaries are absent")
    review_path = result / "independent_review.json"
    review = read_json(review_path)
    if (review.get("status") != "PASS"
            or review.get("execution_manifest_sha256") != manifest_sha
            or review.get("remote_environment_spec_canonical_sha256") != spec_sha
            or review.get("remote_environment_lock_sha256") != lock_sha):
        raise ValueError("Independent review must pass for this manifest, remote spec and environment lock")
    smokes, rng, bindings = [], [], {}
    for group in GROUPS:
        directory = result / "smoke" / group / "seed_42"
        smoke_path = directory / "smoke_report.json"
        report = read_json(smoke_path)
        expected = dict(execution_manifest_sha256=manifest_sha, group=group,
                        seed=42, mode="smoke", test_evaluated=False)
        if (report.get("status") != "PASS" or report.get("formal_eligible") is not False
                or report.get("binding") != expected
                or report.get("environment") != environment
                or report["environment"].get("device") != "cuda"
                or report.get("test_evaluated") is not False
                or report["training"].get("gradient_finite") is not True
                or int(report.get("peak_cuda_reserved_bytes", 0)) <= 0):
            raise ValueError(f"Required real CUDA smoke is missing, changed or invalid: {group}")
        smokes.append(report)
        rng_path = directory / "training_rng_identity.json"
        rng.append(read_json(rng_path))
        bindings[group] = dict(smoke_report_sha256=sha(smoke_path), training_rng_identity_sha256=sha(rng_path))
    if len({report["training"]["image_order_sha256"] for report in smokes}) != 1:
        raise ValueError("Three remote CUDA smokes used different sampled image orders")
    if any(identity != rng[0] for identity in rng[1:]):
        raise ValueError("Three remote CUDA smokes used different training RNG identities")
    required = max(int(report["peak_cuda_reserved_bytes"]) for report in smokes)
    return dict(
        remote_environment_spec_canonical_sha256=spec_sha,
        remote_environment_lock_sha256=lock_sha,
        independent_review_sha256=sha(review_path),
        smoke_bindings=bindings,
        sampled_image_order_sha256=smokes[0]["training"]["image_order_sha256"],
        training_rng_identity_canonical_sha256=canonical_sha(rng[0]),
        required_free_bytes=required,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    if platform.system() != "Linux" or not Path("/proc").is_dir():
        raise RuntimeError("This launcher requires Linux /proc; no platform fallback")
    if args.gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    result = args.result_root.expanduser().resolve()
    launch_path = result / "launch.json"
    formal = result / "formal" / FORMAL_GROUP / "seed_42"
    if launch_path.exists():
        raise FileExistsError("Existing remote launch receipt cannot be overwritten")
    if formal.exists() and any(formal.iterdir()):
        raise FileExistsError("COCO formal output already exists; automatic overwrite or resume is forbidden")
    # Configure visibility before the frozen module imports torch. These are
    # resource overrides, and their actual values must match the remote lock.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path.insert(0, str(ROOT))
    from tools.sonar_yolov9_baseline import train
    train.torch.set_num_threads(2)
    train.verify_manifest(result)
    manifest_sha = sha(result / "execution_manifest.json")
    actual_environment = train.environment("cuda")
    readiness = verify_smokes_and_review(result, manifest_sha, actual_environment)
    duplicates = duplicate_coco_processes(ROOT)
    if duplicates:
        raise RuntimeError(f"Duplicate COCO training process in this checkout: {duplicates}")
    gpu = query_gpu(args.gpu)
    if gpu["utilization_percent"] > 10 or gpu["free_mib"] * 1048576 < readiness["required_free_bytes"]:
        raise RuntimeError("GPU requires utilization <=10 percent and enough free bytes for observed CUDA smoke peak")
    command = [sys.executable, str(ROOT / "tools/sonar_yolov9_baseline/train.py"), "train",
               "--result-root", str(result), "--group", FORMAL_GROUP,
               "--seed", "42", "--device", "cuda"]
    environment = os.environ.copy()
    formal.mkdir(parents=True, exist_ok=True)
    log_path = formal / "stdout.log"
    receipt = dict(
        schema_version=1, group=FORMAL_GROUP, seed=42, epochs=80,
        execution_manifest_sha256=manifest_sha, **readiness,
        launcher_sha256=sha(Path(__file__)), environment=actual_environment,
        gpu=args.gpu, gpu_prelaunch=gpu,
        gpu_idle_policy="utilization_at_most10_and_free_memory_at_least_observed_smoke_peak",
        command=command, cwd=str(ROOT), result_root=str(result), stdout=str(log_path),
        start_new_session=True, test_evaluated=False, interpretation=INTERPRETATION,
        paired_pretraining_effect_claimed=False,
        launched_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    # Exclusive creation reserves this invocation before Popen. The file is
    # written exactly once with the outcome, preventing concurrent re-launches.
    with launch_path.open("x", encoding="utf-8") as launch_handle:
        try:
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            receipt["pid"] = process.pid
            time.sleep(2)
            exit_code = process.poll()
            receipt.update(status="LAUNCHED" if exit_code is None else "FAILED_IMMEDIATE_EXIT",
                           live_after_seconds=2, live_after_check=exit_code is None,
                           immediate_exit_code=exit_code)
        except Exception as error:
            receipt.update(status="FAILED_TO_LAUNCH", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            json.dump(receipt, launch_handle, ensure_ascii=False, indent=2, allow_nan=False)
            launch_handle.flush()
            os.fsync(launch_handle.fileno())
    if receipt["status"] != "LAUNCHED":
        raise RuntimeError("COCO process exited within two seconds; inspect retained launch receipt and stdout")
    print(json.dumps(dict(status=receipt["status"], pid=receipt["pid"], gpu=args.gpu,
                          result_root=str(result), execution_manifest_sha256=manifest_sha,
                          interpretation=INTERPRETATION)), flush=True)


if __name__ == "__main__":
    main()
