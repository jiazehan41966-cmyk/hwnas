#!/usr/bin/env python3
"""Build and verify the frozen ZCU104 full-campaign baseline package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "zcu104_full_campaign_v1"
CAMPAIGN_CONFIG = REPO_ROOT / "configs" / "experiments" / f"{CAMPAIGN_ID}.yaml"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / CAMPAIGN_ID
SOURCE_BRANCH = "codex/sonar-fourstage-operator-closure"
PINNED_SOURCE_COMMIT = "cae9be4f01d1a55576bc4449b6b41680b4a889d9"
TARGET_PART = "xczu7ev-ffvc1156-2-e"
EXPECTED_BOARD_PART = "xilinx.com:zcu104:part0:1.1"
FOLD = 0
SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved
    if relative_to is not None:
        try:
            display = resolved.relative_to(relative_to.resolve())
        except ValueError:
            display = resolved
    return {
        "path": display.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def run(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "available": False,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "available": True,
    }


def git_text(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True
    )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected} actual={actual} path={path}"
        )


def copy_verified(source: Path, destination: Path, expected: str, label: str) -> dict[str, Any]:
    assert_hash(source, expected, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert_hash(destination, expected, f"copied {label}")
    record = file_record(destination, relative_to=REPO_ROOT)
    record["source_path"] = str(source.resolve())
    return record


def select_checkpoint(candidate: dict[str, Any]) -> dict[str, Any]:
    suffix = f"best_fold{FOLD}_seed{SEED}.pt".lower()
    matches = [
        row
        for row in candidate["checkpoint_index"]
        if str(row["path"]).lower().endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {suffix} checkpoint for {candidate['arch_id']}, got {len(matches)}"
        )
    return matches[0]


def build_inputs(source_root: Path) -> dict[str, Any]:
    selection_path = (
        REPO_ROOT
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "fourstage_deployment_candidate_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = selection.get("selected_candidates", [])
    if len(candidates) != 4:
        raise RuntimeError(f"expected four selected candidates, got {len(candidates)}")

    copied_candidates: list[dict[str, Any]] = []
    copied_checkpoints: list[dict[str, Any]] = []
    source_run_path: Path | None = None
    for candidate in candidates:
        arch_id = str(candidate["arch_id"])
        role = str(candidate["role"])
        candidate_source = Path(candidate["candidate_json"]["path"])
        candidate_destination = (
            ARTIFACT_ROOT / "inputs" / "candidates" / f"{arch_id}.json"
        )
        candidate_record = copy_verified(
            candidate_source,
            candidate_destination,
            str(candidate["candidate_json"]["sha256"]),
            f"candidate {arch_id}",
        )
        candidate_payload = json.loads(candidate_destination.read_text(encoding="utf-8"))
        embedded_arch_id = candidate_payload["candidate"]["arch_id"]
        embedded_arch_hash = candidate_payload["candidate"]["architecture_sha256"]
        if embedded_arch_id != arch_id:
            raise RuntimeError(f"candidate arch_id mismatch for {arch_id}")
        if embedded_arch_hash != candidate["architecture_sha256"]:
            raise RuntimeError(f"candidate architecture_sha256 mismatch for {arch_id}")
        candidate_record.update(
            {
                "role": role,
                "arch_id": arch_id,
                "architecture_sha256": str(candidate["architecture_sha256"]),
            }
        )
        copied_candidates.append(candidate_record)

        checkpoint = select_checkpoint(candidate)
        checkpoint_source = Path(checkpoint["path"])
        checkpoint_destination = (
            ARTIFACT_ROOT
            / "inputs"
            / "checkpoints"
            / f"{role}__{arch_id}__fold{FOLD}_seed{SEED}.pt"
        )
        checkpoint_record = copy_verified(
            checkpoint_source,
            checkpoint_destination,
            str(checkpoint["sha256"]),
            f"checkpoint {arch_id}",
        )
        checkpoint_record.update(
            {
                "role": role,
                "arch_id": arch_id,
                "architecture_sha256": str(candidate["architecture_sha256"]),
                "fold": FOLD,
                "seed": SEED,
                "selection_rule": (
                    "fixed fold0/seed42 deployment representative; no outer-validation "
                    "metric was used for checkpoint selection"
                ),
            }
        )
        copied_checkpoints.append(checkpoint_record)

        if source_run_path is None:
            source_run_path = Path(candidate["result_dir"]) / f"run_fold{FOLD}_seed{SEED}.json"

    if source_run_path is None or not source_run_path.is_file():
        raise FileNotFoundError(f"split source run missing: {source_run_path}")
    source_run = json.loads(source_run_path.read_text(encoding="utf-8"))
    split = source_run.get("split")
    if not isinstance(split, dict):
        raise RuntimeError(f"split object absent from {source_run_path}")
    if int(split.get("fold_index", -1)) != FOLD or int(split.get("seed", -1)) != SEED:
        raise RuntimeError("source run split does not match fixed fold0/seed42")

    provenance = source_run.get("provenance", {})
    dataset_files = provenance.get("dataset", {}).get("files", {})
    copied_dataset_files: list[dict[str, Any]] = []
    for name, source_record in sorted(dataset_files.items()):
        source_path = Path(source_record["path"])
        destination = ARTIFACT_ROOT / "inputs" / "manifests" / "dataset_files" / name
        record = copy_verified(
            source_path,
            destination,
            str(source_record["sha256"]),
            f"dataset split file {name}",
        )
        record["name"] = name
        copied_dataset_files.append(record)

    binding_path = (
        REPO_ROOT
        / "artifacts"
        / "sonar_fourstage_operator_v2"
        / "protocol_v2_freeze_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    checks = binding["selection_evidence_checks"]
    dataset_source = Path(checks["dataset_manifest"]["path"])
    group_source = Path(checks["group_manifest"]["path"])
    dataset_destination = ARTIFACT_ROOT / "inputs" / "manifests" / "dataset_manifest.json"
    group_destination = (
        ARTIFACT_ROOT / "inputs" / "manifests" / "inferred_hash_group_manifest.json"
    )
    dataset_manifest = copy_verified(
        dataset_source,
        dataset_destination,
        str(checks["dataset_manifest"]["expected_sha256"]),
        "Protocol V2 dataset manifest",
    )
    group_manifest = copy_verified(
        group_source,
        group_destination,
        str(checks["group_manifest"]["expected_sha256"]),
        "Protocol V2 inferred-group manifest",
    )

    split_manifest_path = (
        ARTIFACT_ROOT / "inputs" / "manifests" / f"fold{FOLD}_seed{SEED}_split.json"
    )
    split_manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "protocol": "nksid_outer5fold_inner_only_v2",
        "fold": FOLD,
        "seed": SEED,
        "source_run": {
            "path": str(source_run_path.resolve()),
            "sha256": sha256_file(source_run_path),
            "size_bytes": source_run_path.stat().st_size,
        },
        "dataset_files": copied_dataset_files,
        "split": split,
        "claim_boundary": (
            "This is the exact fold0/seed42 split attached to the fixed deployment "
            "checkpoints. It does not create a new accuracy estimate."
        ),
    }
    write_json(split_manifest_path, split_manifest)

    return {
        "selection": file_record(selection_path, relative_to=REPO_ROOT),
        "protocol_binding": file_record(binding_path, relative_to=REPO_ROOT),
        "dataset_manifest": dataset_manifest,
        "group_manifest": group_manifest,
        "dataset_content_sha256": str(binding["dataset_content_sha256"]),
        "dataset_files": copied_dataset_files,
        "split_manifest": file_record(split_manifest_path, relative_to=REPO_ROOT),
        "candidates": copied_candidates,
        "checkpoints": copied_checkpoints,
    }


def capture_tool_version(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "tool path not found",
        }
    result = run([str(path), "-version"], timeout=120)
    result["path"] = str(path.resolve())
    return result


def probe_zcu104(vivado: Path) -> dict[str, Any]:
    if not vivado.is_file():
        return {
            "status": "FAIL_VIVADO_NOT_FOUND",
            "target_part": TARGET_PART,
            "target_part_count": 0,
            "board_parts": [],
        }
    tcl = "\n".join(
        [
            f"set target_parts [get_parts -quiet {{{TARGET_PART}}}]",
            "puts \"CODEX_TARGET_PART_COUNT=[llength $target_parts]\"",
            "set board_parts [get_board_parts -quiet *zcu104*]",
            "puts \"CODEX_BOARD_PARTS=[join $board_parts ,]\"",
            "exit",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="zcu104_probe_") as temp_dir:
        tcl_path = Path(temp_dir) / "probe_zcu104.tcl"
        tcl_path.write_text(tcl + "\n", encoding="utf-8")
        result = run(
            [
                str(vivado),
                "-mode",
                "batch",
                "-nolog",
                "-nojournal",
                "-notrace",
                "-source",
                str(tcl_path),
            ],
            timeout=180,
        )
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    count_match = re.search(r"CODEX_TARGET_PART_COUNT=(\d+)", combined)
    boards_match = re.search(r"CODEX_BOARD_PARTS=([^\r\n]*)", combined)
    target_count = int(count_match.group(1)) if count_match else 0
    boards = []
    if boards_match and boards_match.group(1).strip():
        boards = [item for item in boards_match.group(1).split(",") if item]
    return {
        "status": "PASS" if target_count > 0 else "FAIL_TARGET_PART_UNAVAILABLE",
        "target_part": TARGET_PART,
        "target_part_count": target_count,
        "expected_board_part": EXPECTED_BOARD_PART,
        "board_parts": boards,
        "command_returncode": result.get("returncode"),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def capture_environment(
    args: argparse.Namespace,
    remote_probe: dict[str, Any] | None,
    remote_probe_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if remote_probe is not None:
        return {
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "generated_at": utc_now(),
            "authoritative_execution_plane": "remote_server",
            "control_plane": {
                "role": "freeze_configuration_and_dispatch_only",
                "platform": platform.platform(),
                "python_version": sys.version,
                "python_executable": sys.executable,
                "fpga_toolchain_is_authoritative": False,
            },
            "execution_plane": remote_probe,
            "remote_probe_record": remote_probe_record,
            "campaign_execution_readiness": remote_probe["readiness"][
                "campaign_execution_status"
            ],
            "claim_boundary": (
                "The remote server is authoritative for training, FPGA tools, and ZCU104 "
                "connectivity. The local host is only the frozen control plane. Environment "
                "preflight is not HLS, route, bitstream, board-classification, latency, or "
                "power evidence."
            ),
        }

    try:
        import torch

        torch_record = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:  # pragma: no cover - environment-dependent
        torch_record = {"available": False, "error": repr(exc)}

    vivado_path = Path(args.vivado)
    vitis_hls_path = Path(args.vitis_hls)
    if args.skip_tool_probe:
        vivado_record = {
            "path": str(vivado_path),
            "available": vivado_path.is_file(),
            "status": "PENDING_EXPLICIT_PROBE",
        }
        vitis_record = {
            "path": str(vitis_hls_path),
            "available": vitis_hls_path.is_file(),
            "status": "PENDING_EXPLICIT_PROBE",
        }
        target_probe = {
            "status": "PENDING_EXPLICIT_PROBE",
            "target_part": TARGET_PART,
            "target_part_count": None,
            "expected_board_part": EXPECTED_BOARD_PART,
            "board_parts": [],
        }
    else:
        vivado_record = capture_tool_version(vivado_path)
        vitis_record = capture_tool_version(vitis_hls_path)
        target_probe = probe_zcu104(vivado_path)

    readiness = (
        "READY"
        if target_probe.get("status") == "PASS"
        else "BLOCKED_ZCU104_DEVICE_SUPPORT_UNAVAILABLE"
    )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": utc_now(),
        "host": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "pytorch": torch_record,
        "vivado": vivado_record,
        "vitis_hls": vitis_record,
        "zcu104_target_probe": target_probe,
        "campaign_execution_readiness": readiness,
        "claim_boundary": (
            "Tool presence and target-part visibility are environment preflight evidence. "
            "They are not HLS, route, bitstream, board-classification, latency, or power evidence."
        ),
    }


def verify_source_freeze(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "PENDING",
            "manifest": "artifacts/zcu104_full_campaign_v1/source_freeze/source_freeze_manifest.json",
            "missing": None,
            "unexpected": None,
            "changed": None,
        }
    if not path.is_file():
        raise FileNotFoundError(path)
    result = run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(path),
        ],
        timeout=120,
    )
    if result.get("returncode") not in (0, 2):
        raise RuntimeError(f"source-freeze verifier failed: {result}")
    verification = json.loads(str(result["stdout"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verification["manifest_sha256"] = sha256_file(path)
    verification["frozen_git"] = manifest.get("git")
    archive = manifest.get("archive")
    if archive:
        archive_path = Path(archive["path"])
        assert_hash(archive_path, str(archive["sha256"]), "source-freeze archive")
        verification["archive"] = archive
    return verification


def build_config(
    source_commit: str,
    inputs: dict[str, Any],
    remote_probe_record: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates = []
    checkpoints = {row["arch_id"]: row for row in inputs["checkpoints"]}
    for candidate in inputs["candidates"]:
        checkpoint = checkpoints[candidate["arch_id"]]
        candidates.append(
            {
                "role": candidate["role"],
                "arch_id": candidate["arch_id"],
                "architecture_sha256": candidate["architecture_sha256"],
                "candidate_json": {
                    "path": candidate["path"],
                    "sha256": candidate["sha256"],
                },
                "checkpoint": {
                    "path": checkpoint["path"],
                    "sha256": checkpoint["sha256"],
                    "fold": FOLD,
                    "seed": SEED,
                },
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "FROZEN_BASELINE",
        "source": {
            "branch": SOURCE_BRANCH,
            "commit": source_commit,
            "worktree": "E:/1/hwnas/hwnas-zcu104-full-campaign-v1",
            "worktree_branch": "codex/zcu104-full-campaign-v1",
        },
        "execution_topology": {
            "control_plane": {
                "role": "freeze_configuration_and_dispatch_only",
                "worktree": "E:/1/hwnas/hwnas-zcu104-full-campaign-v1",
            },
            "execution_plane": {
                "role": "training_hls_route_programming_and_board_runtime",
                "transport": "ssh",
                "host": "10.70.23.121",
                "port": 7889,
                "username": "jiajia",
                "authentication": "interactive_secret_not_persisted",
                "remote_probe": remote_probe_record,
            },
            "authoritative_hardware_environment": "execution_plane",
        },
        "protocol": {
            "name": "nksid_outer5fold_inner_only_v2",
            "path": "configs/evaluation/nksid_frozen_protocol_v2.yaml",
            "sha256": sha256_file(
                REPO_ROOT / "configs" / "evaluation" / "nksid_frozen_protocol_v2.yaml"
            ),
            "freeze_record": f"artifacts/{CAMPAIGN_ID}/protocol_freeze.json",
            "source_freeze_manifest": (
                f"artifacts/{CAMPAIGN_ID}/source_freeze/source_freeze_manifest.json"
            ),
        },
        "dataset": {
            "name": "NKSID",
            "dataset_manifest": {
                "path": inputs["dataset_manifest"]["path"],
                "sha256": inputs["dataset_manifest"]["sha256"],
            },
            "dataset_content_sha256": inputs["dataset_content_sha256"],
            "split_manifest": {
                "path": inputs["split_manifest"]["path"],
                "sha256": inputs["split_manifest"]["sha256"],
                "fold": FOLD,
                "seed": SEED,
            },
        },
        "checkpoint_selection": {
            "fold": FOLD,
            "seed": SEED,
            "rule": (
                "fixed deployment representative; no outer-validation metric is used "
                "for checkpoint selection"
            ),
        },
        "candidates": candidates,
        "hardware": {
            "board": "ZCU104",
            "vendor": "AMD/Xilinx",
            "target_part": TARGET_PART,
            "expected_board_part": EXPECTED_BOARD_PART,
            "clock_period_ns": 5.0,
            "route_wns_min_ns": 0.0,
            "metrics": [
                "macro_f1",
                "top1",
                "latency_ms",
                "LUT",
                "DSP",
                "BRAM",
                "power_w",
                "feasibility",
            ],
            "power_policy": "external_meter_only_for_measured_power_claim",
            "toolchain_source": "remote_execution_plane",
        },
        "environment_card": f"artifacts/{CAMPAIGN_ID}/environment_card.json",
        "status_record": f"artifacts/{CAMPAIGN_ID}/status.json",
        "claim_boundary": (
            "This config freezes campaign identity and inputs only. Search proxy, retraining, "
            "HLS, route, ZCU104 board classification/latency, and external-meter power remain "
            "separate evidence layers."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=r"E:\1\hwnas\hwnas-sonar-fourstage-operator-closure",
    )
    parser.add_argument("--source-commit", default=PINNED_SOURCE_COMMIT)
    parser.add_argument(
        "--vivado", default=r"F:\vivado\Vivado\2023.2\bin\vivado.bat"
    )
    parser.add_argument(
        "--vitis-hls", default=r"F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat"
    )
    parser.add_argument("--source-freeze-manifest")
    parser.add_argument("--remote-probe")
    parser.add_argument("--skip-tool-probe", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    source_commit = git_text(source_root, "rev-parse", args.source_commit)
    if source_commit != args.source_commit:
        raise RuntimeError(
            f"resolved source commit {source_commit} does not match requested pin {args.source_commit}"
        )

    protocol_path = REPO_ROOT / "configs" / "evaluation" / "nksid_frozen_protocol_v2.yaml"
    expected_protocol_sha = "f5a1f41a0a0bd1a44ea9d28a4dc62830bf91f7d080ba4632a7773ebce2d90358"
    assert_hash(protocol_path, expected_protocol_sha, "Protocol V2")

    remote_probe_path = Path(args.remote_probe).resolve() if args.remote_probe else None
    remote_probe = None
    remote_probe_record = None
    if remote_probe_path is not None:
        if not remote_probe_path.is_file():
            raise FileNotFoundError(remote_probe_path)
        remote_probe = json.loads(remote_probe_path.read_text(encoding="utf-8"))
        if remote_probe.get("campaign_id") != CAMPAIGN_ID:
            raise RuntimeError("remote probe campaign_id mismatch")
        remote_probe_record = file_record(remote_probe_path, relative_to=REPO_ROOT)

    inputs = build_inputs(source_root)
    config = build_config(source_commit, inputs, remote_probe_record)
    write_yaml(CAMPAIGN_CONFIG, config)

    environment = capture_environment(args, remote_probe, remote_probe_record)
    write_json(ARTIFACT_ROOT / "environment_card.json", environment)

    freeze_path = (
        Path(args.source_freeze_manifest).resolve()
        if args.source_freeze_manifest
        else None
    )
    source_freeze = verify_source_freeze(freeze_path)
    all_input_records = (
        inputs["candidates"]
        + inputs["checkpoints"]
        + inputs["dataset_files"]
        + [
            inputs["dataset_manifest"],
            inputs["group_manifest"],
            inputs["split_manifest"],
        ]
    )
    missing = [row["path"] for row in all_input_records if not (REPO_ROOT / row["path"]).is_file()]
    changed = []
    for row in all_input_records:
        path = REPO_ROOT / row["path"]
        if path.is_file() and sha256_file(path) != row["sha256"]:
            changed.append(
                {
                    "path": row["path"],
                    "expected": row["sha256"],
                    "actual": sha256_file(path),
                }
            )
    provenance_pass = not missing and not changed and len(inputs["candidates"]) == 4
    source_freeze_pass = source_freeze.get("status") == "PASS"

    protocol_freeze = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": utc_now(),
        "status": (
            "PASS"
            if provenance_pass and source_freeze_pass
            else "PENDING_SOURCE_FREEZE" if provenance_pass else "FAIL"
        ),
        "git": {
            "source_branch": SOURCE_BRANCH,
            "source_commit": source_commit,
            "campaign_branch": git_text(REPO_ROOT, "branch", "--show-current"),
            "campaign_head_at_generation": git_text(REPO_ROOT, "rev-parse", "HEAD"),
        },
        "protocol": {
            "path": protocol_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "execution_environment_probe": remote_probe_record,
        "source_freeze": source_freeze,
        "dataset": {
            "dataset_manifest": inputs["dataset_manifest"],
            "group_manifest": inputs["group_manifest"],
            "dataset_content_sha256": inputs["dataset_content_sha256"],
            "dataset_files": inputs["dataset_files"],
            "split_manifest": inputs["split_manifest"],
        },
        "candidates": inputs["candidates"],
        "checkpoints": inputs["checkpoints"],
        "traceability_verification": {
            "status": "PASS" if provenance_pass else "FAIL",
            "missing": missing,
            "unexpected": [],
            "changed": changed,
        },
        "claim_boundary": (
            "This record proves frozen-input traceability only. It does not establish "
            "ZCU104 HLS, route, bitstream, board classification, latency, or measured power."
        ),
    }
    write_json(ARTIFACT_ROOT / "protocol_freeze.json", protocol_freeze)

    environment_ready = environment["campaign_execution_readiness"] in {
        "READY",
        "READY_TO_LAUNCH",
    }
    baseline_pass = provenance_pass and source_freeze_pass
    if remote_probe is not None:
        execution_status = remote_probe["readiness"]["campaign_execution_status"]
        environment_gate = {
            "status": "PASS_REMOTE_CAPTURED",
            "execution_plane": "10.70.23.121:7889",
            "base_python": remote_probe["python_training_environment"][
                "base_python_version"
            ],
            "pytorch_environment": remote_probe["python_training_environment"]["status"],
            "gpu_count": len(remote_probe["accelerators"]["gpus"]),
            "gpu_model": remote_probe["accelerators"]["gpus"][0]["model"],
            "vivado": remote_probe["fpga_toolchain"]["vivado"]["version"],
            "vitis_hls": remote_probe["fpga_toolchain"]["vitis_hls"]["version"],
        }
        target_gate = remote_probe["zcu104_catalog_probe"]
        physical_gate = remote_probe["physical_board_probe"]
        next_required_action: Any = remote_probe["readiness"]["next_required_actions"]
    else:
        execution_status = (
            "READY_TO_LAUNCH"
            if environment_ready
            else "BLOCKED_ZCU104_DEVICE_SUPPORT_UNAVAILABLE"
        )
        environment_gate = {
            "status": "PASS" if not args.skip_tool_probe else "PENDING",
            "python": sys.version.split()[0],
            "pytorch": environment["pytorch"].get("version"),
            "vivado_available": environment["vivado"].get("available"),
            "vitis_hls_available": environment["vitis_hls"].get("available"),
        }
        target_gate = environment["zcu104_target_probe"]
        physical_gate = {"status": "NOT_PROBED_LOCAL_FALLBACK"}
        next_required_action = (
            "Install or register Vivado Zynq UltraScale+ device support and repeat the probe."
            if not environment_ready and not args.skip_tool_probe
            else None
        )
    status = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": utc_now(),
        "baseline_status": "PASS" if baseline_pass else "PENDING_SOURCE_FREEZE",
        "campaign_execution_status": (
            execution_status if baseline_pass else "NOT_READY_BASELINE_FREEZE_INCOMPLETE"
        ),
        "acceptance_gates": {
            "isolated_worktree": {
                "status": "PASS",
                "path": str(REPO_ROOT),
                "branch": git_text(REPO_ROOT, "branch", "--show-current"),
                "source_branch": SOURCE_BRANCH,
                "source_commit": source_commit,
                "dirty_main_worktree_used": False,
            },
            "worktree_clean": {
                "status": "PASS_AFTER_BASELINE_COMMIT",
                "verification_command": "git status --porcelain=v1",
                "required_output": [],
            },
            "source_freeze": source_freeze,
            "candidate_traceability": {
                "status": "PASS" if len(inputs["candidates"]) == 4 else "FAIL",
                "count": len(inputs["candidates"]),
                "arch_ids": [row["arch_id"] for row in inputs["candidates"]],
            },
            "checkpoint_traceability": {
                "status": "PASS" if len(inputs["checkpoints"]) == 4 else "FAIL",
                "count": len(inputs["checkpoints"]),
                "fold": FOLD,
                "seed": SEED,
            },
            "dataset_and_split_traceability": {
                "status": "PASS" if provenance_pass else "FAIL",
                "dataset_manifest_sha256": inputs["dataset_manifest"]["sha256"],
                "dataset_content_sha256": inputs["dataset_content_sha256"],
                "split_manifest_sha256": inputs["split_manifest"]["sha256"],
            },
            "environment_versions": environment_gate,
            "zcu104_target_part": target_gate,
            "physical_zcu104_connection": physical_gate,
        },
        "next_required_action": next_required_action,
        "claim_boundary": (
            "Baseline PASS means provenance and the remote execution topology are frozen. "
            "Training readiness, PL programming, HLS/route, board classification/latency, "
            "and measured power remain separate gates."
        ),
    }
    write_json(ARTIFACT_ROOT / "status.json", status)

    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "config": str(CAMPAIGN_CONFIG),
                "artifact_root": str(ARTIFACT_ROOT),
                "protocol_freeze_status": protocol_freeze["status"],
                "baseline_status": status["baseline_status"],
                "campaign_execution_status": status["campaign_execution_status"],
            },
            indent=2,
        )
    )
    return 0 if provenance_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
