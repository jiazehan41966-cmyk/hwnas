#!/usr/bin/env python3
"""Freeze the four-stage K5 full-network deployment candidate queue.

This script does not run new search, training, HLS, routing, bitstream
generation, COM5 measurement, or power measurement. It converts the existing
Protocol V2 formal results into a small, machine-checkable deployment queue and
archives the source-snapshot ZIP provenance as path+SHA256 indexes so those
archives can remain local or live in a release/archive area instead of being
tracked as mainline Git payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts" / "sonar_fourstage_operator_v2"
RESULT_ROOT = ROOT / "results" / "sonar_fourstage_operator_v2"
BASE8_RESULTS = RESULT_ROOT / "base8_formal"
EXTENDED_RESULTS = RESULT_ROOT / "extended_formal"

BASELINE_ID = "fourstage_s2_k3_e3_s4_mbconv_k3_e3"
STAGE2_K5_ALLOWED_STAGE4 = ("s4_mbconv_k3_e3", "s4_skip")
DEPLOYMENT_OUTPUT = ARTIFACT_ROOT / "fourstage_deployment_candidate_selection.json"
SOURCE_SNAPSHOT_OUTPUT = ARTIFACT_ROOT / "source_snapshot_archive_index.json"
CHECKPOINT_EXPORT_SUMMARY = ARTIFACT_ROOT / "fourstage_checkpoint_export_summary.json"
INT8_REFERENCE_SUMMARY = ARTIFACT_ROOT / "fourstage_int8_reference_summary.json"
CSIM_ZERO_MISMATCH_SUMMARY = ARTIFACT_ROOT / "fourstage_csim_zero_mismatch_summary.json"
HLS_SYNTHESIS_SUMMARY = ARTIFACT_ROOT / "fourstage_hls_synthesis_summary.json"
EXTERNAL_SCRATCH_HLS_SUMMARY = (
    ARTIFACT_ROOT / "fourstage_external_scratch_hls_summary.json"
)
EXTERNAL_SCRATCH_RTL_COSIM_SUMMARY = (
    ARTIFACT_ROOT / "fourstage_external_scratch_rtl_cosim_summary.json"
)

SOURCE_FREEZE_MANIFESTS = [
    {
        "role": "protocol_v2_selection_source",
        "relative_manifest": "source_freeze/source_freeze_manifest.json",
        "claim_scope": "Protocol V2 inner-only selection and freeze binding source",
    },
    {
        "role": "dir_v1_source",
        "relative_manifest": "dir_v1_source_freeze/source_freeze_manifest.json",
        "claim_scope": "Dir-v1 implementation and pretraining gates source",
    },
    {
        "role": "k5_formal_source",
        "relative_manifest": "k5_formal_source_freeze/source_freeze_manifest.json",
        "claim_scope": "Stage4-K5 formal accuracy and exact-shape micro-gate source",
    },
    {
        "role": "experiment_closure_source",
        "relative_manifest": "closure_source_freeze/source_freeze_manifest.json",
        "claim_scope": "Formal four-stage operator closure source before PR housekeeping",
    },
    {
        "role": "slim_pr_current_source",
        "relative_manifest": "slim_pr_source_freeze/source_freeze_manifest.json",
        "claim_scope": "Current slim PR code, protocol ledger, and deployment queue source",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def maybe_evidence(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return evidence(path)


def resolve_recorded_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def is_tracked(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative).replace("\\", "/")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def verify_source_freeze(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"status": "MISSING", "manifest": str(manifest_path.resolve())}
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_experiment_source.py"),
            "verify",
            "--manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "PARSE_FAIL",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    payload["returncode"] = completed.returncode
    return payload


def build_source_snapshot_index() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for spec in SOURCE_FREEZE_MANIFESTS:
        manifest_path = ARTIFACT_ROOT / spec["relative_manifest"]
        entry: dict[str, Any] = {
            "role": spec["role"],
            "claim_scope": spec["claim_scope"],
            "manifest": maybe_evidence(manifest_path),
        }
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            archive = manifest.get("archive") or {}
            archive_path = resolve_recorded_path(archive.get("path"))
            entry["recorded_git_head"] = (manifest.get("git") or {}).get("head")
            entry["recorded_git_status_porcelain"] = (
                (manifest.get("git") or {}).get("status_porcelain") or []
            )
            if spec["role"] == "slim_pr_current_source":
                entry["verification"] = verify_source_freeze(manifest_path)
            else:
                entry["current_verification"] = (
                    "NOT_RECHECKED_AFTER_SLIM_PR_HOUSEKEEPING"
                )
                entry["reason"] = (
                    "This manifest records the source state at its original "
                    "experiment gate. The current PR source is verified by "
                    "slim_pr_current_source."
                )
            if archive_path is not None and archive_path.is_file():
                entry["archive"] = {
                    "path": str(archive_path.resolve()),
                    "recorded_sha256": archive.get("sha256"),
                    "actual_sha256": sha256_file(archive_path),
                    "recorded_bytes": archive.get("bytes"),
                    "actual_bytes": archive_path.stat().st_size,
                    "git_tracked": is_tracked(archive_path),
                    "mainline_policy": (
                        "keep_local_or_release_archive_only; mainline tracks "
                        "manifest path, bytes, and SHA256, not ZIP bytes"
                    ),
                }
            else:
                entry["archive"] = {
                    "path": "" if archive_path is None else str(archive_path),
                    "status": "MISSING_LOCAL_ARCHIVE",
                    "mainline_policy": (
                        "manifest remains the evidence index; ZIP bytes are "
                        "not required in mainline Git"
                    ),
                }
        else:
            entry["archive"] = {"status": "MISSING_MANIFEST"}
        entries.append(entry)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "snapshot_policy": {
            "tracked_in_mainline": "source_freeze_manifest.json and this hash index",
            "not_tracked_in_mainline": "source_snapshot.zip binary archives",
            "rationale": (
                "Keep the PR reviewable while preserving byte-addressable "
                "provenance for local or release/archive storage."
            ),
        },
        "entries": entries,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def full_network_gate_sequence() -> list[dict[str, Any]]:
    checkpoint = maybe_evidence(CHECKPOINT_EXPORT_SUMMARY)
    int8 = maybe_evidence(INT8_REFERENCE_SUMMARY)
    csim = maybe_evidence(CSIM_ZERO_MISMATCH_SUMMARY)
    static_hls = maybe_evidence(HLS_SYNTHESIS_SUMMARY)
    external_hls = maybe_evidence(EXTERNAL_SCRATCH_HLS_SUMMARY)
    rtl = maybe_evidence(EXTERNAL_SCRATCH_RTL_COSIM_SUMMARY)
    hls = external_hls or static_hls
    checkpoint_status = "PENDING"
    int8_status = "PENDING"
    reference_status = "PENDING"
    csim_status = "PENDING"
    zero_mismatch_status = "PENDING"
    hls_status = "PENDING"
    rtl_status = "PENDING"
    route_status = "PENDING"
    if checkpoint is not None:
        checkpoint_payload = read_json(CHECKPOINT_EXPORT_SUMMARY)
        if checkpoint_payload.get("status") == "PASS":
            checkpoint_status = "PASS"
    if int8 is not None:
        int8_payload = read_json(INT8_REFERENCE_SUMMARY)
        if int8_payload.get("status") == "PASS":
            int8_status = "PASS"
            reference_status = "PASS"
    if csim is not None:
        csim_payload = read_json(CSIM_ZERO_MISMATCH_SUMMARY)
        if (
            csim_payload.get("status") == "PASS"
            and csim_payload.get("formal_scope") is True
            and csim_payload.get("zero_mismatch") is True
        ):
            csim_status = "PASS"
            zero_mismatch_status = "PASS"
    if external_hls is not None:
        hls_payload = read_json(EXTERNAL_SCRATCH_HLS_SUMMARY)
        hls_status = str(hls_payload.get("status") or "PENDING")
        if hls_status == "PASS":
            rtl_status = "PENDING"
            route_status = "NOT_RUN_RTL_COSIM_NOT_PASSED"
        else:
            rtl_status = "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
            route_status = "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
    elif static_hls is not None:
        hls_payload = read_json(HLS_SYNTHESIS_SUMMARY)
        hls_status = str(hls_payload.get("status") or "PENDING")
        if hls_status != "PASS":
            rtl_status = "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
            route_status = "NOT_RUN_HLS_SYNTHESIS_NOT_PASSED"
    if rtl is not None:
        rtl_payload = read_json(EXTERNAL_SCRATCH_RTL_COSIM_SUMMARY)
        rtl_status = str(rtl_payload.get("status") or "PENDING")
        route_status = (
            "PENDING"
            if rtl_status == "PASS"
            else "NOT_RUN_RTL_COSIM_NOT_PASSED"
        )
    return [
        {
            "gate": "real_checkpoint_export",
            "status": checkpoint_status,
            "required_evidence": "checkpoint path and SHA256 from selected candidate index",
            "evidence": checkpoint,
        },
        {
            "gate": "int8_calibration",
            "status": int8_status,
            "required_evidence": "real activation calibration set, quantization scales, zero-points, hashes",
            "evidence": int8,
        },
        {
            "gate": "python_integer_reference",
            "status": reference_status,
            "required_evidence": "bit-exact full-network INT8 reference outputs",
            "evidence": int8,
        },
        {
            "gate": "full_network_c_sim",
            "status": csim_status,
            "required_evidence": "complete network C simulation transcript and output tensors",
            "evidence": csim,
        },
        {
            "gate": "pytorch_int8_vs_csim_zero_mismatch",
            "status": zero_mismatch_status,
            "required_evidence": "zero mismatch report on integer outputs",
            "evidence": csim,
        },
        {
            "gate": "hls_synthesis",
            "status": hls_status,
            "required_evidence": "complete-network HLS reports",
            "evidence": hls,
            "static_full_buffer_probe": static_hls,
        },
        {
            "gate": "rtl_cosim",
            "status": rtl_status,
            "required_evidence": "complete-network RTL co-simulation transcript and output tensors",
            "evidence": rtl,
        },
        {
            "gate": "place_and_route_5ns",
            "status": route_status,
            "required_evidence": "AV7K325 5ns implementation reports",
            "acceptance": {"wns_ns": ">=0", "route_dsp": "<=700"},
        },
        {
            "gate": "bitstream",
            "status": "NOT_GENERATED",
            "required_evidence": "bitstream path and SHA256",
        },
        {
            "gate": "com5_board_latency",
            "status": "NOT_RUN",
            "required_evidence": "COM5 transaction logs and latency JSON",
        },
        {
            "gate": "external_meter_power",
            "status": "NOT_MEASURED",
            "required_evidence": "external instrument CSV and acquisition metadata",
        },
    ]


def load_result_summary(arch_id: str) -> tuple[Path, dict[str, Any]]:
    for root in (BASE8_RESULTS, EXTENDED_RESULTS):
        path = root / arch_id / "protocol_summary.json"
        if path.is_file():
            return path, read_json(path)
    raise FileNotFoundError(f"Missing Protocol V2 summary for {arch_id}")


def index_files(paths: list[Path]) -> list[dict[str, Any]]:
    return [evidence(path) for path in sorted(paths)]


def summarize_candidate(
    role: str,
    arch_id: str,
    role_reason: str,
    candidate_manifest_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary_path, summary = load_result_summary(arch_id)
    result_dir = summary_path.parent
    candidate_row = candidate_manifest_rows[arch_id]
    candidate_path = resolve_recorded_path(candidate_row["path"])
    if candidate_path is None or not candidate_path.is_file():
        raise FileNotFoundError(f"Missing candidate JSON for {arch_id}")
    checkpoints = list(result_dir.glob("best_fold*_seed*.pt"))
    predictions = list(result_dir.glob("outer_predictions_fold*_seed*.jsonl"))
    run_jsons = list(result_dir.glob("run_fold*_seed*.json"))
    source_freeze = (summary.get("provenance") or {}).get("source_freeze") or {}
    return {
        "role": role,
        "arch_id": arch_id,
        "role_reason": role_reason,
        "factors": candidate_row.get("factors") or {},
        "architecture_sha256": candidate_row.get("architecture_sha256"),
        "parameter_count": candidate_row.get("parameter_count"),
        "macs": candidate_row.get("macs"),
        "candidate_json": evidence(candidate_path),
        "protocol_summary": evidence(summary_path),
        "result_dir": str(result_dir.resolve()),
        "formal_protocol_units": {
            "expected": 15,
            "observed_run_json_count": len(run_jsons),
            "checkpoint_count": len(checkpoints),
            "outer_prediction_count": len(predictions),
        },
        "metrics": {
            "outer_macro_f1": summary.get("outer_macro_f1"),
            "outer_top1": summary.get("outer_top1"),
            "outer_weighted_f1": summary.get("outer_weighted_f1"),
            "per_class_f1_mean": summary.get("per_class_f1_mean"),
        },
        "checkpoint_index": index_files(checkpoints),
        "outer_prediction_index": index_files(predictions),
        "source_freeze": {
            "manifest_path": source_freeze.get("path"),
            "manifest_sha256": source_freeze.get("manifest_sha256"),
            "verification_status": source_freeze.get("verification_status"),
            "archive_path": source_freeze.get("archive_path"),
            "archive_sha256": source_freeze.get("archive_sha256"),
            "archive_bytes": source_freeze.get("archive_bytes"),
        },
        "deployment_state": "PENDING_FULL_NETWORK_HARDWARE_CLOSURE",
    }


def load_candidate_manifest_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for relative in (
        "base8_candidates/base8_manifest.json",
        "extended_candidates/extended_manifest.json",
    ):
        manifest = read_json(ARTIFACT_ROOT / relative)
        for row in manifest.get("rows", []):
            rows[row["arch_id"]] = row
    return rows


def select_candidate_ids() -> dict[str, tuple[str, str]]:
    base8 = read_json(ARTIFACT_ROOT / "base8_formal_analysis.json")
    stage4_k5 = read_json(ARTIFACT_ROOT / "stage4_k5_formal_analysis.json")
    base8_summaries = base8["architecture_summaries"]

    def mean_macro(arch_id: str) -> float:
        return float(base8_summaries[arch_id]["macro_f1"]["mean"])

    stage2_pool = [
        arch_id
        for arch_id in base8_summaries
        if "s2_k5" in arch_id
        and any(marker in arch_id for marker in STAGE2_K5_ALLOWED_STAGE4)
    ]
    stage2_k5 = max(stage2_pool, key=mean_macro)

    skip_pool = [
        arch_id for arch_id in base8_summaries if arch_id.endswith("s4_skip")
    ]
    low_cost = max(skip_pool, key=mean_macro)

    passed_stage4_k5 = [
        row["k5_architecture"]
        for row in stage4_k5["comparisons"].values()
        if row["passes_stage4_k5_accuracy_gate"]
    ]

    def extended_mean_macro(arch_id: str) -> float:
        _path, summary = load_result_summary(arch_id)
        return float(summary["outer_macro_f1"]["mean"])

    stage4_k5_rep = max(passed_stage4_k5, key=extended_mean_macro)

    return {
        "original_baseline": (
            BASELINE_ID,
            "Fixed original mature baseline: Stage2 K3-e3 and Stage4 K3-e3.",
        ),
        "stage2_k5_representative": (
            stage2_k5,
            (
                "Best existing formal Stage2-K5 candidate among the pre-K5 "
                "Stage4 backgrounds K3-e3 and Skip; no new search."
            ),
        ),
        "stage4_k5_representative": (
            stage4_k5_rep,
            (
                "Best existing formal Stage4-K5 candidate among the "
                "preregistered Stage2 backgrounds that passed the Stage4-K5 "
                "accuracy gate."
            ),
        ),
        "low_cost_skip_representative": (
            low_cost,
            "Best existing formal Skip candidate by Protocol V2 mean macro_f1.",
        ),
    }


def build_deployment_selection(
    source_snapshot_index: dict[str, Any],
) -> dict[str, Any]:
    base8 = read_json(ARTIFACT_ROOT / "base8_formal_analysis.json")
    stage4_k5 = read_json(ARTIFACT_ROOT / "stage4_k5_formal_analysis.json")
    dir_gate = read_json(ARTIFACT_ROOT / "dir_accuracy_gate.json")
    direction_gate = read_json(ARTIFACT_ROOT / "direction_gate_summary.json")
    candidate_rows = load_candidate_manifest_rows()
    selected_ids = select_candidate_ids()
    selected = [
        summarize_candidate(role, arch_id, reason, candidate_rows)
        for role, (arch_id, reason) in selected_ids.items()
    ]
    selected_arch_ids = [row["arch_id"] for row in selected]
    mature_stage4_accuracy_space = {
        "status": "ACCURACY_LEVEL_SUPPORTED_PENDING_FULL_NETWORK_HARDWARE",
        "stage2_candidates": [
            "MBConv-k3-e3",
            "MBConv-k3-e6",
            "MBConv-k5-e3",
            "MBConv-k5-e6",
        ],
        "stage4_candidates": [
            "Skip",
            "MBConv-k3-e3",
            "MBConv-k5-e3",
        ],
        "structure_count": 12,
        "boundary": (
            "This is not the final deployable space until complete-network "
            "INT8/HLS/route/bitstream/COM5 gates pass for Stage4-K5."
        ),
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "GENERAL_OP_SELECTED",
        "stage": "FULL_NETWORK_DEPLOYMENT_CLOSURE_QUEUE_FROZEN",
        "selection_protocol": (
            "fixed_rule_from_existing_ProtocolV2_formal_results_no_new_search"
        ),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1").splitlines(),
        },
        "frozen_experiment_conclusion": {
            "evaluated_structure_count": 16,
            "formal_protocol_unit_count": (
                int(base8["formal_unit_count"])
                + int(stage4_k5["formal_k5_units"])
                + int(dir_gate["formal_dir_units"])
            ),
            "stage2_k5_state": "READY_FORMAL_ACCURACY_SUPPORTED",
            "stage4_k5_state": "READY_ACCURACY_SUPPORTED",
            "dir_mbconv3_split11_e3_v1": "NOT_ADMITTED_ACCURACY_GATE_FAILED",
            "direction_basis": direction_gate["status"],
            "current_stage": "GENERAL_OP_SELECTED",
            "claim": (
                "Mature MBConv-k5 is selected at the algorithm/accuracy layer. "
                "Dir-v1 remains a negative result: the sonar directional basis "
                "diagnostic passed, but this directional operator design did not "
                "meet the preregistered accuracy admission gate."
            ),
        },
        "stopped_work": {
            "dir_v1_downstream": {
                "robustness": "NOT_RUN_ACCURACY_GATE_FAILED",
                "real_checkpoint_int8": "NOT_RUN_ACCURACY_GATE_FAILED",
                "rtl_cosim": "NOT_RUN_ACCURACY_GATE_FAILED",
                "full_network_route": "NOT_RUN_ACCURACY_GATE_FAILED",
                "bitstream": "NOT_GENERATED_ACCURACY_GATE_FAILED",
                "power": "NOT_MEASURED",
            }
        },
        "selected_candidate_count": len(selected),
        "selected_arch_ids": selected_arch_ids,
        "selected_candidates": selected,
        "full_network_gate_sequence": full_network_gate_sequence(),
        "mature_accuracy_space_pending_hardware": mature_stage4_accuracy_space,
        "final_deployable_space_rule": {
            "if_stage4_k5_complete_network_passes": {
                "status": "DEPLOYABLE_SPACE_12",
                "stage2_x_stage4": "4x3",
                "structure_count": 12,
                "stage4_candidates": ["K3-e3", "K5-e3", "Skip"],
            },
            "if_stage4_k5_complete_network_fails": {
                "status": "DEPLOYABLE_SPACE_8",
                "stage2_x_stage4": "4x2",
                "structure_count": 8,
                "stage4_candidates": ["K3-e3", "Skip"],
            },
        },
        "hardware_claim_boundary": {
            "stage4_k5_exact_shape_micro_harness_route": (
                "PASS, operator-slot evidence only"
            ),
            "rtl_cosim": (
                read_json(EXTERNAL_SCRATCH_RTL_COSIM_SUMMARY).get("status")
                if EXTERNAL_SCRATCH_RTL_COSIM_SUMMARY.is_file()
                else "NOT_RUN"
            ),
            "complete_network_route": "NOT_RUN",
            "bitstream": "NOT_GENERATED",
            "com5": "NOT_RUN",
            "power": "NOT_MEASURED",
            "strict_lut_proxy": "NOT_A_ROUTE_RESULT",
        },
        "source_snapshot_archive_index": {
            "path": str(SOURCE_SNAPSHOT_OUTPUT.resolve()),
            "sha256": source_snapshot_index["payload_sha256"],
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    source_snapshot_index = build_source_snapshot_index()
    write_json(SOURCE_SNAPSHOT_OUTPUT, source_snapshot_index)
    deployment_selection = build_deployment_selection(source_snapshot_index)
    write_json(DEPLOYMENT_OUTPUT, deployment_selection)
    print(json.dumps(deployment_selection, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
