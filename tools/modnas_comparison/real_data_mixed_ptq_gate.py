"""Source-bound real Dataset A train64 gate for all three software profiles.

Uses the unchanged one-step E0/E1 smoke weights and the existing W8A8 v2 gate's
provenance. Train64 is observed once per candidate and shared by all profiles;
no weight update, accuracy evaluation, test access, HLS, route or board action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import __version__ as pillow_version
import torch
import yaml

from tools.modnas_comparison import real_data_ptq_gate as preserved_gate
from tools.modnas_comparison.quantization_mixed import (
    ADAPTER_VERSION, SUPPORTED_PROFILES, TrainCalibration, model_state_hash,
    prepare_elastic_profiles, verify_fake_integer_parity, save_ptq_artifacts,
)
from tools.modnas_elastic.cost_graph import canonical_hash
from tools.modnas_elastic.model import ElasticNeckSupernet, StaticElasticNeckDetector, architecture_id

sha, read, save = preserved_gate.sha, preserved_gate.read, preserved_gate.save
DEFAULT_OUTPUT = ROOT / "results/modnas_vs_ours_20261007/ptq_real_dataset_a/mixed_head_v1"
UNIFORM_GATE = ROOT / "results/modnas_vs_ours_20261007/ptq_real_dataset_a/attempt_v2/real_data_ptq_gate.json"


def code_binding():
    result = preserved_gate.code_binding()
    for path in ("tools/modnas_comparison/real_data_mixed_ptq_gate.py",
                 "tools/modnas_comparison/quantization_mixed.py", "tools/fpga_h4/quant_contracts.py",
                 "reports/fpga_h2_contract/interface_contract_v1.json",
                 "reports/fpga_h2_contract/closure_manifest.json"):
        result[path] = sha(ROOT / path)
    return result


def run(output=DEFAULT_OUTPUT, *, threads=4, batch_size=1):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("new software gate requires a new empty output directory")
    torch.set_num_threads(threads)
    uniform = read(UNIFORM_GATE)
    if (uniform["status"] != "PASS_REAL_DATASET_A_SOFTWARE_PTQ_GATE"
            or uniform["formal_eligible"] is not False
            or uniform["weight_status"] != "preserved_one_step_diagnostic_not_formally_trained"):
        raise ValueError("preserved W8A8 gate does not establish the required diagnostic origin")
    sources = code_binding()
    if any(sources[path] != expected for path, expected in uniform["execution_code_sha256"].items()):
        raise ValueError("an executed live W8A8 v2 source changed before the mixed extension")
    source = Path(uniform["source_directory"])
    if any(sha(source / name) != expected for name, expected in uniform["source_artifact_sha256"].items()):
        raise ValueError("preserved checkpoint/execution artifacts changed")
    smoke, execution = read(source / "smoke_report.json"), read(source / "execution_manifest.json")
    baseline = yaml.safe_load((ROOT / "configs/baseline_sonar_frozen.yaml").read_text(encoding="utf-8"))
    frozen = read(ROOT / baseline["baseline_freeze"]["manifest"])
    entries = preserved_gate.normalize_entries(execution["dataset"]["train"]["entries"])
    if entries != preserved_gate.normalize_entries(frozen["datasets"]["train"]["entries"]):
        raise ValueError("preserved smoke and baseline have different frozen train identities")
    calibration = TrainCalibration(ROOT / baseline["dataset"]["data_dir"], entries,
                                   num_images=64, formal=True, batch_size=batch_size)
    if calibration.manifest != uniform["calibration"]:
        raise ValueError("mixed profiles must reuse the identical frozen real train64 contract")
    images = next(calibration.batches())[:1]
    supernet_state = torch.load(source / "smoke_checkpoint.pt", map_location="cpu", weights_only=True)
    exported_state = torch.load(source / "exported_candidate_smoke.pt", map_location="cpu", weights_only=True)
    if (supernet_state["optimizer_steps"] != 1 or supernet_state["formal_eligible"] is not False
            or supernet_state["execution_manifest_sha256"] != uniform["source_checkpoint_execution_manifest_sha256"]):
        raise ValueError("checkpoint training/execution origin changed")
    if exported_state["architecture"] != smoke["exports"][0]["architecture"]:
        raise ValueError("preserved independent E0 export architecture changed")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "real_data_mixed_ptq_gate.json"
    report = dict(status="RUNNING", adapter_version=ADAPTER_VERSION,
        scope="new_E0_E1_train64_three_software_profiles_one_actual_train_image_all_nodes",
        started_at_utc=datetime.now(timezone.utc).isoformat(), supported_software_profiles=list(SUPPORTED_PROFILES),
        preserved_uniform_gate_path=str(UNIFORM_GATE), preserved_uniform_gate_sha256=sha(UNIFORM_GATE),
        source_artifact_sha256=deepcopy(uniform["source_artifact_sha256"]), source_directory=str(source),
        source_checkpoint_execution_manifest_sha256=uniform["source_checkpoint_execution_manifest_sha256"],
        execution_code_sha256=sources, calibration=deepcopy(calibration.manifest),
        command_argv=[sys.executable, *sys.argv],
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                         pillow=pillow_version, device="cpu", threads=threads, batch_size=batch_size),
        weights="preserved_one_step_diagnostic_not_formally_trained", optimizer_steps_executed=0,
        accuracy_evaluated=False, quantized_accuracy_retention="NOT_EVALUATED", formal_eligible=False,
        test_image_or_label_files_accessed=False, test_evaluated=False,
        HLS_executed=False, route_executed=False, board_executed=False, hardware_feasibility="UNKNOWN",
        candidates={}, software_profile_evidence={})
    save(report_path, report)
    started = time.perf_counter()
    try:
        for label, index in (("E0", 0), ("E1", 2)):
            architecture = deepcopy(smoke["exports"][index]["architecture"])
            if label == "E0":
                model = StaticElasticNeckDetector(architecture, seed=42, num_classes=2).eval()
                model.load_state_dict(exported_state["model_state_dict"], strict=True)
                checkpoint = "exported_candidate_smoke.pt"
                bn_state = "preserved_independent_export_BN_no_new_updates"
            else:
                supernet = ElasticNeckSupernet(seed=42, num_classes=2).eval()
                supernet.load_state_dict(supernet_state["model_state_dict"], strict=True)
                model = supernet.export_subnet(architecture).eval()
                del supernet
                checkpoint = "smoke_checkpoint.pt"
                bn_state = "preserved_supernet_BN_buffers_no_new_updates"
            before = model_state_hash(model)
            if before != uniform["candidates"][label]["candidate_state_sha256"]:
                raise ValueError("mixed candidate state differs from the preserved real W8A8 gate")
            print(f"{label} REAL TRAIN64 THREE-PROFILE PREPARATION {architecture_id(architecture)}", flush=True)
            candidate_started = time.perf_counter()
            profiles = prepare_elastic_profiles(model, calibration)
            rows = {}
            for profile, prepared in profiles.items():
                witness = verify_fake_integer_parity(prepared, images)
                if model_state_hash(model) != before:
                    raise ValueError("mixed software gate modified source FP32 model or BN")
                if profile == "W8A8" and prepared.profile["integer_parameters_sha256"] != uniform["candidates"][label]["integer_parameters_sha256"]:
                    raise ValueError("uniform W8A8 parameters drifted during the mixed extension")
                folder = save_ptq_artifacts(output / label / profile, prepared, witness=witness)
                rows[profile] = dict(
                    software_parity_status=witness["status"], actual_mismatch_count=witness["mismatches"],
                    compared_nodes=len(witness["rows"]), compared_elements=sum(row["element_count"] for row in witness["rows"]),
                    numeric_contract_hash=prepared.profile["numeric_contract_hash"],
                    numeric_graph_sha256=prepared.profile["numeric_graph_sha256"],
                    integer_parameters_sha256=prepared.profile["integer_parameters_sha256"],
                    source_state_sha256=before, original_FP32_weights_and_BN_unchanged=True,
                    calibration_sha256=calibration.manifest["calibration_sha256"],
                    actual_train_calibration_images=64, integer_witness_train_images=1,
                    artifact_sha256={path.name: sha(path) for path in folder.iterdir() if path.is_file()},
                    software_only=True, quantized_accuracy_retention="NOT_EVALUATED", formal_eligible=False,
                    HLS_parity="NOT_RUN", board_parity="NOT_RUN", hardware_feasibility="UNKNOWN")
                print(f"{label} {profile} {witness['status']} mismatches={witness['mismatches']} nodes={len(witness['rows'])}", flush=True)
            report["candidates"][label] = dict(architecture=architecture, architecture_id=architecture_id(architecture),
                architecture_sha256=canonical_hash(architecture), source_checkpoint=checkpoint,
                source_checkpoint_sha256=uniform["source_artifact_sha256"][checkpoint], source_BN_state=bn_state,
                source_state_sha256=before, profiles=rows, wall_seconds=time.perf_counter()-candidate_started)
            save(report_path, report)
            del profiles, model
            gc.collect()
        if code_binding() != sources or any(sha(source / name) != expected for name, expected in report["source_artifact_sha256"].items()):
            raise ValueError("source/checkpoint drift during real mixed gate")
        for profile in SUPPORTED_PROFILES:
            counts = [candidate["profiles"][profile]["actual_mismatch_count"] for candidate in report["candidates"].values()]
            report["software_profile_evidence"][profile] = dict(
                implementation_status="IMPLEMENTED", real_DatasetA_E0_E1_all_node_parity="PASS" if counts == [0,0] else "FAIL",
                total_mismatches=sum(counts), hardware_status="UNKNOWN", accuracy_retention="NOT_EVALUATED")
        report["status"] = ("PASS_REAL_DATASET_A_THREE_PROFILE_SOFTWARE_PTQ_GATE" if
            all(row["total_mismatches"] == 0 for row in report["software_profile_evidence"].values())
            else "FAIL_REAL_DATASET_A_THREE_PROFILE_SOFTWARE_PTQ_GATE")
    except Exception as error:
        report.update(status="FAILED_THREE_PROFILE_SOFTWARE_GATE_EXECUTION", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
        save(report_path, report)
    snapshot = output / "executed_source"
    snapshot.mkdir()
    for path, expected in sources.items():
        if sha(ROOT / path) != expected:
            raise ValueError("executed source drift before archive")
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    save(output / "executed_source_manifest.json", dict(gate_sha256=sha(report_path), executed_sources_sha256=sources))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    result = run(args.output, threads=args.threads, batch_size=args.batch_size)
    print(json.dumps(dict(status=result["status"], report=str(args.output / "real_data_mixed_ptq_gate.json")), indent=2))
    if result["status"] != "PASS_REAL_DATASET_A_THREE_PROFILE_SOFTWARE_PTQ_GATE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
