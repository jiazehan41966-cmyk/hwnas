"""Run a real Dataset A train64 software integer gate using preserved smoke weights.

No training, optimizer update, accuracy evaluation, test access, HLS, route, or
board command occurs here. E0 loads the already BN-calibrated exported smoke
candidate. E1 exports the report's third architecture from the stored supernet,
using its stored BN buffers. Both weights remain one-step diagnostics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import __version__ as pillow_version
import torch
import yaml

from tools.modnas_elastic.model import ElasticNeckSupernet, StaticElasticNeckDetector, architecture_id
from tools.modnas_elastic.cost_graph import canonical_hash
from tools.modnas_comparison.quantization import (
    ADAPTER_VERSION, TrainCalibration, model_state_hash, prepare_elastic_ptq,
    verify_fake_integer_parity, save_ptq_artifacts,
)

DEFAULT_SOURCE = ROOT / "results/modnas_elastic_adapter_20261007/local_smoke_v1"
PRIOR_FAILED_OUTPUT = ROOT / "results/modnas_vs_ours_20261007/ptq_real_dataset_a"
DEFAULT_OUTPUT = PRIOR_FAILED_OUTPUT / "attempt_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalize_entries(entries):
    return [dict(image=entry["image"].replace("\\", "/"),
                 image_sha256=entry["image_sha256"],
                 annotation=entry["annotation"].replace("\\", "/"),
                 annotation_sha256=entry["annotation_sha256"]) for entry in entries]


def code_binding():
    # Bind the execution-relevant new graph, calibration, numeric, folding and
    # static-model modules; the old smoke's entire closure remains archived.
    paths = [
        "tools/modnas_comparison/real_data_ptq_gate.py", "tools/modnas_comparison/quantization.py",
        "tools/modnas_elastic/model.py", "tools/modnas_elastic/cost_graph.py",
        "tools/detector_elastic_pilot/model.py", "tools/fpga_h2/numeric.py",
        "src/hwnas_fpga/models/builder.py", "src/hwnas_fpga/models/detection.py",
        "src/hwnas_fpga/models/sonar_candidate/__init__.py",
        "src/hwnas_fpga/models/sonar_candidate_v0_2/__init__.py",
        "src/hwnas_fpga/deploy/h2_ptq.py", "src/hwnas_fpga/deploy/h2_contract.py",
        "src/hwnas_fpga/deploy/ptq_eval.py", "reports/fpga_h2_contract/numeric_contract_v1.json",
        "configs/search_space_sonar_v0_1.yaml", "configs/detector_elastic_pilot_model.yaml",
        "configs/baseline_sonar_frozen.yaml", "reports/baseline_sonar_frozen/freeze_manifest.json",
    ]
    return {path: sha(ROOT / path) for path in paths}


def run(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, *, threads=4, batch_size=1):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("real-data PTQ gate requires a new empty output directory")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    torch.set_num_threads(threads)
    original_inputs = {name: sha(source / name) for name in (
        "smoke_report.json", "execution_manifest.json", "smoke_checkpoint.pt", "exported_candidate_smoke.pt")}
    smoke, execution = read(source / "smoke_report.json"), read(source / "execution_manifest.json")
    if (smoke.get("optimizer_steps") != 1 or smoke.get("formal_eligible") is not False
            or smoke.get("test_evaluated") is not False or smoke.get("seed") != 42):
        raise ValueError("source must be the retained one-step seed42 diagnostic smoke")
    for name, expected in smoke["artifacts"].items():
        if name in original_inputs and original_inputs[name] != expected:
            raise ValueError("retained smoke artifact hash mismatch: " + name)
    baseline_path = ROOT / "configs/baseline_sonar_frozen.yaml"
    if sha(baseline_path) != execution["source_sha256"]["configs/baseline_sonar_frozen.yaml"]:
        raise ValueError("baseline differs from the retained smoke execution")
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    freeze_path = ROOT / baseline["baseline_freeze"]["manifest"]
    if sha(freeze_path) != execution["source_sha256"][freeze_path.relative_to(ROOT).as_posix()]:
        raise ValueError("frozen data manifest differs from the retained smoke")
    frozen = read(freeze_path)
    train_entries = normalize_entries(execution["dataset"]["train"]["entries"])
    if train_entries != normalize_entries(frozen["datasets"]["train"]["entries"]):
        raise ValueError("retained smoke train identities differ from the frozen Dataset A")
    sources = code_binding()
    for path in ("tools/modnas_elastic/model.py", "tools/detector_elastic_pilot/model.py",
                 "src/hwnas_fpga/models/builder.py", "src/hwnas_fpga/models/detection.py",
                 "src/hwnas_fpga/models/sonar_candidate/__init__.py",
                 "src/hwnas_fpga/models/sonar_candidate_v0_2/__init__.py"):
        if sources[path] != execution["source_sha256"][path]:
            raise ValueError("static model code differs from the retained checkpoint origin: " + path)
    calibration = TrainCalibration(ROOT / baseline["dataset"]["data_dir"], train_entries,
                                   batch_size=batch_size, num_images=64, formal=True)
    images = next(calibration.batches())[:1]
    supernet_state = torch.load(source / "smoke_checkpoint.pt", map_location="cpu", weights_only=True)
    exported_state = torch.load(source / "exported_candidate_smoke.pt", map_location="cpu", weights_only=True)
    if (supernet_state.get("execution_manifest_sha256") != original_inputs["execution_manifest.json"]
            or supernet_state.get("optimizer_steps") != 1 or supernet_state.get("formal_eligible") is not False):
        raise ValueError("stored supernet checkpoint origin differs")
    exports = smoke["exports"]
    if exports[0]["architecture"]["pan"] or not exports[2]["architecture"]["pan"]:
        raise ValueError("expected E0 export0 and E1 export2 in the preserved smoke report")
    if exported_state["architecture"] != exports[0]["architecture"]:
        raise ValueError("calibrated E0 export architecture differs from the smoke report")
    report = dict(
        scope="real_DatasetA_train64_software_PTQ_and_one_real_train_image_all_node_parity",
        adapter_version=ADAPTER_VERSION,
        status="RUNNING", started_at_utc=datetime.now(timezone.utc).isoformat(),
        source_directory=str(source), source_artifact_sha256=original_inputs,
        source_checkpoint_execution_manifest_sha256=supernet_state["execution_manifest_sha256"],
        execution_code_sha256=sources, calibration=deepcopy(calibration.manifest),
        command_argv=[sys.executable, *sys.argv], environment=dict(
            python=sys.version, torch=torch.__version__, numpy=np.__version__,
            pillow=pillow_version, device="cpu", threads=threads, batch_size=batch_size),
        weight_status="preserved_one_step_diagnostic_not_formally_trained",
        optimizer_steps_executed=0, trained_epochs=0, accuracy_evaluated=False,
        quantized_accuracy_retention="NOT_EVALUATED", formal_eligible=False,
        test_image_or_label_files_accessed=False, test_evaluated=False,
        HLS_executed=False, route_executed=False, board_executed=False,
        hardware_feasibility="UNKNOWN", candidates={},
    )
    prior_path = PRIOR_FAILED_OUTPUT / "real_data_ptq_gate.json"
    if prior_path.is_file():
        prior = read(prior_path)
        if prior.get("status") != "FAILED_REAL_DATASET_A_SOFTWARE_PTQ_EXECUTION":
            raise ValueError("registered prior attempt is not the retained execution failure")
        archives = {}
        for path, archived in (("tools/modnas_comparison/quantization.py", "quantization_v1.py"),
                               ("tools/modnas_comparison/real_data_ptq_gate.py", "real_data_ptq_gate_v1.py")):
            archive_path = PRIOR_FAILED_OUTPUT / "failed_attempt_source" / archived
            if sha(archive_path) != prior["execution_code_sha256"][path]:
                raise ValueError("retained failed-attempt source does not match its executed hash")
            archives[str(archive_path)] = sha(archive_path)
        report["retained_prior_failure"] = dict(
            path=str(prior_path), sha256=sha(prior_path), error=prior["error"],
            archived_executed_source_sha256=archives,
            registered_change="paired_FP64_mathematical_BN_fold_reference_from_unchanged_FP32_checkpoint",
            original_FP32_atol_and_rtol_remain=1e-5,
            original_FP32_tolerance_relaxed=False,
        )
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "real_data_ptq_gate.json"
    save(report_path, report)
    start = time.perf_counter()
    try:
        for label, export_index in (("E0", 0), ("E1", 2)):
            architecture = deepcopy(exports[export_index]["architecture"])
            if label == "E0":
                model = StaticElasticNeckDetector(architecture, seed=42, num_classes=2).eval()
                model.load_state_dict(exported_state["model_state_dict"], strict=True)
                origin = dict(checkpoint="exported_candidate_smoke.pt", export_index=0,
                    BN_state="retained_export_after_one_train_image_BN_calibration_no_new_BN_updates")
            else:
                supernet = ElasticNeckSupernet(seed=42, num_classes=2).eval()
                supernet.load_state_dict(supernet_state["model_state_dict"], strict=True)
                model = supernet.export_subnet(architecture).eval()
                origin = dict(checkpoint="smoke_checkpoint.pt", export_index=2,
                    BN_state="retained_supernet_buffers_no_new_BN_calibration_or_updates")
                del supernet
            print(label + " REAL TRAIN64 PTQ START " + architecture_id(architecture), flush=True)
            candidate_start = time.perf_counter()
            weight_hash = model_state_hash(model)
            prepared = prepare_elastic_ptq(model, calibration, device="cpu", profile="W8A8")
            witness = verify_fake_integer_parity(prepared, images)
            if model_state_hash(model) != weight_hash:
                raise ValueError("real-data software gate changed original candidate state")
            folder = save_ptq_artifacts(output / label, prepared, witness=witness)
            record = dict(
                architecture=architecture, architecture_id=architecture_id(architecture),
                architecture_sha256=canonical_hash(architecture), source=origin,
                source_checkpoint_sha256=original_inputs[origin["checkpoint"]],
                candidate_state_sha256=weight_hash, original_weights_and_BN_unchanged=True,
                calibrated_real_train_images=64, integer_parity_real_train_images=1,
                float_fold_witness=prepared.profile["bn_folding"]["witness"],
                software_parity_status=witness["status"], actual_mismatch_count=witness["mismatches"],
                compared_nodes=len(witness["rows"]), compared_elements=sum(row["element_count"] for row in witness["rows"]),
                calibration_sha256=calibration.manifest["calibration_sha256"],
                numeric_graph_sha256=prepared.profile["numeric_graph_sha256"],
                integer_parameters_sha256=prepared.profile["integer_parameters_sha256"],
                artifact_sha256={path.name: sha(path) for path in folder.iterdir() if path.is_file()},
                wall_seconds=time.perf_counter() - candidate_start,
                formal_eligible=False, accuracy_retention="NOT_EVALUATED", HLS_parity="NOT_RUN", board_parity="NOT_RUN",
            )
            report["candidates"][label] = record
            save(report_path, report)
            print(f"{label} SOFTWARE WITNESS {witness['status']} mismatches={witness['mismatches']} "
                  f"nodes={record['compared_nodes']} seconds={record['wall_seconds']:.3f}", flush=True)
            del prepared, model
            gc.collect()
        if any(sha(source / name) != expected for name, expected in original_inputs.items()):
            raise ValueError("preserved checkpoint/execution artifacts changed during software gate")
        if code_binding() != sources:
            raise ValueError("execution source changed during software gate")
        report["status"] = ("PASS_REAL_DATASET_A_SOFTWARE_PTQ_GATE" if
            all(record["actual_mismatch_count"] == 0 for record in report["candidates"].values())
            else "FAIL_REAL_DATASET_A_SOFTWARE_PTQ_GATE")
    except Exception as error:
        report.update(status="FAILED_REAL_DATASET_A_SOFTWARE_PTQ_EXECUTION",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(wall_seconds=time.perf_counter() - start,
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-smoke", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    result = run(args.source_smoke, args.output, threads=args.threads, batch_size=args.batch_size)
    print(json.dumps(dict(status=result["status"], report=str(args.output / "real_data_ptq_gate.json")), indent=2))
    if result["status"] != "PASS_REAL_DATASET_A_SOFTWARE_PTQ_GATE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
