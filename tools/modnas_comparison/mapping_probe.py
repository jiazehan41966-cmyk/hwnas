"""Prepare isolated fixed-B1 stem PE8/16/32 witnesses; no HLS launch.

Only OUT_TILE changes in a retained source-verified H3 packed convolution.
Input SIMD1, W8A8/ACC32 numeric arithmetic, 8-bit input and 256-bit output,
640x640 input, source witness and 5ns/ZCU104/2022.2 platform remain bound.
Local NumPy integer replay certifies the retained reference; the changed C++
still requires its own actual C-sim before any CSYNTH result is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_elastic.cost_graph import DEFAULT_PRECISION, build_cost_graph, canonical_hash
from tools.modnas_elastic.hardware import PrimitiveEvidenceCache, PART

PARALLELISM = (8, 16, 32)
RESOURCE_BUDGET = dict(LUT=161280, DSP=1209, BRAM18=436)
WITNESS_FILES = ("weights.bin", "biases.bin", "layers.bin", "input.bin", "reference.bin")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_module():
    spec = importlib.util.spec_from_file_location("modnas_probe_h2_numeric", ROOT / "tools/fpga_h2/numeric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline() -> dict[str, Any]:
    graph = build_cost_graph(dict(pan=False, td_kernels=[3, 3, 3], pan_kernels=[]))
    stem = graph["nodes"][0]
    if stem["id"] != "backbone.stem.conv.conv" or stem["output_shape"] != [32, 320, 320]:
        raise ValueError("Fixed B1 stem contract changed")
    database = PrimitiveEvidenceCache()
    evidence = database.query(stem)
    if evidence["status"] != "EXACT_PRIMITIVE_HLS_EVIDENCE":
        raise ValueError("Cannot derive a mapping pilot from unverified baseline source")
    selected = evidence["measurement"]
    if selected["PE"] != 2 or selected["SIMD"] != 1:
        raise ValueError("Expected certified H3 stem baseline PE2/SIMD1")
    folder = ROOT / selected["case_dir"]
    key = json.loads((folder / "measurement_key.json").read_text(encoding="utf-8"))
    case = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    return dict(node=stem, evidence=evidence, key=key, case=case, folder=folder,
                lookup_accounting=dict(database.counts))


def replay_reference(folder: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Independent full-size INT64/modulo32 NumPy oracle; never claim C-sim."""
    if (case["ci"], case["co"], case["h"], case["w"], case["k"], case["stride"],
            case["oh"], case["ow"], case["input_pack"], case["output_pack"]) != (1, 32, 640, 640, 3, 2, 320, 320, 1, 32):
        raise ValueError("Reference replay requires exact fixed-B1 stem shape")
    numeric = _numeric_module()
    node = case["nodes"][0]
    records = np.fromfile(folder / "layers.bin", dtype="<i4").reshape(-1, 32)
    record = records[node["layer_id"]]
    if record[1:9].tolist() != [1, 32, 640, 640, 3, 2, 1, 1]:
        raise ValueError("Retained numerical layer record shape mismatch")
    image = np.fromfile(folder / "input.bin", dtype=np.int8).reshape(640, 640, 1).transpose(2, 0, 1)
    weights = np.fromfile(folder / "weights.bin", dtype=np.int8, count=int(record[18]), offset=int(record[16])).reshape(32, 1, 3, 3)
    bias = np.fromfile(folder / "biases.bin", dtype="<i4", count=int(record[19]), offset=int(record[17]))
    meta = dict(stride=2, padding=1, groups=1, multiplier=int(record[12]), shift=int(record[13]),
                activation="relu", output_exp=int(record[11]))
    if list(record[12:16]) != [node["multiplier"], node["shift"], 1, node["q6"]]:
        raise ValueError("Retained numerical activation/requantization record mismatch")
    output = numeric.conv(image, weights, bias, meta).transpose(1, 2, 0).copy().tobytes()
    reference = (folder / "reference.bin").read_bytes()
    if output != reference:
        raise ValueError("Full-size NumPy integer reference replay mismatch")
    return dict(status="PASS_NUMPY_INTEGER_REFERENCE_REPLAY", samples=len(output),
                reference_sha256=hashlib.sha256(reference).hexdigest(),
                input_shape=[1, 640, 640], output_shape=[32, 320, 320],
                numeric_source_sha256=sha(ROOT / "tools/fpga_h2/numeric.py"),
                synthetic_contract_witness_not_trained_accuracy=True,
                modified_Cplusplus_csim_status="NOT_RUN")


def rewrite_parallelism(source: str, pe: int) -> str:
    if type(pe) is not int or pe not in PARALLELISM or 32 % pe:
        raise ValueError("Bounded pilot only permits PE8/16/32 with Cout32")
    if not re.search(r"static constexpr int IN_TILE = 1;", source):
        raise ValueError("Baseline input SIMD must be one")
    expected = "static constexpr int OUT_TILE = 2;"
    if source.count(expected) != 1:
        raise ValueError("Baseline output parallelism constant must occur exactly once")
    if "h2::finish(acc_tile[lane], numeric)" not in source:
        raise ValueError("Canonical H2 numeric finish missing")
    if "#pragma HLS ARRAY_PARTITION variable=weights dim=1 cyclic factor=OUT_TILE" not in source:
        raise ValueError("Template cannot partition weights by requested output PE")
    return source.replace(expected, f"static constexpr int OUT_TILE = {pe};")


def prepare_mapping_probe(output: str | Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new isolated mapping probe path; retained experiments are preserved")
    baseline = _baseline()
    source_folder, source_key, original_case = baseline["folder"], baseline["key"], baseline["case"]
    source = (source_folder / "kernel.cpp").read_text(encoding="utf-8")
    replay = replay_reference(source_folder, original_case)
    # Validate all requested transformations before any case is written.
    kernels = {pe: rewrite_parallelism(source, pe) for pe in PARALLELISM}
    output.mkdir(parents=True)
    cases = []
    for pe in PARALLELISM:
        name = f"stem_c12eb274bd029575_pe{pe}_simd1"
        folder = output / "cases" / name
        folder.mkdir(parents=True)
        for filename in (*WITNESS_FILES, "common_defs.h", "common_types.h", "numeric_v1.h", "tb.cpp", "run.tcl"):
            shutil.copyfile(source_folder / filename, folder / filename)
        (folder / "kernel.cpp").write_text(kernels[pe], encoding="utf-8")
        if sha(folder / "numeric_v1.h") != source_key["source_files"]["numeric_v1.h"]:
            raise ValueError("Numeric V1 implementation drift")
        case = dict(original_case, name=name,
                    source_hash=sha(folder / "kernel.cpp"),
                    mapping_probe=dict(PE=pe, SIMD=1, isolated_from_old_H3=True))
        (folder / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        source_files = {path.name: sha(path) for path in folder.iterdir() if path.suffix in (".h", ".cpp", ".tcl")}
        witnesses = {filename: sha(folder / filename) for filename in WITNESS_FILES}
        key = {name: value for name, value in source_key.items()
               if name not in ("extra", "sources", "split", "case_dir", "source_hash", "source_files")}
        key.update(point_id=name, PE=pe, SIMD=1, input_tile=1, output_tile=pe,
                   source_files=source_files, source_hash=canonical_hash(source_files),
                   witness_files=witnesses, source_baseline_point_id=baseline["evidence"]["measurement"]["point_id"],
                   source_baseline_report_sha256=baseline["evidence"]["measurement"]["source_report_sha256"],
                   case_dir=str(folder.resolve()), LEGAL_MAPPING_CHECK="PASS", rejection_reason="",
                   sources="NEW72_fixed_B1_stem_primitive_only", split="hardware_mapping_pilot_not_accuracy")
        if not (key["Cout"] % pe == 0 and key["Cin"] == key["SIMD"] == key["input_tile"] == 1
                and key["output_stream_bits"] == 256 and key["input_stream_bits"] == 8
                and key["linebuffer_bytes"] <= 262144 and key["part"] == PART
                and key["tool_version"] == "2022.2" and key["target_clock_ns"] == 5
                and all(key[field] == value for field, value in DEFAULT_PRECISION.items())):
            raise ValueError("Prepared exact shape/mapping/precision violates bounded probe contract")
        (folder / "measurement_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        cases.append(dict(name=name, case_dir=str(folder.resolve()), PE=pe, SIMD=1,
                          source_hash=key["source_hash"], source_files=source_files, witness_files=witnesses,
                          numerical_reference_status=replay["status"], csim_status="NOT_RUN", csynth_status="NOT_RUN",
                          latency_us=None, LUT=None, DSP=None, BRAM18=None,
                          resource_budget=RESOURCE_BUDGET,
                          expected_speedup=None, eligibility="UNMEASURED_NO_FEASIBILITY_CLAIM"))
    manifest = dict(schema_version=1, probe="fixed_B1_stem_parallelism_PE8_PE16_PE32_SIMD1",
                    stage="PREPARED_SOURCE_BOUND_CASES_NOT_HLS_RESULTS", case_count=3,
                    baseline_exact_measurement=baseline["evidence"]["measurement"],
                    baseline_source_files=source_key["source_files"],
                    baseline_witness_sha256={filename: sha(source_folder / filename) for filename in WITNESS_FILES},
                    baseline_numpy_reference_replay=replay, cases=cases,
                    measurement_scope="OP_LEVEL_HLS_ONLY_after_actual_CSIM_and_CSYNTH",
                    whole_network_fit_or_latency="NOT_CLAIMED", no_training=True, no_test_images_accessed=True,
                    no_hls_or_board_launch=True, parallel_probe_does_not_change_NEW72_model_architecture=True,
                    source_library_lookup_accounting=baseline["lookup_accounting"],
                    preparation_source_sha256=sha(Path(__file__)),
                    candidate_latency_budget_us=120000,
                    acceptance="Admit only source-bound bit-exact C-sim, timing-eligible 5ns CSYNTH, "
                               "and resource budgets; a faster stem alone does not close other 19 known "
                               "backbone costs or 41 missing NEW-neck primitives.")
    (output / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def readonly_remote_preflight() -> dict[str, Any]:
    """Use existing Credential Manager transport; inspect tool/CPU queue only."""
    from tools.detector_v0_2.hardware.remote_hls import preflight
    return preflight()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-preflight", action="store_true")
    args = parser.parse_args()
    manifest = prepare_mapping_probe(args.output)
    if args.remote_preflight:
        status = readonly_remote_preflight()
        (args.output / "remote_preflight.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(stage=manifest["stage"], cases=manifest["case_count"],
                          reference_replay=manifest["baseline_numpy_reference_replay"]["status"],
                          csim="NOT_RUN", csynth="NOT_RUN"), indent=2))


if __name__ == "__main__":
    main()
