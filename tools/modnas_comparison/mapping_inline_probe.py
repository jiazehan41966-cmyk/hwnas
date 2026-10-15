"""Prepare ONE PE8 follow-up to test the measured shared requant bottleneck.

The source edit adds HLS INLINE to finish and requant in a NEW header copy.
Deleting those two pragma lines must recover the exact original header bytes.
Numerical and interface contracts, all witness bytes, PE8 and TCL stay fixed.
This program neither launches HLS nor changes the serial 120ms budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.modnas_elastic.cost_graph import canonical_hash
from tools.modnas_comparison.mapping_probe import WITNESS_FILES, replay_reference, sha
from tools.modnas_comparison.remote_worker import parse_report, verify_case

DEFAULT_PREVIOUS = ROOT / "results/modnas_vs_ours_20261007/mapping_probe"
INLINE_PRAGMA = "#pragma HLS INLINE\n"


def force_inline_numeric(source: str) -> str:
    signatures = ("inline ap_int<8> requant(ap_int<32> a,int multiplier,int shift) {\n",
                  "inline ap_int<8> finish(ap_int<32> a,const int* p) {\n")
    if "#pragma HLS INLINE" in source:
        raise ValueError("Original numeric header has unexpected INLINE directives")
    transformed = source
    for signature in signatures:
        if transformed.count(signature) != 1:
            raise ValueError("Numeric function signature drift; cannot preserve exact arithmetic")
        transformed = transformed.replace(signature, signature + INLINE_PRAGMA)
    if transformed.count(INLINE_PRAGMA) != 2 or transformed.replace(INLINE_PRAGMA, "") != source:
        raise ValueError("Only two scheduling pragma additions are permitted")
    return transformed


def diagnose_previous(previous: str | Path = DEFAULT_PREVIOUS) -> dict:
    previous = Path(previous)
    evidence = previous / "remote_evidence"
    preparation = json.loads((previous / "preparation_manifest.json").read_text(encoding="utf-8"))
    baseline_latency = preparation["baseline_exact_measurement"]["latency_us"]
    measured = []
    for record in preparation["cases"]:
        folder = evidence / "cases" / record["name"]
        key = verify_case(folder)
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if result["status"] != "PASS_CSIM_AND_CSYNTH" or result["source_hash"] != key["source_hash"]:
            raise ValueError("Previous measured probe lacks source-bound complete status")
        output = folder / "project/solution1/csim/build/hardware_output.bin"
        if sha(output) != sha(folder / "reference.bin") or sha(output) != result["csim_output_sha256"]:
            raise ValueError("Previous complete C-sim byte witness mismatch")
        report_dir = folder / "project/solution1/syn/report"
        report = report_dir / "closure_kernel_csynth.xml"
        if sha(report) != result["source_report_sha256"]:
            raise ValueError("Previous CSYNTH report hash mismatch")
        metrics = parse_report(report)
        if metrics != result["metrics"]:
            raise ValueError("Previous recorded metrics disagree with raw XML")
        top = ET.parse(report).getroot()
        scan = list(top.find(".//SummaryOfLoopLatency"))[0]
        group = next((child for child in scan if child.find("TripCount") is not None), None)
        def loop_value(element, tag):
            child = element.find(tag)
            return child.text if len(child) == 0 else dict(min=child.findtext("range/min"), max=child.findtext("range/max"))
        finish_path = report_dir / "finish_r_csynth.xml"
        helper = ET.parse(finish_path).getroot()
        top_rpt = report_dir / "closure_kernel_csynth.rpt"
        text = top_rpt.read_text(encoding="utf-8")
        if "grp_finish_r_fu_" not in text or "|finish_r " not in text:
            raise ValueError("Shared finish helper evidence is absent; revise causal diagnosis first")
        measured.append(dict(PE=key["PE"], SIMD=key["SIMD"], source_hash=key["source_hash"],
                             metrics=metrics, speedup_vs_PE2=baseline_latency / metrics["latency_us"],
                             scan_loop=dict(name=scan.tag, trip_count=int(scan.findtext("TripCount")),
                                            iteration_latency=loop_value(scan, "IterationLatency"),
                                            pipeline="NOT_PIPELINED_in_top_rpt"),
                             output_group_loop=dict(name=group.tag if group is not None else "ELIMINATED_SINGLE_GROUP_PE32",
                                                    trip_count=int(group.findtext("TripCount")) if group is not None else 1,
                                                    iteration_latency=int(group.findtext("IterationLatency")) if group is not None else None),
                             finish_helper=dict(module="finish_r", single_instance_in_top_report=True,
                                                latency_cycles=int(helper.findtext(".//SummaryOfOverallLatency/Worst-caseLatency")),
                                                interval_cycles=int(helper.findtext(".//SummaryOfOverallLatency/Interval-max")),
                                                DSP=int(helper.findtext(".//AreaEstimates/Resources/DSP")),
                                                LUT=int(helper.findtext(".//AreaEstimates/Resources/LUT"))),
                             evidence_files={str(path): sha(path) for path in (report, top_rpt, finish_path)}))
    return dict(schema_version=1, status="SOURCE_BOUND_POST_CSYNTH_CAUSAL_DIAGNOSIS",
                baseline_PE2_latency_us=baseline_latency, previous= str(previous.resolve()), measured=measured,
                observed="All PE8/16/32 MAC lane counts increase, while finish_r remains one six-cycle "
                         "shared helper. Output lane UNROLL therefore does not parallelize the integer finish path.",
                inference="Shared requantization/activation and a sequential pixel loop explain diminishing "
                          "benefit from increasing PE. Inline the helper before changing pixel scheduling.",
                next_probe="ONE_PE8_INLINE_finish_AND_requant_ONLY",
                counterfactual_speedup_or_resources=None,
                budget_or_interface_change=False,
                CSYNTH_bound_note="Top Worst-caseLatency applies the largest branch latency to every "
                                    "scan-loop iteration. This is a conservative operator CSYNTH estimate; "
                                    "it is not measured cycle-accurate board latency.")


def prepare_inline_probe(output: str | Path, *, previous: str | Path = DEFAULT_PREVIOUS) -> dict:
    output, previous = Path(output), Path(previous)
    if output.exists():
        raise FileExistsError("Preserve previous HLS attempts; use a fresh inline probe directory")
    diagnosis = diagnose_previous(previous)
    source_folder = previous / "cases/stem_c12eb274bd029575_pe8_simd1"
    source_key = verify_case(source_folder)
    case = json.loads((source_folder / "case.json").read_text(encoding="utf-8"))
    reference_replay = replay_reference(source_folder, case)
    original_header_bytes = (source_folder / "numeric_v1.h").read_bytes()
    original_header = original_header_bytes.decode("utf-8")
    modified_header = force_inline_numeric(original_header)
    output.mkdir(parents=True)
    diagnosis_path = output / "causal_diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
    name = "stem_c12eb274bd029575_pe8_simd1_inline_finish_requant"
    folder = output / "cases" / name
    folder.mkdir(parents=True)
    for filename in (*WITNESS_FILES, "common_defs.h", "common_types.h", "numeric_v1.h", "tb.cpp", "run.tcl", "kernel.cpp"):
        shutil.copyfile(source_folder / filename, folder / filename)
    (folder / "numeric_v1.h").write_bytes(modified_header.encode("utf-8"))
    if (folder / "numeric_v1.h").read_bytes().replace(INLINE_PRAGMA.encode("utf-8"), b"") != original_header_bytes:
        raise ValueError("Original header bytes changed while writing scheduling pragmas")
    case = dict(case, name=name, numeric_implementation_variant="INLINE_FINISH_REQUANT_PRESERVING_V1_ARITHMETIC")
    (folder / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    source_files = {path.name: sha(path) for path in folder.iterdir() if path.suffix in (".h", ".cpp", ".tcl")}
    witnesses = {filename: sha(folder / filename) for filename in WITNESS_FILES}
    if witnesses != source_key["witness_files"]:
        raise ValueError("Scheduling probe changed retained numerical witness bytes")
    key = deepcopy(source_key)
    key.update(point_id=name, case_dir=str(folder.resolve()), source_files=source_files,
               source_hash=canonical_hash(source_files), witness_files=witnesses,
               numeric_implementation_variant="INLINE_FINISH_REQUANT_PRESERVING_V1_ARITHMETIC",
               original_numeric_header_sha256=sha(source_folder / "numeric_v1.h"),
               parent_probe_source_hash=source_key["source_hash"])
    (folder / "measurement_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    verify_case(folder)
    record = dict(name=name, case_dir=str(folder.resolve()), PE=8, SIMD=1,
                  source_hash=key["source_hash"], source_files=source_files, witness_files=witnesses,
                  numerical_reference_status=reference_replay["status"], csim_status="NOT_RUN", csynth_status="NOT_RUN",
                  latency_us=None, LUT=None, DSP=None, BRAM18=None, expected_speedup=None,
                  eligibility="UNMEASURED_SOURCE_CHANGED_ACTUAL_CSIM_CSYNTH_REQUIRED")
    manifest = dict(schema_version=1, probe="ONE_PE8_INLINE_finish_requant_causal_followup",
                    stage="PREPARED_SOURCE_BOUND_CASES_NOT_HLS_RESULTS", case_count=1, cases=[record],
                    parent_probe=str(previous.resolve()), causal_diagnosis_sha256=sha(diagnosis_path),
                    source_change="Only two HLS INLINE pragmas inside new numeric_v1.h copy",
                    original_numeric_header_sha256=sha(source_folder / "numeric_v1.h"),
                    semantic_numeric_contract_hash=key["numeric_contract_hash"],
                    interface_contract_hash=key["interface_contract_hash"],
                    baseline_numpy_reference_replay=reference_replay,
                    no_budget_change=True, no_arithmetic_change=True, no_pixel_pipeline_change=True,
                    no_hls_or_board_launch=True, no_training=True, no_test_images_accessed=True,
                    preparation_source_sha256=sha(Path(__file__)),
                    acceptance="Require full-byte and framing C-sim parity, source-bound XML, 5ns timing and "
                               "resource eligibility. Test whether shared finish_r disappears and latency "
                               "improves before adding pixel-pipeline or other changes.")
    (output / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    args = parser.parse_args()
    manifest = prepare_inline_probe(args.output, previous=args.previous)
    print(json.dumps(dict(case_count=manifest["case_count"], stage=manifest["stage"],
                          change=manifest["source_change"], csim="NOT_RUN", csynth="NOT_RUN"), indent=2))


if __name__ == "__main__":
    main()
