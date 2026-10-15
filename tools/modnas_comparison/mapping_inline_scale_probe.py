"""Prepare two PE16/PE32 follow-ups after the measured PE8 INLINE result.

Copy the exact retained PE16/32 kernels and witnesses, and add only the same
two scheduling pragmas to numeric_v1.h. No HLS, board or training is launched.
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

from tools.modnas_comparison.mapping_inline_probe import (
    DEFAULT_PREVIOUS, INLINE_PRAGMA, force_inline_numeric,
)
from tools.modnas_comparison.mapping_probe import RESOURCE_BUDGET, WITNESS_FILES, replay_reference, sha
from tools.modnas_comparison.remote_worker import parse_report, verify_case
from tools.modnas_elastic.cost_graph import canonical_hash

DEFAULT_INLINE = ROOT / "results/modnas_vs_ours_20261007/mapping_inline_probe"
CASE_FILES = (*WITNESS_FILES, "common_defs.h", "common_types.h", "numeric_v1.h", "tb.cpp", "run.tcl", "kernel.cpp")


def diagnose_inline_pe8(inline: str | Path = DEFAULT_INLINE,
                        previous: str | Path = DEFAULT_PREVIOUS) -> dict:
    """Recheck complete PE8 evidence and the disappearance of shared finish_r."""
    inline, previous = Path(inline), Path(previous)
    verified_path = inline / "verified_results.json"
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    prep_path = inline / "preparation_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if verified["status"] != "PASS_EVIDENCE_COLLECTION" or len(verified["results"]) != 1:
        raise ValueError("PE8 INLINE must have complete collected measurement evidence")
    if verified["preparation_manifest_sha256"] != sha(prep_path) or prep["case_count"] != 1:
        raise ValueError("PE8 INLINE preparation provenance drift")
    row = verified["results"][0]
    if row["PE"] != 8 or row["status"] != "PASS_CSIM_AND_CSYNTH":
        raise ValueError("Expected complete PE8 INLINE C-sim and CSYNTH")
    folder = inline / "remote_evidence/cases" / row["case_name"]
    key = verify_case(folder)
    if key["source_hash"] != canonical_hash(key["source_files"]) or key["source_hash"] != row["source_hash"]:
        raise ValueError("PE8 INLINE aggregate source hash mismatch")
    if prep["cases"][0]["source_hash"] != key["source_hash"]:
        raise ValueError("PE8 INLINE result differs from the prepared source")
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["source_hash"] != row["source_hash"] or result["status"] != row["status"]:
        raise ValueError("PE8 INLINE raw result provenance mismatch")
    report = folder / "project/solution1/syn/report/closure_kernel_csynth.xml"
    rpt = report.with_suffix(".rpt")
    output = folder / "project/solution1/csim/build/hardware_output.bin"
    if not output.is_file() or output.stat().st_size != 3276800:
        raise ValueError("Complete PE8 INLINE output bytes are unavailable")
    if sha(output) != sha(folder / "reference.bin") or sha(output) != result["csim_output_sha256"]:
        raise ValueError("PE8 INLINE integer byte witness mismatch")
    if sha(report) != result["source_report_sha256"] or sha(report) != row["report_sha256"]:
        raise ValueError("PE8 INLINE raw XML provenance mismatch")
    metrics = parse_report(report)
    if metrics != row["metrics"] or metrics != result["metrics"]:
        raise ValueError("PE8 INLINE stored metrics differ from actual raw XML")
    if not metrics["timing_eligible"] or not metrics["resource_screen_eligible"]:
        raise ValueError("PE8 INLINE is not eligible for the next fixed-budget probe")
    if "finish_r" in rpt.read_text(encoding="utf-8"):
        raise ValueError("Shared finish_r remains: revise the causal follow-up before scaling")
    old_prep = json.loads((previous / "preparation_manifest.json").read_text(encoding="utf-8"))
    baseline = old_prep["baseline_exact_measurement"]
    if baseline != verified["baseline_exact_measurement"]:
        raise ValueError("Original PE2 baseline identity drift")
    baseline_report = ROOT / baseline["source_report_path"]
    if sha(baseline_report) != baseline["source_report_sha256"]:
        raise ValueError("Original PE2 raw report hash drift")
    old_header = previous / "cases/stem_c12eb274bd029575_pe8_simd1/numeric_v1.h"
    header = (folder / "numeric_v1.h").read_bytes()
    pragma = INLINE_PRAGMA.encode("utf-8")
    if header.count(pragma) != 2 or header.replace(pragma, b"") != old_header.read_bytes():
        raise ValueError("Measured PE8 INLINE changed bytes beyond the two scheduling pragmas")
    xml = ET.parse(report).getroot()
    scan = list(xml.find(".//SummaryOfLoopLatency"))[0]
    group = next((child for child in scan if child.find("TripCount") is not None), None)
    if group is None:
        raise ValueError("PE8 output-group scheduling evidence is unavailable")
    return dict(schema_version=1, status="SOURCE_BOUND_PE8_INLINE_CAUSAL_EVIDENCE",
                case_name=row["case_name"], source_hash=key["source_hash"], metrics=metrics,
                numeric_contract_hash=key["numeric_contract_hash"], interface_contract_hash=key["interface_contract_hash"],
                full_byte_csim_verified=True, output_bytes=output.stat().st_size, shared_finish_helper_present=False,
                scan_loop=dict(name=scan.tag, trip_count=int(scan.findtext("TripCount")),
                               maximum_iteration_latency_cycles=int(scan.findtext("IterationLatency/range/max"))),
                output_group_loop=dict(name=group.tag, trip_count=int(group.findtext("TripCount")),
                                       iteration_latency_cycles=int(group.findtext("IterationLatency"))),
                evidence_files={str(p.resolve()): sha(p) for p in (verified_path, prep_path, result_path, report, rpt, output)},
                baseline_exact_measurement=baseline,
                observed="The source-bound PE8 INLINE report has no shared finish_r. Its four output groups "
                         "remain sequential; inspect actual PE16/32 reports before changing pixel scheduling.",
                inference="Scale internal output lanes with the same two INLINE pragmas to test group overhead.",
                counterfactual_latency_or_resources=None, board_measured=False, full_network_measured=False)


def prepare_inline_scale_probe(output: str | Path, *, previous: str | Path = DEFAULT_PREVIOUS,
                               inline: str | Path = DEFAULT_INLINE) -> dict:
    output, previous = Path(output), Path(previous)
    if output.exists():
        raise FileExistsError("Preserve earlier measured attempts; use a fresh inline scale probe directory")
    diagnosis = diagnose_inline_pe8(inline, previous)
    prepared = []
    for pe in (16, 32):
        source = previous / "cases" / f"stem_c12eb274bd029575_pe{pe}_simd1"
        source_key = verify_case(source)
        if source_key["source_hash"] != canonical_hash(source_key["source_files"]):
            raise ValueError("Retained PE source aggregate hash drift")
        case = json.loads((source / "case.json").read_text(encoding="utf-8"))
        original_header = (source / "numeric_v1.h").read_bytes()
        header = force_inline_numeric(original_header.decode("utf-8")).encode("utf-8")
        if header.replace(INLINE_PRAGMA.encode("utf-8"), b"") != original_header:
            raise ValueError("Only exact two pragma bytes may change the original header")
        replay = replay_reference(source, case)
        prepared.append((pe, source, source_key, case, header, replay))
    output.mkdir(parents=True)
    diagnosis_path = output / "causal_diagnosis.json"
    diagnosis_path.write_bytes((json.dumps(diagnosis, indent=2) + "\n").encode("utf-8"))
    records = []
    for pe, source, source_key, case, header, replay in prepared:
        name = f"stem_c12eb274bd029575_pe{pe}_simd1_inline_finish_requant"
        folder = output / "cases" / name
        folder.mkdir(parents=True)
        for filename in CASE_FILES:
            shutil.copyfile(source / filename, folder / filename)
        (folder / "numeric_v1.h").write_bytes(header)
        case = dict(case, name=name, numeric_implementation_variant="INLINE_FINISH_REQUANT_PRESERVING_V1_ARITHMETIC")
        (folder / "case.json").write_bytes((json.dumps(case, indent=2) + "\n").encode("utf-8"))
        source_files = {p.name: sha(p) for p in folder.iterdir() if p.suffix in (".h", ".cpp", ".tcl")}
        witnesses = {filename: sha(folder / filename) for filename in WITNESS_FILES}
        if witnesses != source_key["witness_files"]:
            raise ValueError("Inline scale probe changed a retained numerical witness")
        key = deepcopy(source_key)
        key.update(point_id=name, case_dir=str(folder.resolve()), source_files=source_files,
                   source_hash=canonical_hash(source_files), witness_files=witnesses,
                   numeric_implementation_variant="INLINE_FINISH_REQUANT_PRESERVING_V1_ARITHMETIC",
                   original_numeric_header_sha256=sha(source / "numeric_v1.h"),
                   parent_probe_source_hash=source_key["source_hash"],
                   measured_PE8_inline_source_hash=diagnosis["source_hash"])
        (folder / "measurement_key.json").write_bytes((json.dumps(key, indent=2) + "\n").encode("utf-8"))
        verify_case(folder)
        records.append(dict(name=name, case_dir=str(folder.resolve()), PE=pe, SIMD=1,
                            source_hash=key["source_hash"], source_files=source_files, witness_files=witnesses,
                            numerical_reference_status=replay["status"], csim_status="NOT_RUN", csynth_status="NOT_RUN",
                            latency_us=None, LUT=None, DSP=None, BRAM18=None, expected_speedup=None,
                            resource_budget=RESOURCE_BUDGET,
                            eligibility="UNMEASURED_SOURCE_CHANGED_ACTUAL_CSIM_CSYNTH_REQUIRED"))
    manifest = dict(schema_version=1, probe="PE16_PE32_INLINE_finish_requant_after_measured_PE8",
                    stage="PREPARED_SOURCE_BOUND_CASES_NOT_HLS_RESULTS", case_count=2, cases=records,
                    baseline_exact_measurement=diagnosis["baseline_exact_measurement"],
                    parent_probe=str(previous.resolve()), measured_inline_probe=str(Path(inline).resolve()),
                    measured_PE8_inline_evidence=diagnosis, causal_diagnosis_sha256=sha(diagnosis_path),
                    baseline_numpy_reference_replay=prepared[0][5],
                    source_change="Copy retained PE16/32 source bytes and add only two HLS INLINE pragmas to numeric_v1.h",
                    semantic_numeric_contract_hash=diagnosis["numeric_contract_hash"],
                    interface_contract_hash=diagnosis["interface_contract_hash"], candidate_latency_budget_us=120000,
                    measurement_scope="OP_LEVEL_HLS_ONLY_after_actual_CSIM_and_CSYNTH",
                    no_budget_change=True, no_arithmetic_change=True, no_pixel_pipeline_change=True,
                    no_hls_or_board_launch=True, no_training=True, no_test_images_accessed=True,
                    preparation_source_sha256=sha(Path(__file__)),
                    acceptance="Require complete byte/framing C-sim parity, source-bound 2022.2 XML, 5ns timing and "
                               "fixed resource budgets. Inspect both actual reports before another source change; "
                               "stem results do not establish full-network latency or board feasibility.")
    (output / "preparation_manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--inline", type=Path, default=DEFAULT_INLINE)
    args = parser.parse_args()
    manifest = prepare_inline_scale_probe(args.output, previous=args.previous, inline=args.inline)
    print(json.dumps(dict(case_count=manifest["case_count"], stage=manifest["stage"],
                          measured_PE8_latency_us=manifest["measured_PE8_inline_evidence"]["metrics"]["latency_us"],
                          csim="NOT_RUN", csynth="NOT_RUN"), indent=2))


if __name__ == "__main__":
    main()
