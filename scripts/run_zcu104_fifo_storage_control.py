"""One-variable FIFO storage experiment on an already generated pilot harness.

Preserves architecture, RTL behavior, depth, clock, HLS cores and parameters.
Synthesizes first; only continues implementation when resource budgets pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resource_metrics(path):
    text = path.read_text()
    labels = {"lut": "CLB LUTs", "lut_as_logic": "LUT as Logic", "lut_as_memory": "LUT as Memory",
              "ff": "CLB Registers", "bram36": "Block RAM Tile", "dsp": "DSPs", "clb": "CLB"}
    result = {}
    for key, label in labels.items():
        match = re.search(r"\|\s*" + re.escape(label) + r"\s*\*?\s*\|\s*([\d,.]+)\s*\|", text)
        if not match:
            if key == "clb":
                result[key] = None  # CLB packing is not reported before placement.
                continue
            raise ValueError(f"Missing resource metric: {label}")
        result[key] = float(match.group(1).replace(",", ""))
    return result


def resource_violations(metrics):
    limits = {"lut": 161280, "lut_as_memory": 101760, "ff": 460800, "bram36": 312, "dsp": 1209}
    return [f"{key}>{cap}" for key, cap in limits.items() if metrics[key] > cap]


def resume_synthesized(args):
    """Resume only the completed, hash-verified synthesis of this exact control."""
    status_path = args.output / "status.json"
    previous = json.loads(status_path.read_text())
    name = "pilot_c_tail_e1_sync_distributed_fifo_d64"
    if previous.get("run_name") != name or previous.get("status") != "ERROR":
        raise ValueError("Resume requires the recorded failed post-synthesis control run")
    if previous.get("error") != "Missing resource metric: CLB LUTs":
        raise ValueError("This resume is limited to the known resource-parser failure")
    candidate = args.reference_output / "pilot_c_tail_e1_candidate.json"
    if digest(candidate) != previous["candidate_sha256"]:
        raise ValueError("Candidate changed since synthesis")
    run = args.work / name
    project = run / "harness_project"
    reference = args.reference_work / "integration/pilot_c_tail_e1/harness_project"
    hashes = previous.get("evidence_sha256", {})
    if not hashes:
        raise ValueError("Missing synthesis evidence hashes")
    for relative, expected in hashes.items():
        if digest(run / relative) != expected:
            raise ValueError(f"Changed synthesis evidence: {relative}")
    original_files = sorted(p.relative_to(reference) for d in ["rtl", "export_rtl"] for p in (reference / d).glob("*.v"))
    variant_files = sorted(p.relative_to(project) for d in ["rtl", "export_rtl"] for p in (project / d).glob("*.v"))
    if not original_files or original_files != variant_files:
        raise ValueError("RTL file set changed")
    for relative in original_files:
        expected = (reference / relative).read_text()
        if relative.as_posix() == "rtl/axis_simple_fifo.v":
            expected = expected.replace('ram_style = "block"', 'ram_style = "distributed"')
        if (project / relative).read_text() != expected:
            raise ValueError(f"RTL changed since the storage control: {relative}")
    metrics = resource_metrics(project / "reports/storage_control_synth_utilization.rpt")
    violations = resource_violations(metrics)
    backup = args.output / ("status_before_parser_fix_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    shutil.copy2(status_path, backup)
    state = dict(previous)
    state.pop("error", None)
    state.update(synthesis_metrics=metrics, synthesis_violations=violations,
                 resume_source_sha256=digest(Path(__file__)), verified_rtl_files=len(original_files),
                 status="SYNTHESIS_RESOURCE_REJECTED" if violations else "RUNNING_IMPLEMENTATION",
                 resumed_at=datetime.now(timezone.utc).isoformat())
    write(status_path, state)
    if violations:
        return
    original_tcl = (project / "scripts/build_bitstream.tcl").read_text()
    marker = "launch_runs impl_1 -to_step route_design -jobs 8"
    if original_tcl.count(marker) != 1:
        raise ValueError("Unexpected implementation script")
    tail = original_tcl.split(marker, 1)[1]
    tcl = project / "scripts/storage_control_route.tcl"
    tcl.write_text(original_tcl.split("create_project", 1)[0]
                   + '\nopen_project ${vivado_project_dir}/${project_name}.xpr\n' + marker + tail)
    command = ["/opt/Xilinx/Vivado/2022.2/bin/vivado", "-mode", "batch", "-source", str(tcl),
               "-log", str(args.output / "route_vivado.log"), "-journal", str(args.output / "route_vivado.jou")]
    state.setdefault("commands", []).append(command)
    write(status_path, state)
    try:
        with (args.output / "implementation.log").open("w") as log:
            process = subprocess.Popen(command, cwd=args.bundle, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = process.wait(timeout=240 * 60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("Implementation exceeded 240 minutes")
        state.update(status="IMPLEMENTATION_FINISHED_PENDING_REPORT_AUDIT" if code == 0 else "IMPLEMENTATION_FAILED",
                     implementation_returncode=code)
    except Exception as error:
        state.update(status="ERROR", error=str(error))
        raise
    finally:
        for source in run.rglob("*"):
            if source.is_file() and source.suffix in {".json", ".rpt", ".tcl", ".xdc", ".csv"}:
                relative = source.relative_to(run)
                dest = args.output / "evidence" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                hashes[str(relative)] = digest(dest)
        state.update(evidence_sha256=hashes, updated_at=datetime.now(timezone.utc).isoformat())
        write(status_path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-work", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--resume-after-synthesis", action="store_true")
    args = parser.parse_args()
    if args.resume_after_synthesis:
        return resume_synthesized(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.work.exists():
        raise FileExistsError("Use a new isolated work directory for each storage control")
    name = "pilot_c_tail_e1_sync_distributed_fifo_d64"
    reference = args.reference_work / "integration/pilot_c_tail_e1/harness_project"
    project = args.work / name / "harness_project"
    candidate = args.reference_output / "pilot_c_tail_e1_candidate.json"
    report = args.reference_output / "pilot_c_tail_e1_candidate_hls_report.json"
    state = {"run_name": name, "candidate_sha256": digest(candidate), "nas_resume_allowed": False,
             "started_at": datetime.now(timezone.utc).isoformat(), "commands": [],
             "evidence_scope": "deterministic hardware harness; no trained accuracy or board runtime"}
    status = args.output / "status.json"
    vivado = "/opt/Xilinx/Vivado/2022.2/bin/vivado"

    def execute(command, phase, cwd=None):
        state.update(status=f"RUNNING_{phase.upper()}")
        state["commands"].append(command)
        write(status, state)
        with (args.output / f"{phase}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = process.wait(timeout=(240 if phase == "implementation" else 90) * 60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"{phase} exceeded its time budget")
        if code:
            raise RuntimeError(f"{phase} failed: {code}")

    try:
        script = args.bundle / "hls_lut_builder/board_harness/scripts/rl_best_e2e_board_validation.py"
        execute([sys.executable, str(script), "generate", "--candidate-path", str(candidate),
                 "--candidate-hls-report", str(report), "--candidate-hls-cache-root",
                 str(args.reference_work / "candidate_cache"), "--result-root", str(args.work),
                 "--run-name", name, "--fifo-style", "bram", "--fifo-depth", "64",
                 "--board-config", str(args.bundle / "hls_lut_builder/board_harness/configs/board_zcu104.yaml")], "generate")
        rtl_files = sorted(p.relative_to(reference) for d in ["rtl", "export_rtl"]
                           for p in (reference / d).glob("*.v"))
        new_files = sorted(p.relative_to(project) for d in ["rtl", "export_rtl"]
                           for p in (project / d).glob("*.v"))
        if rtl_files != new_files:
            raise ValueError("Controlled build changed the RTL file set")
        if any((reference / p).read_text() != (project / p).read_text() for p in rtl_files):
            raise ValueError("Generated control does not match the reference RTL before the storage edit")
        fifo = project / "rtl/axis_simple_fifo.v"
        original = fifo.read_text()
        if original.count('ram_style = "block"') != 1:
            raise ValueError("Expected exactly one FIFO storage hint")
        fifo.write_text(original.replace('ram_style = "block"', 'ram_style = "distributed"'))
        after = fifo.read_text()
        if after.replace('ram_style = "distributed"', 'ram_style = "block"') != original:
            raise ValueError("FIFO behavior changed beyond the storage hint")
        contract = {"architecture_changed": False, "clock_mhz": 200, "fifo_depth": 64,
                    "physical_mapping": "independent physical instance per block",
                    "only_rtl_change": "axis_simple_fifo.v: ram_style block -> distributed",
                    "reference_fifo_sha256": digest(reference / "rtl/axis_simple_fifo.v"),
                    "variant_fifo_sha256": digest(fifo), "unchanged_rtl_files": len(rtl_files) - 1}
        write(args.output / "controlled_change.json", contract)
        manifest_path = project / "harness_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(harness_fifo_style="sync_distributed_same_control_as_bram", storage_control=contract)
        write(manifest_path, manifest)
        original_tcl = (project / "scripts/build_bitstream.tcl").read_text()
        marker = "launch_runs impl_1 -to_step route_design -jobs 8"
        if original_tcl.count(marker) != 1:
            raise ValueError("Unknown implementation script structure")
        prefix, tail = original_tcl.split(marker)
        synth_tcl = project / "scripts/storage_control_synth.tcl"
        synth_tcl.write_text(prefix + '\nopen_run synth_1\nfile mkdir $reports_output_dir\n'
                            'report_utilization -file ${reports_output_dir}/storage_control_synth_utilization.rpt\n'
                            'report_utilization -hierarchical -hierarchical_depth 2 -file ${reports_output_dir}/storage_control_synth_hier.rpt\nexit\n')
        execute([vivado, "-mode", "batch", "-source", str(synth_tcl), "-log", str(args.output / "synth_vivado.log"),
                 "-journal", str(args.output / "synth_vivado.jou")], "synthesis", args.bundle)
        metrics = resource_metrics(project / "reports/storage_control_synth_utilization.rpt")
        violations = resource_violations(metrics)
        state.update(synthesis_metrics=metrics, synthesis_violations=violations,
                     status="SYNTHESIS_RESOURCE_REJECTED" if violations else "SYNTHESIS_RESOURCES_PASS")
        write(status, state)
        if violations:
            return
        # Reopen the synthesized project, keeping its frozen implementation directives.
        definitions = original_tcl.split("create_project", 1)[0]
        continue_tcl = project / "scripts/storage_control_route.tcl"
        continue_tcl.write_text(definitions + '\nopen_project ${vivado_project_dir}/${project_name}.xpr\n'
                               + marker + tail)
        execute([vivado, "-mode", "batch", "-source", str(continue_tcl), "-log", str(args.output / "route_vivado.log"),
                 "-journal", str(args.output / "route_vivado.jou")], "implementation", args.bundle)
        state.update(status="IMPLEMENTATION_FINISHED_PENDING_REPORT_AUDIT")
    except Exception as error:
        state.update(status="ERROR", error=str(error))
        raise
    finally:
        hashes = {}
        run = args.work / name
        if run.exists():
            for source in run.rglob("*"):
                if source.is_file() and source.suffix in {".json", ".rpt", ".tcl", ".xdc", ".csv"}:
                    dest = args.output / "evidence" / source.relative_to(run)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    hashes[str(source.relative_to(run))] = digest(dest)
        state.update(evidence_sha256=hashes, updated_at=datetime.now(timezone.utc).isoformat())
        write(status, state)


if __name__ == "__main__":
    main()
