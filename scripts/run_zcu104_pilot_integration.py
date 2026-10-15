"""Integrate a frozen width/e pilot with the existing whole-network harness.

Hardware-only deterministic parameters; never programs the board or starts NAS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--operator-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--pilot", default="pilot_c_tail_e1")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle / "pilot_manifest.json").read_text())
    pilot = next(p for p in manifest["pilots"] if p["pilot_id"] == args.pilot)
    op_map = {r["case_name"]: r for r in manifest["operators"]}
    candidate = args.output / f"{args.pilot}_candidate.json"
    candidate_data = {"arch_id": args.pilot, "encoding": pilot["architecture"],
                      "accuracy_claim": "none; hardware pilot only"}
    if candidate.exists() and json.loads(candidate.read_text()) != candidate_data:
        raise ValueError("Refusing to replace a different frozen candidate")
    write(candidate, candidate_data)
    cache = args.work / "candidate_cache"
    cache.mkdir(exist_ok=True)
    roles = []
    role_names = ["stem", *[f"stage{i}_block0" for i in range(8)],
                  "head_conv", "global_avg_pool", "classifier"]
    for name, role in zip(pilot["case_names"], role_names, strict=True):
        record = json.loads((args.operator_output / "operators" / f"{name}.json").read_text())
        if record["status"] != "PASS_OPERATOR_REPORTS" or not record["timing_met"]:
            raise ValueError(f"Operator evidence incomplete: {name}")
        root = Path(record["source_root"])
        for rel, expected in record["evidence_sha256"].items():
            if sha(root / rel) != expected:
                raise ValueError(f"Changed operator evidence: {root / rel}")
        logical_op = op_map[name]["parameters"]["operator_name"].split("_h")[0]
        if logical_op == "skip":
            roles.append({"role": role, "logical_op": "skip", "evidence_status": "structural_skip"})
            continue
        link = cache / name
        if not link.exists():
            link.symlink_to(root, target_is_directory=True)
        if link.resolve() != root.resolve():
            raise ValueError(f"Cache identity mismatch: {name}")
        xml_rel = next(k for k in record["evidence_sha256"] if k.endswith("_csynth.xml"))
        roles.append({"role": role, "logical_op": logical_op, "case_name": name,
                      "evidence_status": "measured_csynth", "report_path": str(root / xml_rel),
                      "report_sha256": record["evidence_sha256"][xml_rel]})
    report = args.output / f"{args.pilot}_candidate_hls_report.json"
    write(report, {"evidence_complete": True, "evidence_scope": "isolated HLS only",
                   "candidate_sha256": sha(candidate), "roles": roles})
    script = args.bundle / "hls_lut_builder/board_harness/scripts/rl_best_e2e_board_validation.py"
    result_root = args.work / "integration"
    common = ["--candidate-path", str(candidate), "--candidate-hls-report", str(report),
              "--candidate-hls-cache-root", str(cache), "--result-root", str(result_root),
              "--run-name", args.pilot, "--fifo-style", "bram", "--fifo-depth", "64",
              "--board-config", str(args.bundle / "hls_lut_builder/board_harness/configs/board_zcu104.yaml"),
              "--vivado", "/opt/Xilinx/Vivado/2022.2/bin/vivado",
              "--bitstream-timeout-minutes", "240"]
    state_path = args.output / f"{args.pilot}_integration_status.json"
    state = {"pilot_id": args.pilot, "candidate_sha256": sha(candidate),
             "started_at": datetime.now(timezone.utc).isoformat(), "nas_resume_allowed": False,
             "parameter_mode": "latency_only_deterministic", "board_runtime_measured": False,
             "harness_fifo_depth": 64, "harness_fifo_style": "bram", "commands": []}
    try:
        project_manifest = result_root / args.pilot / "project/harness_manifest.json"
        # Existing harness computes its own project path; generated spec is the resume marker.
        generated = list((result_root / args.pilot).rglob("harness_manifest.json"))
        for mode in (["generate"] if args.generate_only else (["build"] if generated else ["generate", "build"])):
            command = [sys.executable, str(script), mode, *common]
            state.update(status=f"RUNNING_{mode.upper()}")
            state["commands"].append(command)
            write(state_path, state)
            with (args.output / f"{args.pilot}_{mode}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError(f"{mode} exited {result.returncode}; inspect {args.pilot}_{mode}.log")
        build_path = result_root / args.pilot / "build_result.json"
        build = json.loads(build_path.read_text()) if build_path.exists() else {}
        state.update(status="GENERATED" if args.generate_only else "BUILD_FINISHED",
                     build_status=build.get("status"), failure_summary=build.get("failure_summary"),
                     full_network_implementation_accepted=None)
    except Exception as error:
        state.update(status="ERROR", error=str(error), full_network_implementation_accepted=None)
        raise
    finally:
        evidence = args.output / args.pilot
        run = result_root / args.pilot
        hashes = {}
        if run.exists():
            for source in run.rglob("*"):
                if source.is_file() and source.suffix in {".json", ".rpt", ".log", ".tcl", ".xdc", ".csv"}:
                    dest = evidence / source.relative_to(run)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    hashes[str(source.relative_to(run))] = sha(dest)
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), evidence_sha256=hashes)
        write(state_path, state)


if __name__ == "__main__":
    main()
