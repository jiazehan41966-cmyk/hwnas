"""Verify downloaded pilot evidence and build a strictly scoped measured LUT.

The small grid is an operator-cost diagnostic, not an admitted NAS space.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def aggregate(names, records):
    # The integrated harness connects an identity Skip directly, without a core.
    selected = [records[n] for n in names if not n.startswith("skip_")]
    metrics = {key: sum(r["post_route"][key] for r in selected)
               for key in ["lut", "lut_as_logic", "lut_as_memory", "ff", "dsp", "block_ram_tile"]}
    metrics["bram36"] = math.ceil(metrics.pop("block_ram_tile"))
    metrics["latency_ms"] = sum(r["hls"]["latency_cycles"] for r in selected) / 200000
    metrics["energy_mj_vectorless_proxy"] = sum(r["hls"]["latency_cycles"] / 200000 * r["post_route"]["power_w"] for r in selected)
    metrics["max_operator_power_w_vectorless_proxy"] = max(r["post_route"]["power_w"] for r in selected)
    limits = {"bram36": 312, "lut": 161280, "ff": 460800, "dsp": 1209,
              "lut_as_memory": 101760, "latency_ms": 120,
              "energy_mj_vectorless_proxy": 375, "max_operator_power_w_vectorless_proxy": 15}
    violations = [f"{k}>{v}" for k, v in limits.items() if metrics[k] > v]
    if any(not r["timing_met"] for r in selected):
        violations.append("operator_timing")
    return {"metrics": metrics, "proxy_feasible": not violations, "violations": violations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest = read(root / "pilot_manifest.json")
    records = {}
    verified = 0
    for op in manifest["operators"]:
        name = op["case_name"]
        r = read(root / "remote_output/operators" / f"{name}.json")
        if r["status"] != "PASS_OPERATOR_REPORTS" or r["source_sha256"] != op["source_sha256_lf"]:
            raise ValueError(f"Incomplete or mismatched operator {name}")
        for rel, expected in r["evidence_sha256"].items():
            path = root / "remote_output/evidence" / name / rel
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Evidence hash mismatch: {path}")
            verified += 1
        route = r["post_route"]
        if route["block_ram_tile"] != route["ramb36"] + route["ramb18"] / 2:
            raise ValueError(f"BRAM unit mismatch: {name}")
        if route["lut"] != route["lut_as_logic"] + route["lut_as_memory"]:
            raise ValueError(f"LUT accounting mismatch: {name}")
        records[name] = r
    write(root / "evidence_audit.json", {"status": "PASS", "operators": len(records),
          "verified_file_hashes": verified, "operator_timing_met": sum(r["timing_met"] for r in records.values()),
          "bram_units": "RAMB36 + RAMB18 / 2", "lut_accounting": "logic + memory",
          "full_network_route": False, "nas_resume_allowed": False})
    lut = []
    for op in manifest["operators"]:
        r = records[op["case_name"]]
        lut.append({"case_name": op["case_name"], "parameters": op["parameters"],
                    "source_sha256": r["source_sha256"], "hls": r["hls"], "post_route": r["post_route"],
                    "evidence_sha256": r["evidence_sha256"], "timing_met": r["timing_met"]})
    write(root / "measured_operator_lut.json", {"schema_version": 1, "scope": "34 exact frozen pilot cases",
          "integration_calibrated": False, "nas_admitted": False, "rows": lut,
          "latency_rule": "latency_cycles / 200000 ms; HLS estimated critical period is not the scheduled 5 ns clock"})
    by_case = {r["case_name"]: r for r in manifest["operators"]}
    drift = []
    for name in manifest["pilots"][0]["case_names"]:
        op = by_case[name]
        old = op.get("legacy_row") or {}
        new = records[name]
        drift.append({"operator": op["operator_name"],
                      "old_bram36": old.get("post_route_metrics", {}).get("block_ram_tile"),
                      "new_bram36": new["post_route"]["block_ram_tile"],
                      "old_cycles": old.get("hls_metrics", {}).get("latency_cycles"),
                      "new_cycles": new["hls"]["latency_cycles"],
                      "source_equal": new["source_match_checks"].get("source"),
                      "all_implementation_files_equal": all(new["source_match_checks"].get(k) is True
                          for k in ["source", "tcl", "header:common_types.h", "header:common_defs.h"])})
    with (root / "baseline_lut_drift.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(drift[0])); writer.writeheader(); writer.writerows(drift)
    adjusted = [{"pilot_id": p["pilot_id"], **aggregate(p["case_names"], records)} for p in manifest["pilots"]]
    write(root / "mapping_aligned_pilot_results.json", {
        "skip_policy": "identity wire; subtract isolated Skip core from the original operator sum",
        "integration_overhead_included": False, "pilots": adjusted,
        "proxy_feasible_count": sum(p["proxy_feasible"] for p in adjusted), "nas_resume_allowed": False})
    # Independent tail expansions reuse already measured exact shapes. Widths
    # remain tied profiles, so no unmeasured cross-profile transition is invented.
    grid = []
    groups = {}
    for p in manifest["pilots"]:
        groups.setdefault(tuple(p["channels"]), {})[p["tail_expansion"]] = p
    for widths, expansions in groups.items():
        for e7, e8 in itertools.product([1, 2, 3], repeat=2):
            names = list(expansions[e7]["case_names"])
            names[8] = expansions[e8]["case_names"][8]
            grid.append({"id": f"w{'_'.join(map(str, widths[-3:]))}_e{e7}_{e8}",
                         "widths": list(widths), "e7": e7, "e8": e8,
                         "case_names": names, **aggregate(names, records)})
    rng = random.Random(42)
    sample = [rng.choice(grid) for _ in range(1000)]
    write(root / "exploratory_grid_preflight.json", {
        "status": "DIAGNOSTIC_ONLY_PENDING_INTEGRATION_CALIBRATION", "seed": 42,
        "grid_size": len(grid), "exact_proxy_feasible_count": sum(p["proxy_feasible"] for p in grid),
        "sample_count": 1000, "sample_proxy_feasible_count": sum(p["proxy_feasible"] for p in sample),
        "sample_unique_count": len({p["id"] for p in sample}), "candidates": grid,
        "excluded_unmeasured_variables": ["DMDC/CFEM admission", "new k5 shapes", "independent arbitrary stage widths"],
        "nas_resume_allowed": False, "physical_feasibility_claim": "none"})
    print(json.dumps({"operators": len(records), "verified_hashes": verified,
                      "pilot_feasible": sum(p["proxy_feasible"] for p in adjusted),
                      "grid_feasible": sum(p["proxy_feasible"] for p in grid)}))


if __name__ == "__main__":
    main()
