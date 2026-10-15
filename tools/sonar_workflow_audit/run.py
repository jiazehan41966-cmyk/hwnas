"""Read-only reconciliation of dated sonar workflow evidence; never runs training.

Outputs go to a new directory. Test datasets and checkpoints are not opened.
Optional runtime/hardware observations are evidence snapshots, not live probes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def truth(value):
    return value is True or value == "True"


def candidate_key(row):
    return tuple(row[k] for k in ("architecture_id", "profile", "mapping_policy"))


def pareto(rows, costs):
    """Maximize accuracy, minimize each cost; preserve distinct tied mappings."""
    points = [(float(r["Accuracy"]), *(float(r[k]) for k in costs)) for r in rows]
    if any(not math.isfinite(v) for point in points for v in point):
        raise ValueError("Non-finite Pareto input")
    return [row for row, point in zip(rows, points) if not any(
        other[0] >= point[0] and all(a <= b for a, b in zip(other[1:], point[1:]))
        and (other[0] > point[0] or any(a < b for a, b in zip(other[1:], point[1:])))
        for other in points)]


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.sources = {}

    def read(self, path):
        path = Path(path)
        path = path if path.is_absolute() else self.root / path
        data = path.read_bytes()
        try:
            key = path.relative_to(self.root).as_posix()
        except ValueError:
            key = str(path.resolve())
        self.sources[key] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data.decode("utf-8-sig")

    def json(self, path):
        return json.loads(self.read(path))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.read(path))))


def observation(inputs, path):
    if path is None:
        return {"collection_status": "UNKNOWN", "reason": "No observation supplied; this does not mean no active run."}
    return {"collection_status": "SUPPLIED_SNAPSHOT", "source": str(path), "payload": inputs.json(path)}


def recompute_cost(inputs):
    """Reuse the canonical pure lookup path, never its artifact-writing main()."""
    if inputs.root.resolve() != ROOT.resolve():
        raise ValueError("Live cost lookup is only supported for the current repository")
    for folder in ("tools/fpga_h4", "tools/fpga_h3"):
        for path in sorted((ROOT / folder).glob("*.py")):
            inputs.read(path)
    for name in ("reports/fpga_h3_cost_db/H3_DATABASE_MANIFEST.json",
                 "artifacts/fpga_h3/20260918_batch01/clean.json",
                 "artifacts/fpga_h3/20260918_batch01/network_usage.json",
                 "artifacts/sonar_joint_cost_v1/clean.json",
                 "configs/search_space_sonar_v0_1.yaml",
                 "reports/phase2f_multiseed/network_candidate_set.json",
                 "reports/phase5a_quantization/PTQ_MANIFEST.json",
                 "reports/phase5a_quantization/CALIBRATION_MANIFEST.json",
                 "reports/phase5a_quantization/CHECKPOINT_SELECTION_MANIFEST.json",
                 "reports/phase5a_quantization/quant_config.yaml",
                 "reports/phase5e_mixed_precision/MIXED_PRECISION_RESULTS.csv",
                 "reports/phase5e_mixed_precision/PHASE5E_MANIFEST.json",
                 "reports/fpga_h2_contract/closure_manifest.json",
                 "reports/fpga_h2_contract/numeric_contract_v1.json",
                 "reports/fpga_h2_contract/interface_contract_v1.json"):
        inputs.read(name)
    usage = inputs.json("artifacts/fpga_h3/20260918_batch01/network_usage.json")
    for package in sorted(set(usage["packages"].values())):
        inputs.read(Path(package) / "graph.json")
    sys.path.insert(0, str(ROOT / "tools/fpga_h4"))
    try:
        from joint_search import run_joint_search
        report = run_joint_search()
    finally:
        sys.path.pop(0)
    return report


def build_audit(root=ROOT, runtime_audit=None, hardware_audit=None, verify_cost=False):
    inputs = Inputs(root)
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    populations = []
    histories = inputs.csv("reports/phase2e_multifidelity/standalone_learning_curves.csv")
    history_audit = {r["architecture_id"]: r for r in inputs.csv("reports/phase2e_multifidelity/standalone_history_audit.csv")}
    inputs.read("reports/phase2e_multifidelity/MULTIFIDELITY_PROTOCOL.yaml")
    history_by_arch = {}
    for row in histories:
        history_by_arch.setdefault(row["architecture_id"], []).append(row)

    def valid_budget_row(row, budget):
        if int(row["budget"]) != budget:
            return False
        if not truth(row["history_reuse"]):
            return int(row["checkpoint_epoch"]) == budget and truth(row["checkpoint_matches_budget"])
        # Original witnesses only retain epoch80 weights. Promotion may reuse
        # their prefix metrics, never their future scores or intermediate state.
        archived = history_audit.get(row["architecture_id"], {})
        if archived.get("history_sha256") != row["source_history_sha256"]:
            return False
        prefix = [r for r in history_by_arch.get(row["architecture_id"], []) if int(r["epoch"]) <= budget]
        if sorted(int(r["epoch"]) for r in prefix) != list(range(1, budget + 1)):
            return False
        best = min(prefix, key=lambda r: (-float(r["mAP50_95"]), int(r["epoch"])))
        endpoint = next(r for r in prefix if int(r["epoch"]) == budget)
        return (int(row["best_epoch"]) == int(best["epoch"])
                and math.isclose(float(row["mAP50_95"]), float(best["mAP50_95"]), rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(float(row["endpoint_mAP50_95"]), float(endpoint["mAP50_95"]), rel_tol=1e-12, abs_tol=1e-12))

    previous = None
    for filename, budget, expected in (("stage1_results.csv", 40, 144),
                                       ("stage2_results.csv", 60, 108),
                                       ("final_results.csv", 80, 72)):
        rows = inputs.csv("reports/phase2e_multifidelity/" + filename)
        identities = {r["architecture_id"] for r in rows}
        valid = (len(rows) == len(identities) == expected and all(valid_budget_row(r, budget) for r in rows))
        check("phase2e_budget_" + str(budget), valid, {"rows": len(rows), "expected": expected})
        if previous is not None:
            check("phase2e_promoted_subset_" + str(budget), identities <= previous)
        previous = identities
        populations.append({"budget": budget, "rows": len(rows), "unique_architectures": len(identities),
                            "history_prefix_reused": sum(truth(r["history_reuse"]) for r in rows)})

    multi = inputs.csv("reports/phase2f_multiseed/multiseed_results.csv")
    by_arch = {}
    for row in multi:
        by_arch.setdefault(row["architecture_id"], []).append(int(row["seed"]))
    check("phase2f_six_by_three", len(multi) == 18 and len(by_arch) == 6
          and all(sorted(v) == [42, 43, 44] for v in by_arch.values()))
    check("phase2f_test_not_evaluated", all(r.get("test_evaluated") == "False" for r in multi))

    summary = inputs.json("results/fpga_h4_joint_search/summary.json")
    q_source = inputs.json("reports/phase5e_mixed_precision/Q_OF_X.json")
    frozen_q = q_source["Q"]
    candidates = inputs.csv("results/fpga_h4_joint_search/FINAL_JOINT_CANDIDATES.csv")
    stored_front = inputs.csv("results/fpga_h4_joint_search/FINAL_JOINT_PARETO.csv")
    expected_count = sum(map(len, frozen_q.values())) * len(summary["mapping_policies"])
    identities_by_label = {r["label"]: r["architecture_id"] for r in multi}
    check("frozen_q_same_as_joint_summary", frozen_q == summary["frozen_q"])
    check("final_candidate_uniqueness", len({candidate_key(r) for r in candidates}) == len(candidates))
    expected_grid = {(identities_by_label[label], profile, policy)
                     for label, profiles in frozen_q.items() for profile in profiles
                     for policy in summary["mapping_policies"]}
    check("final_frozen_q_grid", {candidate_key(r) for r in candidates} == expected_grid,
          {"expected": expected_count, "actual": len(candidates)})
    eligible = [r for r in candidates if r["profile"] in frozen_q.get(r["final_network"], [])
                and identities_by_label.get(r["final_network"]) == r["architecture_id"]
                and r["quant_config"] == r["profile"]
                and r["hardware_status"] == "MEASURED" and r["accuracy_status"] == "MEASURED"
                and math.isfinite(float(r["Accuracy"])) and 0 <= float(r["Accuracy"]) <= 1
                and math.isfinite(float(r["serial_op_estimate_us"])) and float(r["serial_op_estimate_us"]) >= 0]
    check("final_rows_eligible", len(eligible) == len(candidates) == expected_count
          and all(r["q_status"] == "Q_ADMITTED" and truth(r["eligible"]) and truth(r["measurement_pair"])
                  for r in candidates))
    fronts = {"accuracy_vs_serial_op_estimate_us": pareto(eligible, ["serial_op_estimate_us"])}
    fronts.update({"accuracy_vs_" + name.lower(): pareto(eligible, [name]) for name in ("LUT", "FF", "DSP", "BRAM")})
    fronts["accuracy_vs_peak_operator_lut_ff_dsp_bram"] = pareto(eligible, ["LUT", "FF", "DSP", "BRAM"])
    front = fronts["accuracy_vs_serial_op_estimate_us"]
    check("pareto_membership", {candidate_key(r) for r in front} == {candidate_key(r) for r in stored_front})
    check("pareto_sizes_match", {k: len(v) for k, v in fronts.items()} == summary["pareto_front_sizes"])
    check("h4_scope_preserved", summary["measurement_scope"] == "OP_LEVEL_HLS"
          and summary["board_measured_latency"] is False and summary["routed_full_detector_resource"] is False
          and all(r["measurement_scope"] == "OP_LEVEL_HLS" and r["deployment"] == "NOT_CLAIMED" for r in candidates))
    coverage = inputs.json("results/sonar_joint_v1/coverage.json")
    coverage_rows = coverage["rows"]
    coverage_keys = {(r["architecture_id"], r["profile"]) for r in coverage_rows}
    arch_count = len({r[0] for r in coverage_keys})
    check("frozen_single_mapping_coverage", len(coverage_rows) == len(coverage_keys) == 432
          and arch_count == 144 and all(r["status"] == "LOOKED_UP" for r in coverage_rows))
    grid_count = arch_count * len(summary["profiles"]) * len(summary["mapping_policies"])
    check("full_hardware_grid_count", grid_count == summary["total_candidates"] == summary["hardware_measured"] == 2592)
    live_cost = {"status": "NOT_REQUERIED", "note": "2592 is bound to archived summary; 432 frozen rows independently counted."}
    if verify_cost:
        current = recompute_cost(inputs)
        measured = sum(r.get("hardware_status") == "MEASURED" for r in current["rows"])
        check("current_exact_hardware_lookup", measured == len(current["rows"]) == grid_count)
        check("current_exact_final_domain", {candidate_key(r) for r in current["eligible"]} == expected_grid)
        check("current_exact_pareto", {candidate_key(r) for r in current["pareto"]} == {candidate_key(r) for r in front})
        current_by_key = {candidate_key(r): r for r in current["eligible"]}
        check("current_exact_final_metrics", all(
            candidate_key(r) in current_by_key and all(
                math.isclose(float(r[k]), float(current_by_key[candidate_key(r)][k]), rel_tol=1e-12, abs_tol=1e-12)
                for k in ("Accuracy", "serial_op_estimate_us", "LUT", "FF", "DSP", "BRAM")) for r in candidates))
        live_cost = {"status": "RECOMPUTED_READ_ONLY", "hardware_rows": measured,
                     "final_eligible": len(current["eligible"]), "pareto": len(current["pareto"])}

    gate = inputs.json("reports/phase6a1_surrogate_diagnosis/PHASE6A1_ADMISSION.json")
    old_gap = inputs.read("reports/phase6a1_surrogate_diagnosis/FULL_ARCH_HW_COST_GAP.md")
    check("surrogate_denial_preserved", gate["PHASE6B_ALLOWED"] is False and gate["verdict"] == "SURROGATE_NOT_JUSTIFIED")
    p0 = inputs.csv("reports/sonar_operator_pool_p0/P0_SCREENING_RESULTS.csv")
    p0_manifest = inputs.json("reports/sonar_operator_pool_p0/P0_MANIFEST.json")
    p0_complete = [r for r in p0 if r["run_status"] == "COMPLETE" and int(r["epochs_completed"]) == 80]
    check("p0_screening_complete", len(p0_complete) == len(p0) == 20
          and len({(r["operator"], r["stage"]) for r in p0}) == 20
          and all(r["seed"] == "42" and r["test_evaluated"] == "False" for r in p0)
          and p0_manifest["screening_complete"] is True)
    baseline = next(r for r in multi if r["label"] == "D" and r["seed"] == "42")
    baseline_match = math.isclose(float(baseline["mAP50_95"]), p0_manifest["baseline_reference"]["mAP50_95"], abs_tol=1e-12)
    check("p0_reported_baseline_score_matches", baseline_match,
          "Score agreement only; this does not establish source/checkpoint execution parity.")
    structural = inputs.json("reports/phase3d_structural_decomposition/PHASE3D_MANIFEST.json")
    joint_freeze = inputs.json("results/sonar_joint_v1/freeze.json")
    inputs.read("configs/search/sonar_joint_v1.yaml")
    runtime = observation(inputs, runtime_audit)
    hardware = observation(inputs, hardware_audit)
    runtime_payload = runtime.get("payload", {})
    remote = runtime_payload.get("remote", {})
    p0_observation = remote.get("parallel_runs", {}).get("p0_confirmation", {})
    p0_status = p0_observation.get("status") or ("ACTIVE" if p0_observation.get("active") is True else "OBSERVATION_REQUIRED")
    unfinished = [
        {"item": "joint_search", "status": remote.get("run", {}).get("status", "UNKNOWN"),
         "depends_on": ["current frozen run completion", "12 candidate multi-seed confirmation", "quantization evaluation"],
         "parallel_boundary": "Use existing workers; do not infer idle GPUs or launch a duplicate run."},
        {"item": "p0_confirmation", "status": p0_status,
         "observation": p0_observation,
         "depends_on": ["seed43/44 paired evidence", "frozen baseline/source identity"],
         "parallel_boundary": "Observe existing confirmation jobs; single-seed screening is not admission."},
        {"item": "detector_deployment", "status": "PENDING_CANDIDATE_BOUND_EVIDENCE",
         "depends_on": ["final architecture and trained checkpoint", "quantized export and numerical parity",
                        "integrated executor and DMA", "full-network P&R", "board latency", "measured power"],
         "parallel_boundary": "Synthetic contract tests, memory/interface checks, and evidence tooling can proceed without final weights."},
        {"item": "feedback", "status": "SURROGATE_NOT_JUSTIFIED",
         "depends_on": ["valid measurements for constraints/mapping updates"],
         "parallel_boundary": "Use train/validation/hardware feedback; keep test outside selection and leave Phase6B denied."},
    ]
    return {"schema_version": 1, "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "repository artifact reconciliation, not a new research experiment",
            "read_only_inputs": True, "test_data_accessed": False, "training_started_by_audit": False,
            "overall_integrity": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
            "deployment_ready": False, "checks": checks,
            "historical_search": {"populations": populations, "multiseed_architectures": len(by_arch), "multiseed_rows": len(multi),
                                  "population_note": "Promoted finalists do not establish the global optimum over 144.",
                                  "history_reuse_note": "Original witnesses retain epoch80 weights only; their 40/60 prefix scores were recomputed from retained curves, not treated as resumable intermediate checkpoints."},
            "joint_domain": {"frozen_q": frozen_q, "recomputed_eligible": len(eligible), "recomputed_pareto_sizes": {k: len(v) for k, v in fronts.items()},
                             "hardware_coverage_grid": grid_count, "frozen_single_mapping_rows": len(coverage_rows),
                             "cost_lookup": live_cost, "measurement_scope": "OP_LEVEL_HLS",
                             "latency": "serial operator CSYNTH estimate", "resources": "operator peak proxy",
                             "full_space_joint_pareto": False},
            "sonarop": {"p0_screening_rows": len(p0_complete), "p0_scope": "seed42 only", "recommendations": p0_manifest["recommendations"],
                        "phase3d_status": structural["status"], "phase3d_p1_gate": structural["sonarop_p1_evidence_gate"]},
            "surrogate": {"verdict": gate["verdict"], "PHASE6B_ALLOWED": gate["PHASE6B_ALLOWED"],
                          "reason": "Hardware coverage updates do not repair failed accuracy predictors or uncertainty calibration."},
            "dated_evidence": [
                {"source": "reports/phase6a1_surrogate_diagnosis/FULL_ARCH_HW_COST_GAP.md", "status": "SUPERSEDED_FOR_COVERAGE_ONLY" if "3 / 144" in old_gap and grid_count == 2592 else "REVIEW",
                 "preserved": "The original replay remains compute-aware; its failed predictor admission remains valid."},
                {"source": "reports/phase5e_mixed_precision/Q_OF_X.json", "status": "SUPERSEDED_FOR_COST_STATUS_ONLY" if q_source.get("hardware_cost_status") == "NEEDS_MEASUREMENT" else "UNCHANGED",
                 "preserved": "Frozen Q(x) membership is unchanged; new evidence is operator-level only."},
                {"source": "results/sonar_joint_v1/FREEZE_AUDIT.md", "status": "FREEZE_TIME_SNAPSHOT_NOT_RUNTIME_STATUS",
                 "preserved": "training_started=false at freeze never establishes no current run."}],
            "test_boundary": {"joint_test_policy": joint_freeze.get("protocol", {}).get("test_policy", "UNKNOWN"),
                              "joint_policy_source": "configs/search/sonar_joint_v1.yaml", "p0_test_evaluated": p0_manifest["test_evaluated"],
                              "phase3d_test_accessed": structural["phase3d_test_accessed"],
                              "historical_external_transfer_tests": "Separate frozen protocols; no repository-wide assertion that all test data were never accessed.",
                              "feedback_sources": ["train", "validation", "hardware measurements"], "test_feedback_allowed": False},
            "runtime_observation": runtime, "hardware_observation": hardware, "unfinished": unfinished,
            "input_sha256": inputs.sources}


def render(report):
    joint = report["joint_domain"]
    failed = [c["name"] for c in report["checks"] if c["status"] == "FAIL"]
    lines = ["# 声呐协同设计当前证据核查", "", f"生成时间：{report['collected_at_utc']}", "",
             f"数据一致性：{report['overall_integrity']}；部署就绪：否。该审计不训练、不读取测试数据、不改历史产物。", "",
             "| 核查项 | 可复算结果 |", "|---|---|",
             "| 历史独立搜索 | " + " → ".join(str(p["rows"]) for p in report["historical_search"]["populations"]) + " |",
             f"| 历史多种子确认 | {report['historical_search']['multiseed_architectures']} 架构 / {report['historical_search']['multiseed_rows']} 条记录 |",
             f"| 冻结 Q(x) 联合域 | {joint['recomputed_eligible']} 候选；精度—延迟 Pareto {joint['recomputed_pareto_sizes']['accuracy_vs_serial_op_estimate_us']} 点 |",
             f"| 硬件代价覆盖 | {joint['hardware_coverage_grid']} 组合；{joint['cost_lookup']['status']} |",
             f"| SonarOp P0 | {report['sonarop']['p0_screening_rows']}/20，单种子筛查；确认情况见运行快照 |",
             "", "2592 仅指算子级 HLS 代价覆盖；30/4 仅指冻结 B1/C/D 联合域。延迟是算子串行估计，资源是算子峰值代理，不能用于宣称整网部署、真实延迟或功耗。", "",
             "## 历史状态与当前证据", ""]
    for entry in report["dated_evidence"]:
        lines.append(f"- `{entry['source']}`：`{entry['status']}`。{entry['preserved']}")
    lines += ["", "Phase6B 仍为 SURROGATE_NOT_JUSTIFIED。硬件覆盖补齐不改变精度预测器失败结论。", "",
              "## 待完成工作与并行边界", ""]
    for item in report["unfinished"]:
        lines.append(f"- **{item['item']}**：`{item['status']}`。依赖：{'; '.join(item['depends_on'])}。{item['parallel_boundary']}")
    lines += ["", "运行状态以显式传入的 runtime audit 快照为准，未传入为 UNKNOWN，不能推断没有运行。硬件状态以显式传入 hardware audit 的证据范围为准。", "",
              "反馈仅使用训练、验证和硬件测量；测试集不进入结构、量化或映射选择。外部迁移实验历史测试属于各自冻结协议，不能概括为全仓库 test 从未访问。", "",
              f"失败检查：{', '.join(failed) if failed else '无'}。输入文件 SHA256 和各项检查见 audit.json。", ""]
    return "\n".join(lines)


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(render(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path)
    parser.add_argument("--hardware-audit", type=Path)
    parser.add_argument("--recompute-cost", action="store_true", help="Re-query all 2592 canonical cost combinations without writing H4 artifacts.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve evidence")
    report = build_audit(runtime_audit=args.runtime_audit, hardware_audit=args.hardware_audit, verify_cost=args.recompute_cost)
    report["command"] = sys.argv
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_report(report, args.output)
    print(json.dumps({"overall_integrity": report["overall_integrity"], "output": str(args.output.resolve()), "checks": len(report["checks"])}, ensure_ascii=False))
    return 0 if report["overall_integrity"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
