"""Reject stale or corrupted summaries without touching training or test data."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sonar_workflow_audit", ROOT / "tools/sonar_workflow_audit/run.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def real_report():
    return audit.build_audit()


@pytest.fixture
def copied_inputs(tmp_path, real_report):
    for rel in real_report["input_sha256"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return tmp_path


def modify_csv(root, name, modify):
    path = root / name
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    modify(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def failed(report):
    return {c["name"] for c in report["checks"] if c["status"] == "FAIL"}


def test_current_evidence_and_absent_runtime_are_separate(real_report):
    assert real_report["overall_integrity"] == "PASS"
    assert real_report["joint_domain"]["recomputed_eligible"] == 30
    assert real_report["joint_domain"]["recomputed_pareto_sizes"]["accuracy_vs_serial_op_estimate_us"] == 4
    assert real_report["runtime_observation"]["collection_status"] == "UNKNOWN"
    assert real_report["unfinished"][0]["status"] == "UNKNOWN"
    assert not real_report["deployment_ready"]
    assert not real_report["surrogate"]["PHASE6B_ALLOWED"]
    assert not real_report["test_data_accessed"]


def test_duplicate_seed_cannot_masquerade_as_three_seed_confirmation(copied_inputs):
    modify_csv(copied_inputs, "reports/phase2f_multiseed/multiseed_results.csv", lambda rows: rows[1].update(seed=rows[0]["seed"]))
    result = audit.build_audit(copied_inputs)
    assert "phase2f_six_by_three" in failed(result)


def test_witness_future_score_cannot_enter_budget_promotion(copied_inputs):
    def mutate(rows):
        witness = next(r for r in rows if r["history_reuse"] == "True")
        witness["mAP50_95"] = "0.999"
        witness["best_epoch"] = "80"
    modify_csv(copied_inputs, "reports/phase2e_multifidelity/stage1_results.csv", mutate)
    assert "phase2e_budget_40" in failed(audit.build_audit(copied_inputs))


def test_measured_profile_cannot_bypass_frozen_q(copied_inputs):
    modify_csv(copied_inputs, "results/fpga_h4_joint_search/FINAL_JOINT_CANDIDATES.csv",
               lambda rows: rows[0].update(profile="HEAD_W16A16", quant_config="HEAD_W16A16"))
    result = audit.build_audit(copied_inputs)
    assert {"final_frozen_q_grid", "final_rows_eligible"} <= failed(result)


def test_stored_pareto_is_recomputed_not_trusted(copied_inputs):
    modify_csv(copied_inputs, "results/fpga_h4_joint_search/FINAL_JOINT_PARETO.csv", lambda rows: rows.pop())
    result = audit.build_audit(copied_inputs)
    assert "pareto_membership" in failed(result)


def test_operator_cost_cannot_be_promoted_to_board_evidence(copied_inputs):
    path = copied_inputs / "results/fpga_h4_joint_search/summary.json"
    data = json.loads(path.read_text())
    data["board_measured_latency"] = True
    path.write_text(json.dumps(data))
    assert "h4_scope_preserved" in failed(audit.build_audit(copied_inputs))


def test_runtime_snapshot_is_explicit_and_hashed(copied_inputs):
    path = copied_inputs / "runtime.json"
    path.write_text(json.dumps({"remote": {"run": {"status": "RUNNING", "active": True},
                                          "parallel_runs": {"p0_confirmation": {"active": True}}}}))
    result = audit.build_audit(copied_inputs, runtime_audit=path)
    assert result["unfinished"][0]["status"] == "RUNNING"
    assert result["unfinished"][1]["status"] == "ACTIVE"
    assert len(result["input_sha256"]["runtime.json"]["sha256"]) == 64


def test_pareto_handles_tradeoffs_and_equal_points():
    rows = [{"Accuracy": a, "cost": c, "id": i} for i, (a, c) in enumerate([(0.8, 10), (0.7, 8), (0.6, 9), (0.8, 10)])]
    assert [r["id"] for r in audit.pareto(rows, ["cost"])] == [0, 1, 3]
    with pytest.raises(ValueError, match="Non-finite"):
        audit.pareto([{"Accuracy": "nan", "cost": 1}], ["cost"])


def test_existing_output_is_never_overwritten(tmp_path, real_report):
    marker = tmp_path / "audit.json"
    marker.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        audit.write_report(real_report, tmp_path)
    assert marker.read_text() == "previous evidence"
