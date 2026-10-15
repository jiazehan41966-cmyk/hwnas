"""Rule continuation tests; no training or test split access."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase6b_rule_loop", ROOT / "tools/sonar_spec/phase6b_rule_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def fixture_inputs():
    rows = []
    entries = []
    for i in range(12):
        aid = f"arch{i:02d}"
        checksum = f"{i:064x}"
        rows.append(dict(architecture_id=aid, budget=60, seed=42, mAP50_95=.1 + i / 100,
                         checkpoint_sha256=checksum, Params=100, MACs=1000))
        entries.append(dict(architecture_id=aid, eligible_resume60to80=True,
            latest_epoch=60, history_epochs=list(range(1,61)), budget60_hash_matches=True,
            architecture_matches=True, source_binding_matches=True, optimizer_present=True,
            latest_sha256=checksum, mAP60_best=.1 + i / 100))
    return dict(source_protocol_binding="PASS", test_evaluated=False, remaining=entries), rows


def test_selection_fixed_top_four_two_exploration():
    preflight, rows = fixture_inputs()
    selected, eligible = loop.select_candidates(preflight, rows, [])
    assert eligible == 12
    assert [r["architecture_id"] for r in selected[:4]] == ["arch11", "arch10", "arch09", "arch08"]
    assert [r["architecture_id"] for r in selected[4:]] == ["arch01", "arch00"]
    assert len({r["architecture_id"] for r in selected}) == 6
    assert loop.select_candidates(preflight, list(reversed(rows)), [])[0] == selected


@pytest.mark.parametrize("change", ["completed", "witness", "hash", "epoch", "history", "binding", "test", "duplicate"])
def test_bad_parent_or_protocol_rejected(change):
    preflight, rows = fixture_inputs()
    row = preflight["remaining"][0]
    if change == "completed": row["budget80_exists"] = True
    elif change == "witness": row["is_existing_80_witness"] = True
    elif change == "hash": row["latest_sha256"] = "f" * 64
    elif change == "epoch": row["latest_epoch"] = 80
    elif change == "history": row["history_epochs"] = [1, 60]
    elif change == "binding": preflight["source_protocol_binding"] = "FAIL"
    elif change == "test": preflight["test_evaluated"] = True
    elif change == "duplicate": rows.append(dict(rows[0]))
    with pytest.raises(ValueError): loop.select_candidates(preflight, rows, [])


def test_final_population_cannot_be_rerun():
    preflight, rows = fixture_inputs()
    with pytest.raises(ValueError, match="cannot be rerun"):
        loop.select_candidates(preflight, rows, [dict(architecture_id="arch00")])


def test_ineligible_not_selected():
    preflight, rows = fixture_inputs()
    preflight["remaining"][-1]["eligible_resume60to80"] = False
    selected, n = loop.select_candidates(preflight, rows, [])
    assert n == 11
    assert "arch11" not in {r["architecture_id"] for r in selected}


def test_screening_both_metrics_and_no_automatic_follow_on():
    reference = dict(mAP50_95=.03, Recall=.4)
    rows = [dict(architecture_id=f"a{i}", epoch=80, seed=42, test_evaluated=False,
                 mAP50_95=.04, Recall=.3) for i in range(6)]
    assert loop.screening_decision(rows, reference)["status"] == "STOP_NO_SCREENING_CANDIDATE"
    rows[0]["Recall"] = .4
    result = loop.screening_decision(rows, reference)
    assert result["passing_architectures"] == ["a0"]
    assert result["automatic_next_batch"] is False
    assert result["automatic_hardware"] is False
    assert loop.screening_decision(rows[:5], reference)["status"] == "INCOMPLETE"


def test_screening_invalid_metrics():
    rows = [dict(architecture_id=f"a{i}", epoch=80, seed=42, test_evaluated=False,
                 mAP50_95=.04, Recall=.3) for i in range(6)]
    rows[0]["mAP50_95"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        loop.screening_decision(rows, dict(mAP50_95=.03, Recall=.4))


def test_metrics_include_raw_epoch_loss_and_stable_identity(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from hwnas_fpga.metrics_ledger import read_metric_records
    folder = tmp_path / "run"
    folder.mkdir()
    row = dict(epoch=61, valid=dict(mAP50_95=.03, AP50=.1, AP75=.02, Precision=.3, Recall=.4),
               losses=dict(loss=1.2, box=.1, objectness=.8, classification=.3), wall_seconds=10)
    loop.write(folder / "epoch_log.json", [row])
    selected = dict(architecture_id="fixture", parent_checkpoint_sha256="a"*64, Params=10, MACs=100)
    ledger = tmp_path / "METRICS.jsonl"
    loop.mirror_history(folder, selected, "b"*64, ledger, {})
    loop.mirror_history(folder, selected, "b"*64, ledger, {})
    records = read_metric_records(ledger)
    assert len(records) == 1
    assert records[0]["identity"]["raw_epoch_record"]["losses"] == row["losses"]
    assert records[0]["metrics"]["map50_95"]["value"] == .03
    assert records[0]["metrics"]["board_latency_ms"]["status"] == "PENDING"
    assert "excludes checkpoint saving and process startup" in records[0]["identity"]["gpu_time_scope"]


def test_supervisor_lock_rejects_second_owner_and_releases(tmp_path):
    path = tmp_path / "supervisor.lock"
    with loop.supervisor_lock(path):
        with pytest.raises(RuntimeError, match="SUPERVISOR_ALREADY_RUNNING"):
            with loop.supervisor_lock(path):
                pytest.fail("Second supervisor acquired the same lock")
    with loop.supervisor_lock(path):
        pass


def test_reference_archive_does_not_mislabel_best_as_endpoint():
    provenance = loop.reference_provenance()
    assert provenance["endpoint_weight_metric"] is False
    assert "epoch80 archive" in provenance["source_checkpoint_role"]
    assert "same best epoch" in provenance["primary_metric_role"]


@pytest.mark.parametrize("missing", ["python", "numpy", "torch", "cuda", "loader_generator", "sampler_generator"])
def test_checkpoint_rejects_missing_rng_component(missing):
    rng = dict(python="state", numpy="state", torch="state", cuda=["state"],
               loader_generator="state", sampler_generator="state")
    del rng[missing]
    state = dict(architecture_id="x", seed=42, epoch=1, binding={}, smoke=False,
                 test_evaluated=False, history=[dict(epoch=1)], optimizer_state_dict={}, rng=rng)
    with pytest.raises(ValueError, match="RNG must include"):
        loop.check_state(state, dict(architecture_id="x"), {}, 1)


def test_checkpoint_rejects_multiple_cuda_rng_devices():
    rng = dict(python="state", numpy="state", torch="state", cuda=["state", "state"],
               loader_generator="state", sampler_generator="state")
    state = dict(architecture_id="x", seed=42, epoch=1, binding={}, smoke=False,
                 test_evaluated=False, history=[dict(epoch=1)], optimizer_state_dict={}, rng=rng)
    with pytest.raises(ValueError, match="exactly one"):
        loop.check_state(state, dict(architecture_id="x"), {}, 1)


def test_real_preflight_six_without_any_completed80():
    preflight = loop.read(ROOT / "reports/phase6b_rule_loop/preflight.json")
    stage2 = loop.table(ROOT / "reports/phase2e_multifidelity/stage2_results.csv")
    final = loop.table(ROOT / "reports/phase2e_multifidelity/final_results.csv")
    selected, eligible = loop.select_candidates(preflight, stage2, final)
    assert eligible == 30
    assert len(selected) == 6
    assert not ({r["architecture_id"] for r in selected} & {r["architecture_id"] for r in final})
