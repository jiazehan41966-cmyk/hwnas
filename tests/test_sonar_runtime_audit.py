"""Read-only runtime audit: fail-closed identity and process observations."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sonar_runtime_audit", ROOT / "tools/sonar_runtime_audit/run.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def frozen_tree(tmp_path):
    source = tmp_path / "tools/sonar_search/run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# original\n")
    out = tmp_path / audit.RUN_REL
    out.mkdir(parents=True)
    (out / "source_snapshot.zip").write_bytes(b"fixture archive")
    (out / "cost_snapshot.json").write_text("{}")
    audit.save_json(out / "snapshot.json", {"sha256": audit.sha(out / "source_snapshot.zip")})
    audit.save_json(out / "freeze.json", {
        "files": {"tools/sonar_search/run.py": audit.sha(source)},
        "cost_snapshot_sha256": audit.sha(out / "cost_snapshot.json"),
    })
    return source, out


def test_binding_detects_changed_missing_and_added_sources(tmp_path):
    source, out = frozen_tree(tmp_path)
    assert audit.inspect_binding(tmp_path)["status"] == "PASS"
    source.write_text("# changed\n")
    (source.parent / "new.py").write_text("# new\n")
    binding = audit.inspect_binding(tmp_path)
    assert binding["status"] == "DRIFT"
    assert binding["changed_files"] == ["tools/sonar_search/run.py"]
    assert binding["added_files"] == ["tools/sonar_search/new.py"]
    source.unlink()
    assert audit.inspect_binding(tmp_path)["missing_files"] == ["tools/sonar_search/run.py"]


def test_audit_tool_outside_frozen_directories_does_not_change_binding(tmp_path):
    frozen_tree(tmp_path)
    tool = tmp_path / "tools/sonar_runtime_audit/run.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# audit\n")
    assert audit.inspect_binding(tmp_path)["status"] == "PASS"


def test_archive_corruption_and_missing_state_are_not_passed(tmp_path):
    _, out = frozen_tree(tmp_path)
    (out / "source_snapshot.zip").write_bytes(b"changed")
    assert audit.inspect_binding(tmp_path)["status"] == "DRIFT"
    assert audit.summarize_run(tmp_path, audit.RUN_REL, [])["status"] == "NO_STATE"


def test_active_worker_epoch_requires_matching_candidate(tmp_path):
    _, out = frozen_tree(tmp_path)
    audit.save_json(out / "state.json", {"stage": "architecture", "budget": 40, "architecture_id": "candidate", "completed": 2, "total": 144})
    (out / "logs").mkdir()
    (out / "logs/train_candidate_42_40.log").write_text("candidate 42 8 0.1\nother 42 20 0.9\ncandidate 42 9 0.1\n")
    process = dict(pid=123, role="joint_worker", architecture_id="candidate", seed=42, budget=40, mode="train")
    result = audit.summarize_run(tmp_path, audit.RUN_REL, [process])
    assert result["active"] and result["status"] == "RUNNING"
    assert result["current_epoch"] == 9 and result["completed"] == 2


def test_other_user_commands_are_redacted_and_shell_not_counted():
    rows = audit.process_records(
        "123 someone /private/secret/python --token confidential\n"
        "124 taskuser bash -lc python tools/sonar_operator_pool_p0/confirmation_queue.py\n"
        "125 taskuser /venv/bin/python tools/sonar_operator_pool_p0/confirm.py train --operator DW5 --seed 43 --stage 7\n",
        "/isolated", "taskuser")
    assert "confidential" not in json.dumps(rows) and "secret" not in json.dumps(rows)
    assert rows[0]["owner"] == "other_user"
    assert rows[1]["role"] == "other_process"
    assert rows[2]["role"] == "p0_confirmation_worker" and rows[2]["seed"] == 43


def test_remote_payload_is_compilable_and_has_no_write_dispatch():
    payload = audit.remote_script("/isolated", audit.RUN_REL)
    compile(payload, "remote_runtime_audit", "exec")
    assert "module.verify(" in payload
    assert "module.execute(" not in payload and "write_text(" not in payload
