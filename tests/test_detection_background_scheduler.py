"""Ensure the waiting controller never starts training on occupied cards."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.sonar_background_pilot import wait_idle


def setup_campaign(tmp_path, monkeypatch):
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "independent_doc_check.json").write_text('{"status":"PASS"}')
    (tmp_path / "cpu640_data_bindings.json").write_text('{"binding":"fixed"}')
    monkeypatch.setattr(sys, "argv", ["wait_idle", "--campaign", str(tmp_path), "--data-dir", "read_only_data"])


def test_occupied_cards_expire_without_preflight_or_training(tmp_path, monkeypatch):
    setup_campaign(tmp_path, monkeypatch)
    sys.argv += ["--max-wait-hours", "12"]
    ticks = iter([0, 0, 43201])
    monkeypatch.setattr(wait_idle, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
    commands = []
    monkeypatch.setattr(wait_idle, "subprocess", SimpleNamespace(
        check_output=lambda *_a, **_kw: "2048", run=lambda *a, **kw: commands.append(a), STDOUT=-2))
    wait_idle.main()
    status = json.loads((tmp_path / "scheduling_status.json").read_text())
    assert status["status"] == "WAIT_EXPIRED_NO_TRAINING" and not commands


def test_idle_launch_requires_cuda_data_binding_parity(tmp_path, monkeypatch):
    setup_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(wait_idle, "time", SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "preflight" in command:
            (tmp_path / "preflight_cuda.json").write_text(json.dumps({
                "status": "PASS", "data_bindings": {"binding": "DIFFERENT"}}))
        else:
            raise AssertionError("Training must not launch on a different dataset")
    monkeypatch.setattr(wait_idle, "subprocess", SimpleNamespace(check_output=lambda *_a, **_kw: "29", run=run, STDOUT=-2))
    import pytest
    with pytest.raises(RuntimeError, match="dataset parity"):
        wait_idle.main()
    assert len(commands) == 1 and "preflight" in commands[0]
    assert json.loads((tmp_path / "scheduling_status.json").read_text())["status"] == "FAILED"


def test_idle_controller_runs_preflight_frozen_queue_then_analysis(tmp_path, monkeypatch):
    setup_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(wait_idle, "time", SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if "preflight" in command:
            (tmp_path / "preflight_cuda.json").write_text(json.dumps({
                "status": "PASS", "data_bindings": {"binding": "fixed"},
                "variants": {name: {"auxiliary": {"raw": 1.0}} for name in ("rank", "fg_bg")}}))
        elif "queue" in command:
            (tmp_path / "queue_status.json").write_text('{"status":"COMPLETE"}')
    monkeypatch.setattr(wait_idle, "subprocess", SimpleNamespace(check_output=lambda *_a, **_kw: "29", run=run, STDOUT=-2))
    wait_idle.main()
    assert len(commands) == 3
    assert "preflight" in commands[0] and "queue" in commands[1]
    assert commands[1][-2:] == ["--epochs", "30"]
    assert json.loads((tmp_path / "scheduling_status.json").read_text())["status"] == "COMPLETE"
