"""Cross-module evidence boundaries for the current 72-architecture chain."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.modnas_comparison import joint_cost_backend as hardware
from tools.sonar_joint_v2 import policy
from tools.sonar_light_rl import run
from tools.sonar_light_rl.protocol import cost_status, default_protocol, provisional_handoff
from tools.sonar_light_deploy.handoff import prepare_handoff


def test_unmeasured_backend_to_rl_preserves_unknown(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(hardware.empty_manifest()), encoding="utf-8")
    backend = hardware.load_verified_joint_cost_backend(path, num_classes=2,
        profiles=policy.PROFILES, policies=policy.POLICIES, space_version="elastic_neck72_v1")
    requirements = run.requirements()
    assert len(requirements) == 1296
    rows = [backend.query(req["architecture"], req["profile"], req["mapping_policy"])
            for req in requirements]
    cfg = default_protocol()
    assert all(cost_status(row, req, cfg) == "NEEDS_MEASUREMENT"
               for row, req in zip(rows, requirements))
    evidence = run.ReplayEvidence(dict(schema_version="sonar_light_rl_replay_v1",
        costs=rows, evaluations=[], evidence_kind="source_verified_empty_registry"))
    report = run.run_replay(cfg, evidence, tmp_path / "replay")
    assert report["controller_updates"] == 0
    assert report["pareto_by_budget"] == {}
    assert report["training_started"] is False
    assert report["pending_handoffs"] == []
    events = json.loads((tmp_path / "replay/events.json").read_text(encoding="utf-8"))["events"]
    assert events and all(e["reward"] is None and not e["controller_updated"] for e in events)
    changed = dict(rows[0], architecture_hash="0" * 64)
    with pytest.raises(ValueError, match="bind"):
        cost_status(changed, requirements[0], cfg)


def test_rl_provisional_candidate_cannot_certify_deployment():
    req = run.requirements()[0]
    handoff = provisional_handoff(dict(req, seed=42, budget=80, evidence_kind="offline_replay"))
    # Even an added headline PASS cannot replace actual implementation evidence.
    handoff["board_parity"] = "PASS"
    report = prepare_handoff(handoff, ROOT)
    assert report["candidate"]["architecture_hash"] == req["architecture_hash"]
    assert report["status"] == "NOT_READY_FOR_DEPLOYMENT"
    assert report["formal_eligible"] is False
    assert report["launch_hardware"] is False
    assert report["stages"]["trained_checkpoint"]["status"] == "BLOCKED"
    assert report["stages"]["board_parity"]["status"] == "NOT_VERIFIED"
    assert any("UNSUPPORTED_HARDWARE_EVIDENCE" in item for item in report["blockers"])
