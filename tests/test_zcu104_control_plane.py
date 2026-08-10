import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "artifacts" / "zcu104_full_campaign_v1"


def test_control_config_keeps_formal_60_checkpoint_registry_pending() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "zcu104_full_campaign_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    registry = config["formal_checkpoint_registry"]
    assert registry["expected_candidate_count"] == 4
    assert registry["folds"] == [0, 1, 2, 3, 4]
    assert registry["seeds"] == [42, 43, 44]
    assert registry["expected_checkpoint_count"] == 60
    assert registry["expected_outer_prediction_count"] == 60
    assert registry["representative_checkpoints_bound_in_this_config"] == 4
    assert registry["local_source_presence_preflight"]["status"] == "PASS_PRESENCE_ONLY"
    assert registry["fingerprint_registration"]["status"] == "PENDING"
    assert config["hardware"]["authoritative_toolchain_version"] == "2022.2"


def test_parallel_registry_has_three_non_overlapping_control_bound_lanes() -> None:
    path = CAMPAIGN_ROOT / "control" / "parallel_work_registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["campaign_id"] == "zcu104_full_campaign_v1"
    assert payload["status"] == "PARALLEL_READY"
    assert set(payload["lanes"]) == {"hardware", "runtime", "evidence"}

    expected = {
        "hardware": (
            "codex/zcu104-hardware-lane",
            r"E:\1\hwnas\hwnas-zcu104-hardware",
        ),
        "runtime": (
            "codex/zcu104-runtime-lane",
            r"E:\1\hwnas\hwnas-zcu104-runtime",
        ),
        "evidence": (
            "codex/zcu104-evidence-lane",
            r"E:\1\hwnas\hwnas-zcu104-evidence",
        ),
    }
    protected = set(payload["control_only_files"])
    assert protected == {
        "configs/experiments/zcu104_full_campaign_v1.yaml",
        "artifacts/zcu104_full_campaign_v1/status.json",
        "artifacts/zcu104_full_campaign_v1/campaign_ledger.json",
        "artifacts/zcu104_full_campaign_v1/control/parallel_work_registry.json",
    }

    for name, (branch, worktree) in expected.items():
        lane = payload["lanes"][name]
        assert lane["branch"] == branch
        assert lane["worktree"] == worktree
        assert lane["input_commit"] == payload["lane_base_commit"]
        assert protected.isdisjoint(set(lane["allowed_exact_files"]))
        assert protected.issubset(set(lane["forbidden_exact_files"]))
        assert lane["handoff_files"]
