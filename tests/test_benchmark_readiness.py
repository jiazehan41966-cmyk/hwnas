import json
from pathlib import Path

import pytest

from hwnas_fpga.benchmarks.readiness import build_readiness_report, make_requirement
from scripts.audit_benchmark_readiness import _tracked_patch_integrity


def test_readiness_is_fail_closed_until_every_requirement_passes() -> None:
    report = build_readiness_report(
        "campaign",
        [
            make_requirement("sources", "sources", passed=True, observed=5, required=5),
            make_requirement("formal", "formal", passed=False, observed=0, required=15),
        ],
    )
    assert report["status"] == "NOT_READY"
    assert report["formal_execution_ready"] is False
    assert report["blocking_requirement_ids"] == ["formal"]


def test_readiness_requires_nonempty_complete_requirement_set() -> None:
    empty = build_readiness_report("campaign", [])
    assert empty["status"] == "NOT_READY"
    complete = build_readiness_report(
        "campaign",
        [make_requirement("one", "one", passed=True, observed=True, required=True)],
    )
    assert complete["status"] == "READY"


def test_duplicate_requirement_ids_are_rejected() -> None:
    row = make_requirement("same", "same", passed=True, observed=True, required=True)
    with pytest.raises(ValueError, match="must be unique"):
        build_readiness_report("campaign", [row, row])


def test_readiness_patch_integrity_rejects_overwritten_patch(tmp_path: Path) -> None:
    patch = tmp_path / "code_patch_deadbeef.diff"
    patch.write_bytes(b"original patch\n")
    import hashlib

    expected = hashlib.sha256(patch.read_bytes()).hexdigest()
    summary = {
        "provenance": {
            "code": {
                "dirty": True,
                "tracked_patch": {"path": str(patch), "sha256": expected},
            }
        }
    }
    assert _tracked_patch_integrity(tmp_path, summary)["valid"] is True
    patch.write_bytes(b"overwritten patch\n")
    result = _tracked_patch_integrity(tmp_path, summary)
    assert result["valid"] is False
    assert result["expected_sha256"] != result["observed_sha256"]


def test_campaign_config_points_to_scratch_v2() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/benchmarks/ccf_ab_campaign_v1.yaml").read_text(encoding="utf-8")
    )
    methods = {
        row["method_id"]: row for row in config["classification"]["formal_methods"]
    }
    assert methods["scratch_mobilenet_v2"]["result_dir"].endswith(
        "g1_mobilenet_v2_scratch_v2"
    )
