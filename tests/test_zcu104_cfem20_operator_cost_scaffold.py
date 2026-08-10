import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_zcu104_cfem20_operator_cost_scaffold import build  # noqa: E402


FROZEN = (
    ROOT
    / "artifacts"
    / "sonar_fourstage_operator_v2"
    / "frozen_search_space_v5_cfem20"
)


def test_scaffold_has_9_fresh_rows_and_maps_all_20_candidates(tmp_path):
    result = build(FROZEN, tmp_path)

    scaffold = json.loads(
        (tmp_path / "hardware_cost_lut.scaffold.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (tmp_path / "coverage_audit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "build_manifest.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "PASS_SCAFFOLD_BUILT"
    assert len(scaffold["rows"]) == 9
    assert len({row["cost_key"] for row in scaffold["rows"]}) == 9
    assert all(row["historical_cost_row_allowed"] is False for row in scaffold["rows"])
    assert all(
        value is None
        for row in scaffold["rows"]
        for value in row["metrics"].values()
    )
    assert {row["operator_id"] for row in scaffold["rows"] if row["stage"] == "stage4"} == {
        "skip",
        "mbconv_k3_e3",
        "mbconv_k5_e3",
        "dmdc_conv_sonar",
        "cfem_sonar",
    }
    stage4 = {
        row["operator_id"]: row
        for row in scaffold["rows"]
        if row["stage"] == "stage4"
    }
    assert stage4["dmdc_conv_sonar"]["operator_parameters"]["dilation_rates"] == [
        1,
        3,
        5,
    ]
    assert stage4["cfem_sonar"]["operator_parameters"]["dilation_rates"] == [
        2,
        3,
        5,
    ]
    assert coverage["operator_key_definition_coverage"] == {
        "status": "PASS",
        "covered": 9,
        "expected": 9,
    }
    assert coverage["raw_candidate_key_mapping_coverage"]["covered"] == 20
    assert coverage["measured_operator_coverage"]["covered"] == 0
    assert coverage["search_ready"] is False
    assert manifest["status"] == "PASS_SCAFFOLD_BUILT"
    assert manifest["forbidden_actions_executed"] == []


def test_scaffold_rejects_tampered_frozen_payload(tmp_path):
    freeze_copy = tmp_path / "freeze"
    freeze_copy.mkdir()
    for source in FROZEN.glob("*.json"):
        (freeze_copy / source.name).write_bytes(source.read_bytes())
    audit_path = freeze_copy / "search_space_audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["raw_network_count"] = 19
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        build(freeze_copy, tmp_path / "output")
