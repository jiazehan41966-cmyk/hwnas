import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_zcu104_mobilenetA_formal_lut_preflight import (  # noqa: E402
    EXPECTED_DATA_KIND,
    EXPECTED_HARDWARE_LUT_VERSION,
    EXPECTED_PART,
    EXPECTED_SEARCH_SPACE_VERSION,
    audit,
    canonical_sha256,
)


def write_hashed_json(path: Path, payload: dict) -> None:
    payload = copy.deepcopy(payload)
    payload["payload_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def valid_row(index: int) -> dict:
    return {
        "row_key": f"row_{index:02d}",
        "operator": "pw_conv",
        "shape_name": f"shape_{index:02d}",
        "target_board": "ZCU104",
        "target_part": EXPECTED_PART,
        "target_clock_mhz": 200.0,
        "ready": True,
        "timing_met": index < 80,
        "evidence_source": "base_v1" if index < 75 else "recovery_v2",
        "cost": {
            "latency_cycles_hls": 1000 + index,
            "latency_ns_hls": 5000.0 + index,
            "latency_ms_hls": 0.005 + index / 1_000_000.0,
            "lut_post_route": 100 + index,
            "ff_post_route": 200 + index,
            "dsp_post_route": index % 8,
            "bram_tile_post_route": float(index % 4),
            "power_w_vivado_vectorless": 0.5,
        },
    }


def valid_lut_payload() -> dict:
    payload = {
        "artifact_type": "hardware_cost_lut",
        "data_kind": EXPECTED_DATA_KIND,
        "search_space_version": EXPECTED_SEARCH_SPACE_VERSION,
        "hardware_lut_version": EXPECTED_HARDWARE_LUT_VERSION,
        "search_ready": True,
        "status": "PASS_FORMAL_HLS_ROUTE_COVERAGE_88_OF_88",
        "claim_boundary": {
            "hls_route_cost": "FORMAL_IF_SEARCH_READY_TRUE",
            "board_classification": "NOT_MEASURED_IN_THIS_LUT",
            "external_power": "NOT_MEASURED_IN_THIS_LUT",
            "vivado_power": "VECTORLESS_ESTIMATE_ONLY",
        },
        "rows": [valid_row(index) for index in range(88)],
    }
    payload["rows"][0].update({
        "operator": "global_avg_pool", "shape_name": "gap_7_1280",
        "op_spec": {"in_channels": 1280, "out_channels": 1280, "input_resolution": [7, 7]},
    })
    return payload


def valid_coverage_payload(lut_payload_sha256: str) -> dict:
    return {
        "artifact_type": "operator_cost_lut_coverage_audit",
        "data_kind": EXPECTED_DATA_KIND,
        "search_space_version": EXPECTED_SEARCH_SPACE_VERSION,
        "hardware_lut_version": EXPECTED_HARDWARE_LUT_VERSION,
        "search_ready": True,
        "status": "PASS_FORMAL_HLS_ROUTE_COVERAGE_88_OF_88",
        "expected_rows": 88,
        "ready_rows": 88,
        "operator_cost_payload_sha256": lut_payload_sha256,
        "claim_boundary": {
            "hls_route_cost": "FORMAL_IF_SEARCH_READY_TRUE",
            "board_classification": "NOT_MEASURED_IN_THIS_LUT",
            "external_power": "NOT_MEASURED_IN_THIS_LUT",
            "vivado_power": "VECTORLESS_ESTIMATE_ONLY",
        },
    }


def write_valid_fixture(tmp_path: Path) -> tuple[Path, Path]:
    lut_path = tmp_path / "hardware_cost_lut.formal.json"
    coverage_path = tmp_path / "coverage_audit.formal.json"
    lut_payload = valid_lut_payload()
    write_hashed_json(lut_path, lut_payload)
    lut_with_hash = json.loads(lut_path.read_text(encoding="utf-8"))
    write_hashed_json(coverage_path, valid_coverage_payload(lut_with_hash["payload_sha256"]))
    return lut_path, coverage_path


def test_mobilenetA_formal_lut_preflight_passes_valid_fixture(tmp_path: Path) -> None:
    lut_path, coverage_path = write_valid_fixture(tmp_path)
    output_path = tmp_path / "preflight.formal.json"

    payload = audit(
        lut_path=lut_path,
        coverage_path=coverage_path,
        output_path=output_path,
    )

    assert payload["status"] == "PASS_FORMAL_ZCU104_OPERATOR_COST_LUT_PREFLIGHT"
    assert payload["search_ready"] is True
    assert payload["row_summary"]["rows"] == 88
    assert payload["row_summary"]["ready_rows"] == 88
    assert payload["row_summary"]["timing_met_rows"] == 80
    assert output_path.is_file()


def test_preflight_rejects_old_or_mislabeled_gap_even_when_hashes_pass(tmp_path: Path) -> None:
    lut_path, coverage_path = write_valid_fixture(tmp_path)
    lut = valid_lut_payload()
    lut["rows"][0]["op_spec"]["in_channels"] = 320
    write_hashed_json(lut_path, lut)
    saved = json.loads(lut_path.read_text())
    write_hashed_json(coverage_path, valid_coverage_payload(saved["payload_sha256"]))
    result = audit(lut_path=lut_path, coverage_path=coverage_path, output_path=None)
    assert result["search_ready"] is False
    assert any("gap_7_1280" in issue for issue in result["blockers"])


def test_mobilenetA_formal_lut_preflight_fails_closed_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    lut_path, coverage_path = write_valid_fixture(tmp_path)
    payload = json.loads(lut_path.read_text(encoding="utf-8"))
    payload["rows"][0]["ready"] = False
    lut_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        audit(lut_path=lut_path, coverage_path=coverage_path, output_path=None)
