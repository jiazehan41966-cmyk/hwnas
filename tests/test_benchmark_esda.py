from pathlib import Path

import pytest

from hwnas_fpga.benchmarks.esda import (
    ESDA_AUTHOR_PLATFORM,
    PROJECT_PLATFORM,
    build_esda_evidence_chain_manifest,
    validate_esda_numeric_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_CHECKOUT = ROOT / "reference/ESDA"


def test_esda_author_artifacts_preserve_class_c_boundary() -> None:
    if not AUTHOR_CHECKOUT.exists():
        pytest.skip("ESDA checkout not installed")
    manifest = build_esda_evidence_chain_manifest(AUTHOR_CHECKOUT)
    layers = {row["layer_id"]: row for row in manifest["evidence_layers"]}
    assert manifest["contract_valid"] is True
    assert manifest["comparability_class"] == "C"
    assert manifest["cross_platform_numeric_ranking_allowed"] is False
    assert manifest["project_formal_eligible"] is False
    assert layers["archived_bitstreams"]["artifact_kind"] == "AUTHOR_ARCHIVED_ZCU102_OUTPUT"
    assert layers["power_measurement"]["status"] == "PRESENT"


def test_author_power_archive_does_not_change_project_power_status() -> None:
    if not AUTHOR_CHECKOUT.exists():
        pytest.skip("ESDA checkout not installed")
    manifest = build_esda_evidence_chain_manifest(AUTHOR_CHECKOUT)
    assert manifest["project_gate_status"]["power"] == "NOT_MEASURED"
    assert "do not populate T7/T8" in manifest["boundary"]


def test_cross_board_numeric_ranking_is_rejected() -> None:
    with pytest.raises(ValueError, match="cross-platform numeric ranking is forbidden"):
        validate_esda_numeric_comparison(
            left_platform=ESDA_AUTHOR_PLATFORM,
            right_platform=PROJECT_PLATFORM,
            numeric_ranking=True,
        )


def test_same_board_or_nonranking_comparison_is_allowed() -> None:
    validate_esda_numeric_comparison(
        left_platform=PROJECT_PLATFORM,
        right_platform=PROJECT_PLATFORM,
        numeric_ranking=True,
    )
    validate_esda_numeric_comparison(
        left_platform=ESDA_AUTHOR_PLATFORM,
        right_platform=PROJECT_PLATFORM,
        numeric_ranking=False,
    )
