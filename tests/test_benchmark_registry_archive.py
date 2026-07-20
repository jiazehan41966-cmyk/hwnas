from __future__ import annotations

from pathlib import Path

import pytest

from hwnas_fpga.benchmarks.adapters import AdapterContext, PredictionRecord
from hwnas_fpga.benchmarks.archive import CampaignPaths, validate_prediction_record, write_table_bundle
from hwnas_fpga.benchmarks.registry import audit_source_checkout, load_paper_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_registry_is_pinned_and_unique() -> None:
    papers = load_paper_registry(REPO_ROOT / "configs/benchmarks/paper_registry_v1.yaml")
    assert len(papers) == 6
    assert len({paper.paper_id for paper in papers}) == 6
    assert sum(paper.registry_role == "main" for paper in papers) == 5
    assert sum(paper.registry_role == "supplementary" for paper in papers) == 1
    assert {paper.comparability_class for paper in papers} == {"A", "B", "C"}
    dmcl = next(paper for paper in papers if paper.paper_id == "dmcl_sonar_oltr_2025")
    assert dmcl.license_state == "missing"


def test_checked_out_source_pins_match_registry() -> None:
    papers = load_paper_registry(REPO_ROOT / "configs/benchmarks/paper_registry_v1.yaml")
    audits = [audit_source_checkout(paper, REPO_ROOT) for paper in papers]
    assert all(audit["checkout_exists"] for audit in audits)
    assert all(audit["commit_matches_pin"] for audit in audits)
    assert all(audit["remote_matches_registry"] for audit in audits)
    assert not any(audit["formal_eligible"] for audit in audits)


def test_prediction_schema_and_campaign_table_bundle(tmp_path: Path) -> None:
    digest = "a" * 64
    record = PredictionRecord(
        campaign_id="campaign",
        paper_id="paper",
        method="method",
        fold=0,
        seed=42,
        sample_id="sample.png",
        target=0,
        prediction=0,
        confidence=0.9,
        checkpoint_sha=digest,
        config_sha=digest,
        data_sha=digest,
        split_sha=digest,
        code_commit=None,
        code_state_sha=digest,
        claimability_status="PENDING",
    ).to_dict()
    validate_prediction_record(record)
    record["confidence"] = 1.1
    with pytest.raises(ValueError, match="confidence"):
        validate_prediction_record(record)

    paths = CampaignPaths.from_repo(tmp_path, "campaign").create()
    bundle = write_table_bundle(paths, "T1", [{"paper": "x", "status": "PENDING"}])
    assert all(Path(path).is_file() for path in bundle.values())


def test_adapter_context_rejects_missing_provenance() -> None:
    context = AdapterContext(
        campaign_id="c",
        paper_id="p",
        method="m",
        task="closed_set",
        fold=0,
        seed=42,
        data_dir="data",
        output_dir="output",
        config_sha="short",
        data_sha="a" * 64,
        split_sha="a" * 64,
        code_commit=None,
        code_state_sha="a" * 64,
    )
    with pytest.raises(ValueError, match="config_sha"):
        context.validate()
