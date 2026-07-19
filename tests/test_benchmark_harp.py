from pathlib import Path

import pytest

from hwnas_fpga.benchmarks.harp import (
    HARP_GRAPH_TYPE,
    build_harp_program_graph_manifest,
    inspect_harp_gexf,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE = ROOT / "reference/HARP/dse_database/generated_graphs/machsuite/aes/aes.c"
AUTHOR_GRAPH = (
    ROOT
    / "reference/HARP/dse_database/generated_graphs/machsuite/processed"
    / HARP_GRAPH_TYPE
    / "aes_processed_result.gexf"
)


def test_author_hierarchical_graph_has_harp_attributes() -> None:
    if not AUTHOR_GRAPH.exists():
        pytest.skip("HARP checkout not installed")
    audit = inspect_harp_gexf(AUTHOR_GRAPH)
    assert audit["contract_valid"] is True
    assert audit["node_count"] > 0
    assert audit["edge_count"] > 0
    assert {"block", "function", "text", "type", "full_text"}.issubset(
        audit["node_attributes"]
    )


def test_harp_manifest_preserves_author_data_boundary() -> None:
    if not AUTHOR_GRAPH.exists():
        pytest.skip("HARP checkout not installed")
    manifest = build_harp_program_graph_manifest(
        sample_id="aes",
        hls_source_path=AUTHOR_SOURCE,
        gexf_path=AUTHOR_GRAPH,
        llvm_version="13 author provenance",
        source_kind="author_database",
    )
    assert manifest["contract_valid"] is True
    assert manifest["project_formal_eligible"] is False
    assert manifest["input_domain"] == "llvm_hls_program_graph_with_pragma_features"


def test_architecture_json_is_rejected_as_harp_program_source(tmp_path: Path) -> None:
    architecture = tmp_path / "candidate.json"
    architecture.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="architecture JSON/YAML is not accepted"):
        build_harp_program_graph_manifest(
            sample_id="invalid",
            hls_source_path=architecture,
            gexf_path=AUTHOR_GRAPH,
            llvm_version="13",
            source_kind="project_candidate",
        )
