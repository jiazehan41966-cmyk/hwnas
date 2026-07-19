"""HARP HLS-program graph contract helpers.

HARP consumes LLVM-derived program graphs and pragma/design-point features. A
NAS architecture graph is not interchangeable with this input domain, so the
adapter validates source-to-GEXF provenance before model integration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from hwnas_fpga.benchmarks.registry import sha256_file


HARP_PINNED_COMMIT = "c8bffd9411917b125846429b4d6be4f21c7a7165"
HARP_GRAPH_TYPE = "extended-pseudo-block-connected-hierarchy"
HARP_REQUIRED_NODE_ATTRIBUTES = {"block", "function", "text", "type", "full_text"}
HARP_REQUIRED_EDGE_ATTRIBUTES = {"flow", "position"}
HLS_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def inspect_harp_gexf(path: str | Path) -> dict[str, Any]:
    graph_path = Path(path).resolve()
    if graph_path.suffix.lower() != ".gexf":
        raise ValueError("HARP graph input must be a .gexf file")
    root = ET.parse(graph_path).getroot()
    namespace = root.tag.split("}", 1)[0] + "}" if root.tag.startswith("{") else ""
    graph = root.find(f"{namespace}graph")
    if graph is None:
        raise ValueError("GEXF does not contain a graph element")
    node_attributes: set[str] = set()
    edge_attributes: set[str] = set()
    for attributes in graph.findall(f"{namespace}attributes"):
        target = attributes.attrib.get("class")
        titles = {
            str(attribute.attrib.get("title"))
            for attribute in attributes.findall(f"{namespace}attribute")
        }
        if target == "node":
            node_attributes.update(titles)
        elif target == "edge":
            edge_attributes.update(titles)
    nodes = graph.find(f"{namespace}nodes")
    edges = graph.find(f"{namespace}edges")
    node_count = 0 if nodes is None else len(nodes.findall(f"{namespace}node"))
    edge_count = 0 if edges is None else len(edges.findall(f"{namespace}edge"))
    missing_node = sorted(HARP_REQUIRED_NODE_ATTRIBUTES - node_attributes)
    missing_edge = sorted(HARP_REQUIRED_EDGE_ATTRIBUTES - edge_attributes)
    graph_type_match = HARP_GRAPH_TYPE in graph_path.as_posix()
    blockers = []
    if node_count <= 0:
        blockers.append("empty_node_set")
    if edge_count <= 0:
        blockers.append("empty_edge_set")
    if missing_node:
        blockers.append("missing_required_node_attributes")
    if missing_edge:
        blockers.append("missing_required_edge_attributes")
    if not graph_type_match:
        blockers.append("not_harp_hierarchical_graph_type")
    return {
        "path": str(graph_path),
        "sha256": sha256_file(graph_path),
        "default_edge_type": graph.attrib.get("defaultedgetype"),
        "node_count": node_count,
        "edge_count": edge_count,
        "node_attributes": sorted(node_attributes),
        "edge_attributes": sorted(edge_attributes),
        "missing_required_node_attributes": missing_node,
        "missing_required_edge_attributes": missing_edge,
        "graph_type": HARP_GRAPH_TYPE if graph_type_match else "unknown",
        "contract_valid": not blockers,
        "blockers": blockers,
    }


def build_harp_program_graph_manifest(
    *,
    sample_id: str,
    hls_source_path: str | Path,
    gexf_path: str | Path,
    llvm_version: str,
    source_kind: str,
) -> dict[str, Any]:
    source = Path(hls_source_path).resolve()
    if source.suffix.lower() not in HLS_SOURCE_SUFFIXES:
        raise ValueError(
            "HARP requires generated HLS C/C++ source; architecture JSON/YAML is not accepted"
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    graph = inspect_harp_gexf(gexf_path)
    if source_kind not in {"author_database", "project_candidate"}:
        raise ValueError("source_kind must be author_database or project_candidate")
    blockers = list(graph["blockers"])
    if not str(llvm_version).startswith("13"):
        blockers.append("llvm13_provenance_not_verified")
    return {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "paper_id": "harp_2023",
        "sample_id": str(sample_id),
        "source_kind": source_kind,
        "input_domain": "llvm_hls_program_graph_with_pragma_features",
        "hls_source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "suffix": source.suffix.lower(),
        },
        "llvm_version": str(llvm_version),
        "graph": graph,
        "required_model_targets": [
            "perf",
            "util-LUT",
            "util-FF",
            "util-DSP",
            "util-BRAM",
        ],
        "contract_valid": not blockers,
        "blockers": blockers,
        "project_formal_eligible": False,
        "claimability_status": "NOT_CLAIMABLE",
        "boundary": (
            "A valid graph contract establishes source-to-graph shape only. It is not "
            "a project HLS/route measurement or a trained HARP prediction result."
        ),
    }
