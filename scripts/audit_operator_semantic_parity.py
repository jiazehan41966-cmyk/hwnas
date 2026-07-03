#!/usr/bin/env python3
"""Compare train-time sonar operators with the HLS operators used for costing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.models.builder import DenoiseBlock, EdgeAwareBlock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/first_principles_audit_20260703")
    return parser.parse_args()


def audit() -> dict:
    denoise = DenoiseBlock(32, 32, kernel_size=3)
    edge = EdgeAwareBlock(32, 32, kernel_size=3)
    denoise_hls = (
        REPO_ROOT
        / "hls_lut_builder"
        / "templates"
        / "denoise_serial_lowdsp_stage3_k3.cpp.tmpl"
    )
    edge_hls = (
        REPO_ROOT
        / "hls_lut_builder"
        / "templates"
        / "edge_serial_lowdsp_stage3_k3.cpp.tmpl"
    )
    denoise_text = denoise_hls.read_text(encoding="utf-8")
    edge_text = edge_hls.read_text(encoding="utf-8")

    rows = [
        {
            "operator": "denoise",
            "pytorch_spatial_branches": 2,
            "pytorch_branch_evidence": [
                type(denoise.dw_conv).__name__,
                "smooth_weight",
                "feat + smooth",
            ],
            "pytorch_residual": denoise.use_residual,
            "hls_spatial_branches": 1,
            "hls_branch_evidence": (
                "one spatial_weights[IN_CH][K][K] tensor"
                if "spatial_weights[IN_CH][K][K]" in denoise_text
                else "expected tensor not found"
            ),
            "hls_residual": "identity" in denoise_text,
            "semantic_parity": False,
            "cost_bias": "HLS omits the learned feature DW branch; resource and latency are optimistic.",
        },
        {
            "operator": "edge",
            "pytorch_spatial_branches": len(edge.edge_convs),
            "pytorch_fusion_input_channels": edge.fusion_conv.in_channels,
            "pytorch_residual": edge.use_residual,
            "hls_spatial_branches": 1,
            "hls_fusion_input_channels_expression": "IN_CH",
            "hls_branch_evidence": (
                "one spatial_weights[IN_CH][K][K] tensor"
                if "spatial_weights[IN_CH][K][K]" in edge_text
                else "expected tensor not found"
            ),
            "hls_absolute_value": "edge_value = -edge_value" in edge_text,
            "hls_residual": "identity" in edge_text,
            "semantic_parity": False,
            "cost_bias": "HLS implements one edge kernel instead of four directional branches and 4C fusion.",
        },
    ]
    return {
        "schema_version": 1,
        "overall_status": "failed",
        "rows": rows,
        "admission_decision": (
            "Block denoise and edge from claimable HW-NAS search until PyTorch-HLS "
            "numeric parity and matching weight export are tested."
        ),
        "evidence_boundary": (
            "Existing HLS/route/COM5 artifacts remain valid for the simplified HLS "
            "pipelines only; they do not establish deployment of the trained PyTorch operators."
        ),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Sonar Operator Semantic-Parity Audit",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Admission decision: {payload['admission_decision']}",
        "",
        "| Operator | PyTorch spatial branches | HLS spatial branches | Parity |",
        "|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['operator']} | {row['pytorch_spatial_branches']} | "
            f"{row['hls_spatial_branches']} | {row['semantic_parity']} |"
        )
    lines.extend(["", "## Evidence boundary", "", payload["evidence_boundary"], ""])
    return "\n".join(lines)


def main() -> int:
    payload = audit()
    output_dir = Path(parse_args().output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "operator_semantic_parity_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "operator_semantic_parity_audit.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
