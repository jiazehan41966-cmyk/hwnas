from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from hwnas_fpga.fourstage_operator import (
    RAW20_STAGE4_CHOICES,
    STAGE2_CHOICES,
    build_fourstage_architecture,
    enumerate_extended,
    enumerate_raw20,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import (
    FusedMBConvBlock,
    GhostBottleneckBlock,
    build_model,
)


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ARCHITECTURE_IDS = [
    "fourstage_s2_k3_e3_s4_skip",
    "fourstage_s2_k3_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e3_s4_fused_mbconv_e3",
    "fourstage_s2_k3_e3_s4_ghost_bottleneck",
    "fourstage_s2_k3_e6_s4_skip",
    "fourstage_s2_k3_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e6_s4_fused_mbconv_e3",
    "fourstage_s2_k3_e6_s4_ghost_bottleneck",
    "fourstage_s2_k5_e3_s4_skip",
    "fourstage_s2_k5_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e3_s4_fused_mbconv_e3",
    "fourstage_s2_k5_e3_s4_ghost_bottleneck",
    "fourstage_s2_k5_e6_s4_skip",
    "fourstage_s2_k5_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e6_s4_fused_mbconv_e3",
    "fourstage_s2_k5_e6_s4_ghost_bottleneck",
]


def test_raw20_search_space_is_fixed_unique_and_dir_free():
    rows = enumerate_raw20()
    assert len(STAGE2_CHOICES) == 4
    assert RAW20_STAGE4_CHOICES == (
        "skip",
        "mbconv_k3_e3",
        "mbconv_k5_e3",
        "fused_mbconv_e3",
        "ghost_bottleneck",
    )
    assert len(rows) == 20
    assert [row.arch_id for row in rows] == EXPECTED_ARCHITECTURE_IDS
    assert len({row.arch_id for row in rows}) == 20
    assert all("dir" not in row.arch_id.lower() for row in rows)
    for row in rows:
        validate_frozen_fourstage(row.architecture)


def test_raw20_adds_new_space_without_breaking_historical_dir_space():
    historical_rows = enumerate_extended(include_stage4_k5=True)
    assert len(historical_rows) == 16
    assert any("dir_mbconv3_split11_e3_v1" in row.arch_id for row in historical_rows)
    assert all(
        "dir_mbconv3_split11_e3_v1" not in row.arch_id
        for row in enumerate_raw20()
    )


def test_stage4_fused_and_ghost_build_and_preserve_network_shape():
    cases = [
        ("fused_mbconv_e3", FusedMBConvBlock),
        ("ghost_bottleneck", GhostBottleneckBlock),
    ]
    for stage4_op, expected_type in cases:
        architecture = build_fourstage_architecture(
            stage2_kernel=3,
            stage2_expansion=3,
            stage4_op=stage4_op,
        )
        model = build_model(architecture, num_classes=8).eval()
        assert isinstance(model.stages[3][0], expected_type)
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, 224, 224))
        assert logits.shape == (1, 8)


def test_raw20_freeze_script_writes_required_json_artifacts(tmp_path):
    output_dir = tmp_path / "raw20"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_fourstage_search_space_raw20.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_files = {
        "search_space_definition.json",
        "candidate_manifest_20.json",
        "operator_definition.json",
        "search_space_audit.json",
    }
    assert {path.name for path in output_dir.glob("*.json")} == expected_files
    definition = json.loads(
        (output_dir / "search_space_definition.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "candidate_manifest_20.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (output_dir / "search_space_audit.json").read_text(encoding="utf-8")
    )
    assert definition["stage2_candidate_count"] == 4
    assert definition["stage4_candidate_count"] == 5
    assert definition["candidate_count"] == 20
    assert manifest["candidate_count"] == 20
    assert [row["architecture_id"] for row in manifest["rows"]] == (
        EXPECTED_ARCHITECTURE_IDS
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["claim_boundary"]["hls"] == "NOT_RUN"
    assert audit["claim_boundary"]["route"] == "NOT_RUN"
    assert audit["claim_boundary"]["power"] == "NOT_MEASURED"
