from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "scripts"
    / "rl_best_e2e_board_validation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("rl_best_e2e_board_validation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(tmp_path: Path, candidate_path: Path, *, candidate_sha: str | None = None) -> Path:
    csynth = tmp_path / "role_csynth.xml"
    csynth.write_text("<report />", encoding="utf-8")
    report = {
        "candidate_sha256": candidate_sha or sha256(candidate_path),
        "evidence_complete": True,
        "roles": [
            {
                "role": "stage0_block0",
                "logical_op": "mbconv",
                "evidence_status": "measured_csynth",
                "case_name": "candidate_case_0001_shape_0001_baseline_pi1_po1_u1_main_5ns",
                "report_path": str(csynth),
                "report_sha256": sha256(csynth),
            },
            {
                "role": "stage1_block0",
                "logical_op": "skip",
                "evidence_status": "structural_skip",
            },
        ],
    }
    path = tmp_path / "candidate_hls_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def trace_rows() -> list[dict[str, object]]:
    return [
        {"layer_index": 1, "layer_role": "stage0_block0", "op": "mbconv", "notes": ""},
        {"layer_index": 2, "layer_role": "stage1_block0", "op": "skip", "notes": ""},
    ]


def test_candidate_hls_report_maps_direct_and_structural_skip(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"candidate": {}}', encoding="utf-8")
    report = write_report(tmp_path, candidate)
    rows = trace_rows()

    result = module.apply_candidate_hls_report_mapping(
        rows,
        report_path=report,
        candidate_path=candidate,
    )

    assert result["status"] == "PASS"
    assert result["mapped_component_layers"] == 1
    assert result["structural_skip_layers"] == 1
    assert rows[0]["current84_mapping_status"] == "mapped_to_candidate_hls_cache"
    assert rows[0]["component_roles"] == "direct"
    assert rows[1]["current84_mapping_status"] == "structural_skip"
    assert rows[1]["component_case_names"] == ""


def test_candidate_hls_report_rejects_candidate_hash_mismatch(tmp_path: Path) -> None:
    module = load_module()
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"candidate": {}}', encoding="utf-8")
    report = write_report(tmp_path, candidate, candidate_sha="0" * 64)

    with pytest.raises(ValueError, match="hash does not match"):
        module.apply_candidate_hls_report_mapping(
            trace_rows(),
            report_path=report,
            candidate_path=candidate,
        )


def test_resolver_accepts_candidate_hls_cache_syn_rtl(tmp_path: Path) -> None:
    module = load_module()
    case_dir = tmp_path / "candidate_case_0001"
    (case_dir / "project" / "solution1" / "syn" / "verilog").mkdir(parents=True)

    assert module.resolve_rl_best_case_dir(
        "candidate_case_0001",
        candidate_hls_cache_root=tmp_path,
    ) == case_dir.resolve()


def test_resolver_falls_back_to_default_cache_for_unmodified_case(tmp_path: Path) -> None:
    module = load_module()
    overlay_root = tmp_path / "overlay"
    default_root = tmp_path / "default"
    module.CANDIDATE_HLS_CACHE_ROOT = default_root
    case_dir = default_root / "candidate_case_0002"
    (case_dir / "project" / "solution1" / "impl" / "verilog").mkdir(parents=True)

    assert module.resolve_rl_best_case_dir(
        "candidate_case_0002",
        candidate_hls_cache_root=overlay_root,
    ) == case_dir.resolve()


@pytest.mark.parametrize(
    ("layer_role", "expected"),
    [
        ("stem", "stem_conv_k3_s2"),
        ("head_conv", "pw_conv"),
        ("fc_classifier", "fc_layer"),
        ("stage0_block0", "candidate_case_0001"),
    ],
)
def test_candidate_cache_uses_operator_formula_key_without_changing_case_identity(
    layer_role: str,
    expected: str,
) -> None:
    module = load_module()
    row = {
        "case_name": "candidate_case_0001",
        "layer_role": layer_role,
        "namespace": "candidate_hls_cache",
    }

    assert module.component_metadata_case_name(row) == expected
    assert row["case_name"] == "candidate_case_0001"


def test_non_candidate_namespace_keeps_original_metadata_case_name() -> None:
    module = load_module()
    row = {
        "case_name": "fc_custom_case",
        "layer_role": "fc_classifier",
        "namespace": "current84_fullcombo",
    }

    assert module.component_metadata_case_name(row) == "fc_custom_case"
