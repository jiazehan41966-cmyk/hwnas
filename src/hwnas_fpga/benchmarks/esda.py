"""ESDA author-artifact evidence-chain audit.

ESDA is a class-C workflow reference for this campaign: its archived ZCU102
artifacts are useful for checking the expected software-to-board evidence
layers, but they are not AV7K325 measurements and must never be ranked against
project measurements as if the platforms and protocols were identical.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any, Iterable

from hwnas_fpga.benchmarks.registry import sha256_file


ESDA_PINNED_COMMIT = "b75c8c93ca258158c06a6434f5f0f084add02ee5"
ESDA_AUTHOR_PLATFORM = "ZCU102"
PROJECT_PLATFORM = "AV7K325"


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _existing(checkout: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in checkout.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(checkout).as_posix())


def _layer(
    checkout: Path,
    *,
    layer_id: str,
    description: str,
    artifact_kind: str,
    patterns: list[str],
    representative_limit: int = 5,
) -> dict[str, Any]:
    files = _existing(checkout, patterns)
    representatives = []
    for path in files[:representative_limit]:
        representatives.append(
            {
                "path": path.relative_to(checkout).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "layer_id": layer_id,
        "description": description,
        "status": "PRESENT" if files else "ABSENT",
        "artifact_kind": artifact_kind,
        "file_count": len(files),
        "patterns": patterns,
        "representative_files": representatives,
    }


def validate_esda_numeric_comparison(
    *, left_platform: str, right_platform: str, numeric_ranking: bool
) -> None:
    """Reject direct numerical ranking when the FPGA platforms differ."""

    left = str(left_platform).strip().casefold()
    right = str(right_platform).strip().casefold()
    if not left or not right:
        raise ValueError("both comparison platforms must be explicit")
    if numeric_ranking and left != right:
        raise ValueError(
            "cross-platform numeric ranking is forbidden for the ESDA class-C "
            "reference; rerun both methods on one board and one measurement protocol"
        )


def build_esda_evidence_chain_manifest(checkout_path: str | Path) -> dict[str, Any]:
    checkout = Path(checkout_path).resolve()
    if not checkout.is_dir():
        raise FileNotFoundError(checkout)

    commit = _git_output(checkout, "rev-parse", "HEAD")
    dirty_lines = [
        line for line in _git_output(checkout, "status", "--porcelain").splitlines() if line
    ]
    layers = [
        _layer(
            checkout,
            layer_id="software_training",
            description="Float/quantized model training and search entrypoints",
            artifact_kind="AUTHOR_SOURCE_CODE",
            patterns=["software/main.py", "software/search_sw.py", "software/config/*.py"],
        ),
        _layer(
            checkout,
            layer_id="quantized_inference_export",
            description="INT8 inference/export and quantization implementation",
            artifact_kind="AUTHOR_SOURCE_CODE",
            patterns=[
                "software/int_inference.py",
                "software/src/quantization.cpp",
                "software/models/HAWQ_quant_module/*.py",
            ],
        ),
        _layer(
            checkout,
            layer_id="hls_source_generation",
            description="Configuration-to-HLS source generation templates",
            artifact_kind="AUTHOR_SOURCE_CODE",
            patterns=[
                "hardware/gen_prj.py",
                "hardware/template_e2e/gen_code.py",
                "hardware/template_e2e/top.cpp.tpl",
                "hardware/template_e2e_roshambo/gen_code.py",
                "hardware/template_e2e_roshambo/top.cpp.tpl",
            ],
        ),
        _layer(
            checkout,
            layer_id="hls_vivado_toolflow",
            description="Vitis HLS and Vivado project/synthesis scripts",
            artifact_kind="AUTHOR_SOURCE_CODE",
            patterns=["hardware/**/hls.tcl", "hardware/**/vivado.tcl", "hardware/**/Makefile"],
        ),
        _layer(
            checkout,
            layer_id="archived_bitstreams",
            description="Author-generated ZCU102 bitstreams",
            artifact_kind="AUTHOR_ARCHIVED_ZCU102_OUTPUT",
            patterns=["eventNet/hw/**/top.bit"],
        ),
        _layer(
            checkout,
            layer_id="board_runtime",
            description="PYNQ overlay loading, inference, and latency runtime",
            artifact_kind="AUTHOR_ZCU102_SOURCE_CODE",
            patterns=["hardware/board/evaluate.py", "hardware/board/hw_e2e.py"],
        ),
        _layer(
            checkout,
            layer_id="power_measurement",
            description="ZCU102 INA226/PMBus acquisition code and archived raw arrays",
            artifact_kind="AUTHOR_ZCU102_SOURCE_AND_ARCHIVED_OUTPUT",
            patterns=["hardware/board/power_monitor.py", "eventNet/hw/**/power_record.npy"],
        ),
    ]
    required_layers = {
        "software_training",
        "quantized_inference_export",
        "hls_source_generation",
        "hls_vivado_toolflow",
        "board_runtime",
    }
    missing_required = [
        row["layer_id"]
        for row in layers
        if row["layer_id"] in required_layers and row["status"] != "PRESENT"
    ]
    blockers = []
    if commit != ESDA_PINNED_COMMIT:
        blockers.append("pinned_commit_mismatch")
    if dirty_lines:
        blockers.append("author_checkout_dirty")
    if missing_required:
        blockers.append("required_evidence_layers_missing")

    return {
        "schema_version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "paper_id": "esda_2024",
        "comparability_class": "C",
        "checkout": {
            "path": str(checkout),
            "expected_commit": ESDA_PINNED_COMMIT,
            "observed_commit": commit,
            "commit_match": commit == ESDA_PINNED_COMMIT,
            "dirty": bool(dirty_lines),
            "dirty_entries": dirty_lines,
            "license": "MIT",
            "license_file_sha256": sha256_file(checkout / "LICENSE"),
        },
        "author_platform": ESDA_AUTHOR_PLATFORM,
        "project_platform": PROJECT_PLATFORM,
        "evidence_layers": layers,
        "missing_required_layers": missing_required,
        "contract_valid": not blockers,
        "blockers": blockers,
        "cross_platform_numeric_ranking_allowed": False,
        "project_formal_eligible": False,
        "project_gate_status": {
            "bitstream": "NOT_ESTABLISHED_BY_ESDA_ARTIFACTS",
            "board_measurement": "NOT_ESTABLISHED_BY_ESDA_ARTIFACTS",
            "power": "NOT_MEASURED",
        },
        "claimability_status": "NOT_CLAIMABLE",
        "boundary": (
            "This contract verifies the organization and presence of author ZCU102 "
            "artifacts only. Archived bitstreams and power arrays are not AV7K325 "
            "measurements, do not populate T7/T8, and cannot support cross-board ranking."
        ),
    }
