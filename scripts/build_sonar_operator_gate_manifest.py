"""Assemble the G5 sonar-operator gate manifest from measured fragments.

The script is intentionally conservative: missing fragments leave the
corresponding G5 sub-gates false/zero/TODO so the admission audit remains
blocked until real evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


OPERATORS = ("denoise", "edge")
INPUT_KINDS = ("real_sample", "boundary_tensor", "random_tensor")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_path(raw: Any, *, base_dir: Path | None = None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    path = path.resolve()
    return path if path.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="artifacts/sonar_operator_gate/manifest.g5_v1_pre_e1.json",
        help="Base/pre-E1 manifest with matched-control evidence.",
    )
    parser.add_argument(
        "--bootstrap",
        default="artifacts/sonar_operator_gate/g5_ablation_bootstrap_comparisons.json",
        help="Manifest fragment from compare_sonar_ablation_bootstrap.py.",
    )
    parser.add_argument(
        "--denoise-export",
        default="artifacts/sonar_operator_gate/folded_exports/denoise_fold1_seed42/folded_export_manifest.json",
    )
    parser.add_argument(
        "--edge-export",
        default="artifacts/sonar_operator_gate/folded_exports/edge_fold1_seed42/folded_export_manifest.json",
    )
    parser.add_argument("--denoise-parity-records", default=None)
    parser.add_argument("--edge-parity-records", default=None)
    parser.add_argument("--denoise-hls-evidence", default=None)
    parser.add_argument("--edge-hls-evidence", default=None)
    parser.add_argument(
        "--output",
        default="artifacts/sonar_operator_gate/manifest.g5_v1.json",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def optional_json(path: str | Path | None, missing: list[str]) -> dict[str, Any] | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        missing.append(str(source))
        return None
    return load_json(source)


def merge_bootstrap(manifest: dict[str, Any], bootstrap: Mapping[str, Any]) -> None:
    variants = bootstrap.get("ablation_variants")
    if isinstance(variants, Mapping):
        manifest["ablation_variants"] = dict(variants)
    comparisons = bootstrap.get("comparisons_vs_control")
    if isinstance(comparisons, Mapping):
        manifest["comparisons_vs_control"] = dict(comparisons)


def merge_export(
    manifest: dict[str, Any],
    operator: str,
    export: Mapping[str, Any],
) -> None:
    row = manifest.setdefault("operators", {}).setdefault(operator, {})
    # The software export must never manufacture an HLS-consumption hash.
    row.pop("hls_spec_sha256", None)
    for key in (
        "quantization_contract",
        "software_spec_sha256",
        "weight_export_complete",
        "weight_export_sha256",
    ):
        if key in export:
            row[key] = export[key]
    row["folded_export"] = {
        "manifest_path": export.get("manifest_path"),
        "output_dir": export.get("output_dir"),
        "source_checkpoint": export.get("source_checkpoint"),
        "source_checkpoint_sha256": export.get("source_checkpoint_sha256"),
        "folded_sonar_block_count": export.get("folded_sonar_block_count"),
    }


def read_parity_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


def merge_parity_records(
    manifest: dict[str, Any],
    operator: str,
    records_path: str | Path,
) -> None:
    rows = read_parity_records(records_path)
    kinds = Counter(str(row.get("input_kind", "")) for row in rows)
    compared = sum(int(row.get("element_count", 0)) for row in rows)
    mismatches = sum(int(row.get("mismatch_count", -1)) for row in rows)
    spec_hashes = sorted(
        {
            str(row.get("quantization_spec_sha256", ""))
            for row in rows
            if row.get("quantization_spec_sha256")
        }
    )
    manifest["operators"][operator]["parity"] = {
        "real_sample_count": kinds["real_sample"],
        "boundary_tensor_count": kinds["boundary_tensor"],
        "random_tensor_count": kinds["random_tensor"],
        "compared_element_count": compared,
        "mismatch_count": mismatches,
        "records_path": str(Path(records_path).resolve()),
        "record_count": len(rows),
        "input_kinds": [kind for kind in INPUT_KINDS if kinds[kind] > 0],
        "quantization_spec_sha256s": spec_hashes,
        "per_layer_trace_complete": all(
            row.get("layer")
            and int(row.get("element_count", 0)) > 0
            and len(str(row.get("simulator_sha256", ""))) == 64
            and len(str(row.get("quantization_spec_sha256", ""))) == 64
            for row in rows
        ),
    }


def merge_hls_evidence(
    manifest: dict[str, Any],
    operator: str,
    evidence: Mapping[str, Any],
) -> None:
    hls = manifest["operators"][operator].setdefault("hls", {})
    hls["evidence_complete"] = bool(
        evidence.get("evidence_complete", evidence.get("hls_evidence_complete", False))
    )
    hls["route_feasible"] = bool(
        evidence.get(
            "route_feasible",
            evidence.get("route_pass", evidence.get("timing_met", False)),
        )
    )
    hls["evidence_path"] = evidence.get("evidence_path")
    hls["tool"] = evidence.get("tool")
    hls["tool_version"] = evidence.get("tool_version")
    hls["consumed_spec_path"] = evidence.get("consumed_spec_path")
    hls["consumed_spec_sha256"] = evidence.get("consumed_spec_sha256")
    hls["declared_evidence_sha256"] = evidence.get("evidence_sha256")
    hls["spec_consumed"] = evidence.get("spec_consumed") is True
    hls["boundary"] = evidence.get("boundary")

    source_path = Path(str(evidence.get("_source_path", ""))).resolve()
    base_dir = source_path.parent if source_path.name else None
    consumed_spec = existing_path(hls["consumed_spec_path"], base_dir=base_dir)
    evidence_file = existing_path(hls["evidence_path"], base_dir=base_dir)
    hls["consumed_spec_actual_sha256"] = (
        sha256_file(consumed_spec) if consumed_spec is not None else None
    )
    hls["evidence_sha256"] = (
        sha256_file(evidence_file) if evidence_file is not None else None
    )
    hls["evidence_file_exists"] = evidence_file is not None
    hls["consumed_spec_file_exists"] = consumed_spec is not None


def assemble_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.base)
    missing: list[str] = []
    fragments: dict[str, str] = {"base": str(Path(args.base).resolve())}

    bootstrap = optional_json(args.bootstrap, missing)
    if bootstrap is not None:
        merge_bootstrap(manifest, bootstrap)
        fragments["bootstrap"] = str(Path(args.bootstrap).resolve())

    for operator in OPERATORS:
        export_path = getattr(args, f"{operator}_export")
        export = optional_json(export_path, missing)
        if export is not None:
            merge_export(manifest, operator, export)
            fragments[f"{operator}_export"] = str(Path(export_path).resolve())

        records_path = getattr(args, f"{operator}_parity_records")
        if records_path:
            if Path(records_path).exists():
                merge_parity_records(manifest, operator, records_path)
                fragments[f"{operator}_parity_records"] = str(Path(records_path).resolve())
            else:
                missing.append(str(Path(records_path)))

        hls_path = getattr(args, f"{operator}_hls_evidence")
        hls = optional_json(hls_path, missing)
        if hls is not None:
            hls["_source_path"] = str(Path(hls_path).resolve())
            merge_hls_evidence(manifest, operator, hls)
            fragments[f"{operator}_hls_evidence"] = str(Path(hls_path).resolve())

    manifest["generated_by"] = "scripts/build_sonar_operator_gate_manifest.py"
    manifest["schema_version"] = 2
    manifest["generated"] = datetime.now().isoformat(timespec="seconds")
    manifest["source_fragments"] = fragments
    manifest["missing_fragments"] = missing
    manifest["boundary"] = (
        "Assembled from measured fragments only. Missing parity/HLS/route evidence "
        "keeps the corresponding G5 gates blocked."
    )
    return manifest


def main() -> int:
    args = parse_args()
    manifest = assemble_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "missing_fragments": manifest["missing_fragments"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
