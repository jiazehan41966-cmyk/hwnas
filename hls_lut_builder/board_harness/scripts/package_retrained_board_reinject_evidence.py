#!/usr/bin/env python3
"""Package retrained-weight board reinjection evidence into a compact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path, copied: list[dict[str, Any]], *, required: bool = True) -> None:
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(
        {
            "source": str(src.resolve()),
            "bundle_path": dst.as_posix(),
            "sha256": sha256_file(dst),
            "bytes": dst.stat().st_size,
        }
    )


def run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("rl_arch_*_trainedmem") if path.is_dir())


def bitstream_info(run_root: Path) -> dict[str, str]:
    bitstreams = sorted((run_root / "harness_project" / "bitstream").glob("*.bit"))
    if not bitstreams:
        return {"path": "", "sha256": ""}
    bitstream = bitstreams[0]
    return {"path": str(bitstream.resolve()), "sha256": sha256_file(bitstream)}


def candidate_artifacts_for_run(run_root: Path) -> list[tuple[Path, str, bool]]:
    run_name = run_root.name
    return [
        (run_root / "build_result.json", f"{run_name}/build_result.json", True),
        (run_root / "measurement_preflight.json", f"{run_name}/measurement_preflight.json", False),
        (
            run_root / "harness_project" / "harness_manifest.json",
            f"{run_name}/harness_manifest.json",
            True,
        ),
        (
            run_root / "harness_project" / "trained_weight_manifest.json",
            f"{run_name}/trained_weight_manifest.json",
            True,
        ),
        (
            run_root / "harness_project" / "software_board_parity.json",
            f"{run_name}/software_board_parity.json",
            True,
        ),
        (
            run_root / "measurements" / "stability_runs.json",
            f"{run_name}/measurements/stability_runs.json",
            True,
        ),
        (
            run_root / "measurements" / "measurements_raw.csv",
            f"{run_name}/measurements/measurements_raw.csv",
            False,
        ),
        (
            run_root / "measurements" / "measurements_raw_runs.csv",
            f"{run_name}/measurements/measurements_raw_runs.csv",
            False,
        ),
        (
            run_root
            / "measurements"
            / f"sonar_classifier_rl_best_e2e__{run_name}.measurement.json",
            f"{run_name}/measurements/{run_name}.measurement.json",
            True,
        ),
        (run_root / "logs" / "measurement.out.log", f"{run_name}/logs/measurement.out.log", False),
        (run_root / "logs" / "vivado_bitstream.out.log", f"{run_name}/logs/vivado_bitstream.out.log", False),
        (run_root / "logs" / "vivado_bitstream.err.log", f"{run_name}/logs/vivado_bitstream.err.log", False),
        (run_root / "vivado_reports" / "timing_summary.rpt", f"{run_name}/vivado_reports/timing_summary.rpt", True),
        (run_root / "vivado_reports" / "utilization.rpt", f"{run_name}/vivado_reports/utilization.rpt", True),
        (
            run_root / "vivado_reports" / "harness_top_route_status_routed.rpt",
            f"{run_name}/vivado_reports/harness_top_route_status_routed.rpt",
            False,
        ),
        (
            run_root / "vivado_reports" / "harness_top_timing_summary_routed.rpt",
            f"{run_name}/vivado_reports/harness_top_timing_summary_routed.rpt",
            False,
        ),
        (
            run_root / "vivado_reports" / "harness_top_utilization_placed.rpt",
            f"{run_name}/vivado_reports/harness_top_utilization_placed.rpt",
            False,
        ),
    ]


def copy_power_artifacts(result_root: Path, bundle_dir: Path, copied: list[dict[str, Any]]) -> dict[str, Any]:
    power_root = result_root / "power_measurements"
    if not power_root.exists():
        return {"present": False, "copied_files": 0}

    before = len(copied)
    fixed_files = [
        power_root / "README.md",
        power_root / "power_measurement_manifest.json",
        power_root / "power_measurement_manifest.template.json",
        power_root / "reports" / "retrained_board_power_energy_summary.md",
        power_root / "reports" / "retrained_board_power_energy_summary.json",
        power_root / "reports" / "retrained_board_power_energy_summary.csv",
        power_root / "reports" / "retrained_board_power_energy_acceptance_audit.md",
        power_root / "reports" / "retrained_board_power_energy_acceptance_audit.json",
    ]
    for src in fixed_files:
        copy_file(src, bundle_dir / "power_measurements" / src.relative_to(power_root), copied, required=False)

    raw_root = power_root / "raw"
    if raw_root.exists():
        for src in sorted(raw_root.rglob("*")):
            if src.is_file():
                copy_file(src, bundle_dir / "power_measurements" / src.relative_to(power_root), copied, required=False)
    return {"present": True, "copied_files": len(copied) - before}


def write_readme(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase0 v3 Retrained-Weight Board Reinjection Evidence Bundle",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- source_result_root: `{manifest['source_result_root']}`",
        "- scope: compact evidence bundle for retrained checkpoint weights reinjected into isolated full-network board harness projects.",
        "- boundary: COM5 results are deterministic harness-input board latency/output sanity checks, not full NKSID validation-set board accuracy.",
        "- bitstream policy: bitstream files are not copied into this compact bundle; their absolute paths and SHA256 hashes are recorded in `evidence_bundle_manifest.json`.",
        "",
        "## Included Runs",
        "",
    ]
    for run in manifest["runs"]:
        lines.append(
            "- `{run_name}`: bitstream_sha256 `{bitstream_sha256}`, status `{pass_fail_note}`".format(**run)
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison/`: Markdown/JSON/CSV comparison reports and command record.",
            "- `comparison/retrained_board_reinject_acceptance_audit.*`: acceptance gate audit.",
            "- `power_measurements/`: optional CSV-imported measured power/energy evidence if present.",
            "- `rl_arch_*_trainedmem/`: per-run manifests, measurement outputs, logs, and Vivado reports.",
            "- `SHA256SUMS.txt`: hashes for every copied file in this bundle.",
            "- `evidence_bundle_manifest.json`: source paths, bitstream hashes, and copied-file manifest.",
        ]
    )
    (bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_bundle(result_root: Path, bundle_name: str, *, force: bool) -> Path:
    result_root = result_root.resolve()
    bundle_dir = result_root / "evidence_bundles" / bundle_name
    if bundle_dir.exists():
        if not force:
            raise FileExistsError(f"Bundle already exists: {bundle_dir}. Use --force to replace it.")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    copied: list[dict[str, Any]] = []
    reports = result_root / "reports"
    for name in [
        "retrained_board_reinject_comparison.md",
        "retrained_board_reinject_comparison.json",
        "retrained_board_reinject_comparison.csv",
        "retrained_board_reinject_commands.md",
        "retrained_board_reinject_acceptance_audit.md",
        "retrained_board_reinject_acceptance_audit.json",
        "static_mem_validation.json",
    ]:
        copy_file(reports / name, bundle_dir / "comparison" / name, copied, required=True)

    report_rows = read_json(reports / "retrained_board_reinject_comparison.json").get("rows", [])
    report_by_run = {str(row.get("run_name", "")): row for row in report_rows if isinstance(row, dict)}
    runs: list[dict[str, Any]] = []
    for run_root in run_dirs(result_root):
        run_name = run_root.name
        for src, rel_dst, required in candidate_artifacts_for_run(run_root):
            copy_file(src, bundle_dir / rel_dst, copied, required=required)
        bitstream = bitstream_info(run_root)
        row = report_by_run.get(run_name, {})
        runs.append(
            {
                "run_name": run_name,
                "arch_id": row.get("arch_id", ""),
                "positioning": row.get("positioning", ""),
                "pass_fail_note": row.get("pass_fail_note", ""),
                "bitstream_path": bitstream["path"],
                "bitstream_sha256": bitstream["sha256"],
                "com5_status_code": row.get("com5_status_code", ""),
                "cycles": row.get("cycles", ""),
                "real_e2e_latency_ms": row.get("real_e2e_latency_ms", ""),
                "board_checksum": row.get("board_checksum", ""),
                "board_argmax": row.get("board_argmax", ""),
            }
        )
    power_bundle = copy_power_artifacts(result_root, bundle_dir, copied)

    manifest = {
        "generated_at": timestamp(),
        "purpose": "Freeze Phase0 v3 retrained-weight board reinjection evidence for rl_arch_186 and rl_arch_242.",
        "source_result_root": str(result_root),
        "bundle_dir": str(bundle_dir.resolve()),
        "bitstream_policy": "Bitstream files are not copied; absolute paths and SHA256 hashes are recorded.",
        "accuracy_boundary": "PyTorch retrain150 is the validation-set accuracy source; COM5 is deterministic harness-input board sanity.",
        "runs": runs,
        "power_measurements": power_bundle,
        "copied_files": copied,
    }
    write_json(bundle_dir / "evidence_bundle_manifest.json", manifest)
    write_readme(bundle_dir, manifest)

    hash_entries = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            hash_entries.append(f"{sha256_file(path)}  {path.relative_to(bundle_dir).as_posix()}")
    (bundle_dir / "SHA256SUMS.txt").write_text("\n".join(hash_entries) + "\n", encoding="utf-8")
    return bundle_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package retrained board reinjection evidence.")
    parser.add_argument(
        "--result-root",
        default="hls_lut_builder/board_harness/results/retrain_phase0_v3_board_reinject_20260608",
    )
    parser.add_argument("--bundle-name", default="phase0_v3_retrained_board_reinject_20260608")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = package_bundle(Path(args.result_root), args.bundle_name, force=args.force)
    print(json.dumps({"bundle": str(bundle), "manifest": str(bundle / "evidence_bundle_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
