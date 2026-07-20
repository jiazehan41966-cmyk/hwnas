#!/usr/bin/env python3
"""Run safe source-level smoke checks for the pinned external paper repositories."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths
from hwnas_fpga.benchmarks.registry import load_paper_registry, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/benchmarks/paper_registry_v1.yaml")
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    records = []
    for paper in load_paper_registry(REPO_ROOT / args.registry):
        checkout = (REPO_ROOT / paper.checkout_path).resolve()
        mode = str(paper.smoke.get("mode"))
        files = [str(value) for value in paper.smoke.get("files", [])]
        record = {
            "paper_id": paper.paper_id,
            "mode": mode,
            "checkout": str(checkout),
            "files": files,
            "status": "PENDING",
            "formal_eligible": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
        if not checkout.is_dir():
            record.update(status="BLOCKED", stderr="checkout_missing")
        elif mode == "py_compile":
            missing = [name for name in files if not (checkout / name).is_file()]
            if missing:
                record.update(status="BLOCKED", stderr=f"missing_files:{missing}")
            else:
                # Compile source in memory so the supposedly immutable author checkout
                # does not acquire __pycache__ or .pyc files during audit.
                command = [
                    args.python,
                    "-c",
                    (
                        "import sys; from tokenize import open as source_open; "
                        "[compile(source_open(path).read(), path, 'exec') "
                        "for path in sys.argv[1:]]"
                    ),
                    *files,
                ]
                result = subprocess.run(
                    command,
                    cwd=checkout,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                record.update(
                    status="PASS" if result.returncode == 0 else "FAIL",
                    executed_mode="source_compile_no_bytecode",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    command=command,
                )
        elif mode == "source_completeness_audit":
            missing = [name for name in files if not (checkout / name).is_file()]
            expected_files = [
                str(value) for value in paper.smoke.get("expected_files", [])
            ]
            missing_expected = [
                name for name in expected_files if not (checkout / name).is_file()
            ]
            record["expected_files"] = expected_files
            record["missing_expected_files"] = missing_expected
            if missing:
                record.update(status="BLOCKED", stderr=f"missing_source_files:{missing}")
            else:
                command = [
                    args.python,
                    "-c",
                    (
                        "import sys; from tokenize import open as source_open; "
                        "[compile(source_open(path).read(), path, 'exec') "
                        "for path in sys.argv[1:]]"
                    ),
                    *files,
                ]
                result = subprocess.run(
                    command,
                    cwd=checkout,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                record.update(
                    status=(
                        "BLOCKED_OFFICIAL_CODE_INCOMPLETE"
                        if result.returncode == 0 and missing_expected
                        else "PASS"
                        if result.returncode == 0
                        else "FAIL"
                    ),
                    executed_mode="source_compile_plus_readme_completeness_audit",
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=(
                        f"readme_listed_files_missing:{missing_expected}"
                        if result.returncode == 0 and missing_expected
                        else result.stderr
                    ),
                    command=command,
                )
        elif mode == "archive_audit":
            archives = []
            for name in files:
                path = checkout / name
                if path.is_file():
                    archives.append(
                        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    )
            record["archives"] = archives
            patterns = [str(value) for value in paper.smoke.get("extracted_patterns", [])]
            extracted = []
            missing_patterns = []
            for pattern in patterns:
                matches = sorted(checkout.glob(pattern))
                if not matches:
                    missing_patterns.append(pattern)
                extracted.extend(matches)
            record["extracted_sources"] = [
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(checkout)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in extracted
                if path.is_file()
            ]
            python_sources = [path for path in extracted if path.suffix.lower() == ".py"]
            compile_result = None
            if python_sources and not missing_patterns:
                command = [
                    args.python,
                    "-c",
                    (
                        "import sys; from tokenize import open as source_open; "
                        "[compile(source_open(path).read(), path, 'exec') "
                        "for path in sys.argv[1:]]"
                    ),
                    *[str(path) for path in python_sources],
                ]
                compile_result = subprocess.run(
                    command,
                    cwd=checkout,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                record["extracted_compile"] = {
                    "command": command,
                    "returncode": compile_result.returncode,
                    "stdout": compile_result.stdout,
                    "stderr": compile_result.stderr,
                }
            archive_ready = len(archives) == len(files)
            extracted_ready = not missing_patterns and bool(record["extracted_sources"])
            compile_ready = compile_result is not None and compile_result.returncode == 0
            record["missing_extracted_patterns"] = missing_patterns
            record["status"] = (
                "PASS_SOURCE_PRESENT"
                if archive_ready and extracted_ready and compile_ready
                else "BLOCKED"
            )
            record["stderr"] = (
                "license_unverified_method_adapter_pending"
                if record["status"] == "PASS_SOURCE_PRESENT"
                else "isolated_extraction_or_source_compile_incomplete"
            )
        else:
            record.update(status="BLOCKED", stderr=f"unsupported_smoke_mode:{mode}")
        records.append(record)

    paths = CampaignPaths.from_repo(REPO_ROOT, args.campaign_id).create()
    output = paths.artifact_root / "manifests" / "source_smoke.json"
    payload = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "python": args.python,
        "records": records,
        "all_source_checks_pass": all(
            row["status"] in {"PASS", "PASS_SOURCE_PRESENT"} for row in records
        ),
        "claim_boundary": (
            "In-memory source compilation is a syntax smoke, not a paper reproduction "
            "or local benchmark result."
        ),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
