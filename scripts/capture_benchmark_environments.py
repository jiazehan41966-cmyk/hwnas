#!/usr/bin/env python3
"""Capture non-claimable environment cards for every pinned paper checkout."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths, sha256_file, write_json
from hwnas_fpga.benchmarks.registry import load_paper_registry


def _command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "returncode": None}
    result = subprocess.run(
        [executable, *command[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "available": True,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _dependency_manifests(checkout: Path) -> list[dict[str, Any]]:
    names = {
        "requirements.txt",
        "requirements-dev.txt",
        "environment.yml",
        "environment.yaml",
        "pyproject.toml",
        "setup.py",
        "Pipfile",
    }
    records = []
    if not checkout.exists():
        return records
    for path in checkout.rglob("*"):
        if path.is_file() and path.name in names and len(path.relative_to(checkout).parts) <= 3:
            records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return sorted(records, key=lambda row: row["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/benchmarks/paper_registry_v1.yaml")
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    args = parser.parse_args()
    paths = CampaignPaths.from_repo(PROJECT_ROOT, args.campaign_id).create()
    papers = load_paper_registry(PROJECT_ROOT / args.registry)
    freeze = _command([sys.executable, "-m", "pip", "freeze"])
    common = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "interpreter": sys.executable,
        },
        "cuda": _command(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "vivado": _command(["vivado", "-version"]),
        "vitis_hls": _command(["vitis_hls", "-version"]),
    }
    outputs = []
    dedicated_ready_count = 0
    for paper in papers:
        checkout = PROJECT_ROOT / paper.checkout_path
        commit = _command(["git", "rev-parse", "HEAD"], cwd=checkout) if checkout.exists() else None
        output = paths.artifact_root / "environment" / f"{paper.paper_id}.json"
        previous = (
            json.loads(output.read_text(encoding="utf-8"))
            if output.is_file()
            else {}
        )
        dedicated = previous.get("dedicated_environment")
        isolation_status = str(
            previous.get("isolation_status") or "PENDING_DEDICATED_ENVIRONMENT"
        )
        if isolation_status == "READY_DEDICATED_ENVIRONMENT" and dedicated:
            dedicated_ready_count += 1
        payload = {
            **previous,
            "schema_version": 1,
            "paper_id": paper.paper_id,
            "pinned_commit": paper.pinned_commit,
            "observed_commit": None if not commit else commit.get("stdout"),
            "checkout": str(checkout.resolve()),
            "license_state": paper.license_state,
            "dependency_manifests": _dependency_manifests(checkout),
            "environment": common,
            "pip_freeze": freeze,
            "isolation_status": isolation_status,
            "formal_eligible": False,
            "host_capture_boundary": (
                "Host capture is not the paper runtime. Any preserved dedicated environment "
                "must remain independently probe- and freeze-verified."
            ),
        }
        write_json(output, payload)
        outputs.append(str(output))
    index = write_json(
        paths.artifact_root / "environment" / "index.json",
        {
            "schema_version": 1,
            "campaign_id": args.campaign_id,
            "cards": outputs,
            "ready_count": dedicated_ready_count,
            "required_count": len(papers),
            "all_dedicated_environments_ready": (
                bool(papers) and dedicated_ready_count == len(papers)
            ),
        },
    )
    print(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
