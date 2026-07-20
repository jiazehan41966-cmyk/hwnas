#!/usr/bin/env python3
"""Verify and lock the six paper-specific unified-protocol runtimes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths, sha256_file, write_json
from hwnas_fpga.benchmarks.registry import load_paper_registry
from hwnas_fpga.training.protocol_reporting import canonical_sha256


def _run(command: list[str], *, timeout: int = 180) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    parser.add_argument("--registry", default="configs/benchmarks/paper_registry_v1.yaml")
    parser.add_argument("--runtime-config", default="configs/benchmarks/runtime_environments_v1.yaml")
    args = parser.parse_args()
    paths = CampaignPaths.from_repo(PROJECT_ROOT, args.campaign_id).create()
    papers = {paper.paper_id: paper for paper in load_paper_registry(PROJECT_ROOT / args.registry)}
    config_path = PROJECT_ROOT / args.runtime_config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    runtime_rows = dict(config.get("papers") or {})
    cards = []
    all_ready = True
    for paper_id, runtime in runtime_rows.items():
        paper = papers[paper_id]
        env_root = (PROJECT_ROOT / str(runtime["environment_path"])).resolve()
        python = env_root / "Scripts" / "python.exe"
        probe_command = [
            str(python),
            str(PROJECT_ROOT / "scripts/probe_benchmark_environment.py"),
            "--paper-id", paper_id,
            "--probe", str(runtime["probe"]),
            "--checkout", paper.checkout_path,
            "--pinned-commit", paper.pinned_commit,
        ]
        if runtime.get("archive_sha256"):
            probe_command.extend(["--archive-sha256", str(runtime["archive_sha256"])])
        probe = (
            _run(probe_command) if python.is_file()
            else {"command": probe_command, "returncode": None, "stdout": "", "stderr": "interpreter missing"}
        )
        parsed_probe = None
        if probe["returncode"] == 0:
            try:
                parsed_probe = json.loads(probe["stdout"].splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                probe["stderr"] = f"invalid probe JSON; {probe['stderr']}".strip()
        freeze = _run(["uv", "pip", "freeze", "--python", str(python)]) if python.is_file() else None
        freeze_path = paths.artifact_root / "environment" / f"{paper_id}.lock.txt"
        if freeze and freeze["returncode"] == 0:
            freeze_path.write_text(freeze["stdout"] + "\n", encoding="utf-8")
        freeze_record = {
            "path": str(freeze_path.resolve()),
            "sha256": sha256_file(freeze_path) if freeze_path.is_file() else None,
            "package_count": (
                len(freeze["stdout"].splitlines())
                if freeze and freeze["returncode"] == 0
                else 0
            ),
        }
        dedicated_record = {
            "path": str(env_root),
            "interpreter": str(python),
            "probe": parsed_probe,
            "probe_process": probe,
            "freeze": freeze_record,
        }
        dedicated_record["verification_fingerprint"] = canonical_sha256(
            {
                "runtime_role": config.get("runtime_role"),
                "path": dedicated_record["path"],
                "interpreter": dedicated_record["interpreter"],
                "probe": dedicated_record["probe"],
                "freeze": dedicated_record["freeze"],
            }
        )
        ready = bool(
            parsed_probe
            and parsed_probe.get("status") == "PASS"
            and parsed_probe.get("cuda_available") is True
            and parsed_probe.get("torch") == config.get("expected_torch")
            and str(parsed_probe.get("cuda_build")) == str(config.get("expected_cuda"))
            and Path(str(parsed_probe.get("sys_prefix"))).resolve() == env_root
            and freeze_path.is_file()
        )
        all_ready = all_ready and ready
        card_path = paths.artifact_root / "environment" / f"{paper_id}.json"
        previous = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
        previous.update(
            {
                "dedicated_environment_generated": datetime.now().isoformat(timespec="seconds"),
                "runtime_role": config.get("runtime_role"),
                "dedicated_environment": dedicated_record,
                "isolation_status": "READY_DEDICATED_ENVIRONMENT" if ready else "PARTIAL_DEDICATED_ENVIRONMENT",
                "formal_eligible": False,
                "boundary": (
                    "This is a paper-specific unified-protocol adapter runtime, not a byte-for-byte "
                    "reconstruction of the author's legacy environment. Environment readiness does "
                    "not make smoke or formal results claimable."
                ),
            }
        )
        write_json(card_path, previous)
        cards.append(str(card_path.resolve()))
    index = write_json(
        paths.artifact_root / "environment" / "index.json",
        {
            "schema_version": 2,
            "campaign_id": args.campaign_id,
            "runtime_config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
            "cards": cards,
            "ready_count": sum(
                json.loads(Path(path).read_text(encoding="utf-8")).get("isolation_status")
                == "READY_DEDICATED_ENVIRONMENT"
                for path in cards
            ),
            "required_count": len(runtime_rows),
            "all_dedicated_environments_ready": all_ready,
        },
    )
    print(index)
    return 0 if all_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
