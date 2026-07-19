#!/usr/bin/env python3
"""Audit pinned paper code, licenses, and checkout shape; generate Table T1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.benchmarks.archive import CampaignPaths, TABLE_TITLES, write_table_bundle
from hwnas_fpga.benchmarks.registry import audit_source_checkout, load_paper_registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", default="configs/benchmarks/paper_registry_v1.yaml"
    )
    parser.add_argument("--campaign-id", default="ccf_ab_nksid_av7k325_v1")
    args = parser.parse_args()

    papers = load_paper_registry(REPO_ROOT / args.registry)
    audits = [audit_source_checkout(paper, REPO_ROOT) for paper in papers]
    paths = CampaignPaths.from_repo(REPO_ROOT, args.campaign_id).create()
    output = paths.artifact_root / "manifests" / "source_audit.json"
    payload = {
        "schema_version": 1,
        "campaign_id": args.campaign_id,
        "registry": str((REPO_ROOT / args.registry).resolve()),
        "formal_eligible_count": sum(bool(item["formal_eligible"]) for item in audits),
        "source_audit_pass_count": sum(bool(item.get("source_audit_pass")) for item in audits),
        "claim_boundary": (
            "A source audit or smoke pass is not a reproduced result. Formal eligibility "
            "requires a completed local unified-protocol run."
        ),
        "audits": audits,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for item in audits:
        paper = item["paper"]
        rows.append(
            {
                "paper_id": paper["paper_id"],
                "registry_role": paper["registry_role"],
                "direction": paper["direction"],
                "venue": paper["venue"],
                "class": paper["comparability_class"],
                "commit": item.get("observed_commit") or "MISSING",
                "pin_match": item["commit_matches_pin"],
                "license": paper.get("license_spdx") or "UNVERIFIED",
                "paper_code_correspondence": paper.get("paper_code_correspondence"),
                "source_shape": item["source_shape"],
                "source_audit_pass": item.get("source_audit_pass", False),
                "formal_eligible": item["formal_eligible"],
                "blockers": ";".join(item["blockers"]),
            }
        )
    bundle = write_table_bundle(paths, "T1", rows)
    print(json.dumps({"audit": str(output), "table": bundle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
