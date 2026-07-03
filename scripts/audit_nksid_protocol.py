#!/usr/bin/env python3
"""Write a machine-readable and human-readable NKSID protocol audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hwnas_fpga.data.audit import audit_nksid_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--output-dir", default="results/first_principles_audit_20260703")
    return parser.parse_args()


def render_markdown(payload: dict) -> str:
    integrity = payload["integrity"]
    split = payload["split_protocol"]
    selected = payload["selected_fold"]
    fraction = selected["filename_neighbor_fraction"]
    fraction_text = "n/a" if fraction is None else f"{fraction:.1%}"
    return "\n".join(
        [
            "# NKSID Protocol Audit",
            "",
            f"- Samples: `{payload['sample_count']}`",
            f"- Class counts: `{payload['class_counts']}`",
            f"- Missing files: `{integrity['missing_file_count']}`",
            f"- Exact duplicate copies: `{integrity['exact_duplicate_extra_copies']}`",
            f"- Split records: `{split['record_count']}`",
            f"- Inferred protocol: `{split['inferred_repeats']} x {split['inferred_k']}-fold`",
            f"- Selected record: `{selected['fold_record']}`",
            f"- Selected train/validation: `{selected['train_count']}/{selected['val_count']}`",
            f"- Validation samples with a same-class filename neighbor in train: "
            f"`{selected['val_with_train_filename_neighbor']}/{selected['numbered_val_count']}` "
            f"(`{fraction_text}`)",
            f"- Critical failures: `{payload['critical_failures']}`",
            f"- Warnings: `{payload['warnings']}`",
            "",
            "## Claim boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    payload = audit_nksid_protocol(
        args.data_dir,
        fold=args.fold,
        neighbor_radius=args.neighbor_radius,
        hash_files=args.hash_files,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nksid_protocol_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "nksid_protocol_audit.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["critical_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
