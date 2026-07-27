#!/usr/bin/env python3
"""Build the reproducible NKSID pHash leakage-stress group manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hwnas_fpga.data import (  # noqa: E402
    NKSIDDataset,
    build_inferred_group_manifest,
    write_group_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/NKSID")
    parser.add_argument("--hamming-threshold", type=int, default=4)
    parser.add_argument(
        "--output",
        default="artifacts/datasets/nksid_inferred_group_stress_v1.json",
    )
    args = parser.parse_args()

    dataset = NKSIDDataset(
        args.data_dir,
        is_training=False,
        use_kfold=False,
        split="full",
        augmentation_profile="none",
        geometry_mode="stretch_224",
    )
    payload = build_inferred_group_manifest(
        dataset.samples,
        args.data_dir,
        hamming_threshold=args.hamming_threshold,
    )
    payload["generator"] = {
        "script": str(Path(__file__).resolve()),
        "command": (
            "python scripts/build_nksid_group_manifest.py "
            f"--data-dir {args.data_dir} --hamming-threshold {args.hamming_threshold} "
            f"--output {args.output}"
        ),
    }
    output = write_group_manifest(args.output, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "schema",
                    "group_policy",
                    "sample_count",
                    "group_count",
                    "max_group_size",
                    "claim_boundary",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
