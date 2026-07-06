#!/usr/bin/env python3
"""Fetch and verify the repository's registered external sonar datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.data.external_sources import (
    ExternalDatasetError,
    fetch_figshare_mine_detection,
    fetch_roboflow_cylider2,
    fetch_roboflow_cylider2_public_source,
    resolve_roboflow_api_key,
)


DATASETS = ("figshare_mine_detection", "roboflow_cylider2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download third-party sonar datasets into data/external, verify checksums, "
            "extract ZIP files safely, and write local provenance manifests."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to fetch (default: all).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external"),
        help="Local untracked dataset root.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download and verify archives without extracting them.",
    )
    parser.add_argument(
        "--roboflow-api-key",
        default=None,
        help="Roboflow API key. Prefer the ROBOFLOW_API_KEY environment variable.",
    )
    parser.add_argument(
        "--roboflow-zip",
        type=Path,
        default=None,
        help="Import a Roboflow dataset ZIP exported manually from Universe.",
    )
    parser.add_argument(
        "--roboflow-format",
        default="yolov8",
        help="Roboflow export format (default: yolov8).",
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=None,
        help="Roboflow version. API mode defaults to the latest published version.",
    )
    parser.add_argument(
        "--roboflow-acquisition",
        choices=("export", "public-source"),
        default="export",
        help=(
            "Use the official export ZIP or reconstruct original-resolution public "
            "images, current splits, and YOLO labels (default: export)."
        ),
    )
    parser.add_argument(
        "--roboflow-workers",
        type=int,
        default=6,
        help="Concurrent metadata/image requests for public-source mode.",
    )
    parser.add_argument(
        "--roboflow-tracked-index",
        type=Path,
        default=None,
        help="Optional Git-trackable copy of the reconstructed source index.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    extract = not args.no_extract
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries: dict[str, object] = {}
    failures: list[str] = []

    for dataset in selected:
        print(f"[dataset] acquiring {dataset}")
        try:
            if dataset == "figshare_mine_detection":
                summaries[dataset] = fetch_figshare_mine_detection(
                    output_root, extract=extract
                )
            else:
                api_key = resolve_roboflow_api_key(args.roboflow_api_key)
                if args.roboflow_acquisition == "public-source":
                    if args.roboflow_zip is not None:
                        raise ExternalDatasetError(
                            "--roboflow-zip cannot be combined with public-source mode"
                        )
                    summaries[dataset] = fetch_roboflow_cylider2_public_source(
                        output_root,
                        api_key=api_key or "",
                        version=args.roboflow_version,
                        workers=args.roboflow_workers,
                        tracked_index_output=args.roboflow_tracked_index,
                    )
                else:
                    summaries[dataset] = fetch_roboflow_cylider2(
                        output_root,
                        api_key=api_key,
                        local_zip=args.roboflow_zip,
                        export_format=args.roboflow_format,
                        version=args.roboflow_version,
                        extract=extract,
                    )
        except ExternalDatasetError as exc:
            failures.append(f"{dataset}: {exc}")
            print(f"[dataset] ERROR {failures[-1]}", file=sys.stderr)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "completed": sorted(summaries),
                "failed": failures,
                "manifests": {
                    name: summary.get("manifest_path")
                    for name, summary in summaries.items()
                    if isinstance(summary, dict)
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
