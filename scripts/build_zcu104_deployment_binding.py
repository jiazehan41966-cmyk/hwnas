#!/usr/bin/env python3
"""Build a SHA-bound ZCU104 deployment manifest from produced artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.hardware.deployment_binding import (  # noqa: E402
    audit_zcu104_deployment_binding,
    sha256_file,
)


def artifact_record(value: str) -> dict[str, object]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch-id", required=True)
    parser.add_argument("--architecture-sha256", required=True)
    parser.add_argument("--board-serial", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--xsa", required=True)
    parser.add_argument("--hwh", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--toolchain-manifest", required=True)
    parser.add_argument("--transport-id", default="axi_dma_0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    toolchain = artifact_record(args.toolchain_manifest)
    payload = {
        "schema_version": 1,
        "campaign_id": "zcu104_full_campaign_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "arch_id": args.arch_id,
            "architecture_sha256": args.architecture_sha256,
        },
        "platform": {
            "board_model": "ZCU104",
            "target_part": "xczu7ev-ffvc1156-2-e",
            "board_serial": args.board_serial,
        },
        "transport": {"type": "axi_dma", "id": args.transport_id},
        "toolchain": {
            "vivado_version": "2022.2",
            "vitis_hls_version": "2022.2",
            "manifest_path": toolchain["path"],
            "sha256": toolchain["sha256"],
        },
        "artifacts": {
            "checkpoint": artifact_record(args.checkpoint),
            "bitstream": artifact_record(args.bitstream),
            "xsa": artifact_record(args.xsa),
            "hwh": artifact_record(args.hwh),
            "runtime": artifact_record(args.runtime),
        },
    }
    payload["audit"] = audit_zcu104_deployment_binding(payload)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["audit"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
