"""Build the analytic 640x640 multi-scale detection feature-map cost table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hwnas_fpga.hardware import (  # noqa: E402
    FPGACostEstimator,
    FixedMultiscaleDetectionCostConfig,
    detection_cost_table_payload,
)
from hwnas_fpga.runtime import (  # noqa: E402
    build_constraints,
    build_hardware_spec,
    build_search_space,
    load_config,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "search"
    / "zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "detection_multiscale_640"
    / "hardware_cost_table.analytic.json"
)


def _canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_payload(config_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    dataset_cfg = dict(config.get("dataset") or {})
    detection_cfg = dict(config.get("detection") or {})
    constraints = build_constraints(config)
    hardware_spec = build_hardware_spec(config)
    search_space = build_search_space(
        config,
        image_size=int(dataset_cfg.get("image_size", detection_cfg.get("input_size", 640))),
        input_channels=int(dataset_cfg.get("input_channels", 1)),
        num_classes=int(dataset_cfg.get("num_classes", 1)),
        constraints=constraints,
    )
    architecture = search_space.baseline_architecture()
    cost_config = FixedMultiscaleDetectionCostConfig(
        num_classes=int(dataset_cfg["num_classes"]),
        feature_stage_indices=tuple(
            int(value)
            for value in detection_cfg.get("feature_stage_indices", (3, 5, 7))
        ),
        neck_channels=int(
            (detection_cfg.get("fixed_neck") or {}).get("out_channels", 96)
        ),
        anchors_per_location=int(
            (detection_cfg.get("fixed_head") or {}).get("anchors_per_location", 3)
        ),
    )
    estimator = FPGACostEstimator(
        hardware_spec=hardware_spec,
        constraints=constraints,
        quantization_bits=int((config.get("hardware") or {}).get("quantization_bits", 8)),
    )
    payload = detection_cost_table_payload(
        architecture,
        search_space,
        estimator,
        cost_config,
    )
    payload["config"] = {
        "path": str(config_path.resolve()),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    unsigned = dict(payload)
    payload["payload_sha256"] = _canonical_sha256(unsigned)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build analytic 640x640 detection feature-map cost table"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()

    payload = build_payload(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "payload_sha256": payload["payload_sha256"],
                "rows": len(payload["rows"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
