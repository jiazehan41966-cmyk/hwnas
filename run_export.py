#!/usr/bin/env python3
"""Export a searched/retrained checkpoint to ONNX and prepare HLS project files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.deploy import (
    HLSProjectConfig,
    attach_hls_report,
    create_hls_project_stub,
    export_checkpoint_quantized_weights,
    export_checkpoint_to_onnx,
)
from hwnas_fpga.runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX / HLS stub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or final_best_model.pt")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--output-dir", type=str, default=None, help="Export directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--input-channels", type=int, default=None)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--prepare-hls", action="store_true", help="Create an HLS project stub")
    parser.add_argument("--board", type=str, default=None)
    parser.add_argument("--clock-mhz", type=int, default=None)
    parser.add_argument("--fpga-part", type=str, default=None)
    parser.add_argument("--report", type=str, default=None, help="Optional HLS report to parse")
    parser.add_argument("--quantize-int8", action="store_true", help="Export an INT8 weight package for deployment")
    parser.add_argument("--quantized-output", type=str, default=None, help="Optional INT8 package output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware_cfg = config.get("hardware", {}) if isinstance(config, dict) else {}

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    export_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint.parent
    )
    export_root.mkdir(parents=True, exist_ok=True)
    onnx_path, metadata = export_checkpoint_to_onnx(
        checkpoint,
        export_path=export_root / "model.onnx",
        device=args.device,
        image_size_override=args.image_size,
        input_channels_override=args.input_channels,
        opset_version=args.opset,
    )
    print(f"ONNX exported: {onnx_path}")

    if args.prepare_hls:
        hls_dir = export_root / "hls_project"
        manifest = create_hls_project_stub(
            onnx_path,
            hls_dir,
            config=HLSProjectConfig(
                project_name=checkpoint.stem,
                board=args.board or hardware_cfg.get("board"),
                clock_mhz=args.clock_mhz or int(hardware_cfg.get("clock_mhz", 200)),
                fpga_part=args.fpga_part,
            ),
        )
        print(f"HLS project stub: {hls_dir}")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if args.report:
            metrics = attach_hls_report(hls_dir, args.report)
            print("Parsed HLS report:")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.quantize_int8:
        quantized_path, quant_summary = export_checkpoint_quantized_weights(
            checkpoint,
            output_path=args.quantized_output,
            device=args.device,
        )
        print(f"INT8 weight package: {quantized_path}")
        print(json.dumps(quant_summary, ensure_ascii=False, indent=2))

    metadata_path = onnx_path.with_suffix(".json")
    print(f"Export metadata: {metadata_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
