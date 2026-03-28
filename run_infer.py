#!/usr/bin/env python3
"""Inference entrypoint for searched sonar classifiers."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hwnas_fpga.deploy import (
    iter_image_paths,
    load_checkpoint_model,
    predict_image,
    resolve_class_names,
    resolve_inference_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a searched HW-NAS model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--image-size", type=int, default=None, help="Override input image size")
    parser.add_argument("--input-channels", type=int, default=None, help="Override input channels")
    parser.add_argument("--class-names", type=str, default=None, help="Comma-separated class names or path to txt/json")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to print")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_size, input_channels = resolve_inference_settings(
        args.checkpoint,
        image_size_override=args.image_size,
        input_channels_override=args.input_channels,
    )
    configured_class_names = resolve_class_names(class_names_arg=args.class_names)
    model, architecture, payload, class_names = load_checkpoint_model(
        args.checkpoint,
        device=device,
        class_names=configured_class_names or None,
    )
    if not configured_class_names and class_names:
        configured_class_names = class_names

    results = []
    for image_path in iter_image_paths(args.input):
        result = predict_image(
            model,
            image_path,
            image_size=image_size,
            class_names=configured_class_names or class_names,
            input_channels=input_channels,
            device=device,
            topk=args.topk,
        )
        results.append(result)
        print(f"{result['image_path']}")
        print(
            f"  Pred: {result['predicted_class_name']} "
            f"(idx={result['predicted_class_index']}, conf={result['confidence']:.4f})"
        )
        for rank, item in enumerate(result["topk"], start=1):
            print(
                f"  Top{rank}: {item['class_name']} "
                f"(idx={item['class_index']}, prob={item['probability']:.4f})"
            )

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "device": device,
            "image_size": image_size,
            "input_channels": input_channels,
            "architecture": architecture.to_dict(),
            "results": results,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
