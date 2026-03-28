"""Inference utilities for searched classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import yaml
from PIL import Image

from hwnas_fpga.data import NKSID_CLASSES, get_sonar_transforms
from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _infer_num_classes(
    encoding: dict[str, Any],
    state_dict: dict[str, Any],
    class_names: Optional[list[str]] = None,
) -> int:
    if encoding.get("num_classes") is not None:
        return int(encoding["num_classes"])
    if class_names:
        return len(class_names)
    for key in ("head.fc.weight", "head.fc2.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Unable to infer num_classes from checkpoint.")


def _extract_checkpoint_candidate(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "candidate" in payload and isinstance(payload["candidate"], dict):
        candidate = dict(payload["candidate"])
        encoding = dict(candidate["encoding"])
        return candidate, encoding
    if "architecture" in payload:
        encoding = dict(payload["architecture"])
        candidate = dict(payload.get("source_candidate") or {})
        candidate.setdefault("arch_id", "retrained_best")
        candidate["encoding"] = encoding
        candidate.setdefault("metrics", payload.get("metrics", {}))
        return candidate, encoding
    raise ValueError("Checkpoint does not contain candidate or architecture metadata.")


def load_run_config_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.yaml"
    cli_args_path = run_dir / "cli_args.json"
    cli_args: dict[str, Any] = {}
    if cli_args_path.exists():
        with cli_args_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        if isinstance(payload, dict):
            cli_args = payload
    if not config_path.exists():
        return {"cli_args": cli_args} if cli_args else {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        payload = {}
    if cli_args:
        payload["cli_args"] = cli_args
    return payload


def resolve_class_names(
    *,
    class_names_arg: Optional[str],
    num_classes: Optional[int] = None,
) -> list[str]:
    if class_names_arg:
        path = Path(class_names_arg).expanduser()
        if path.exists():
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, list):
                    raise ValueError("Class name JSON must contain a list.")
                return [str(item) for item in payload]
            with path.open("r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        return [item.strip() for item in class_names_arg.split(",") if item.strip()]

    if num_classes == len(NKSID_CLASSES):
        return list(NKSID_CLASSES)
    if num_classes is None:
        return []
    return [f"class_{index}" for index in range(num_classes)]


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    class_names: Optional[list[str]] = None,
) -> tuple[torch.nn.Module, ArchitectureSpec, dict[str, Any], list[str]]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(checkpoint, map_location=device)
    candidate, encoding = _extract_checkpoint_candidate(payload)
    state_dict = payload["model_state_dict"]
    resolved_class_names = resolve_class_names(
        class_names_arg=None,
        num_classes=_infer_num_classes(encoding, state_dict, class_names),
    )
    if class_names:
        resolved_class_names = class_names

    architecture = ArchitectureSpec.from_dict(encoding)
    num_classes = _infer_num_classes(encoding, state_dict, resolved_class_names)
    model = build_model(architecture, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, architecture, payload, resolved_class_names


def resolve_inference_settings(
    checkpoint_path: str | Path,
    *,
    image_size_override: Optional[int] = None,
    input_channels_override: Optional[int] = None,
) -> tuple[int, int]:
    config = load_run_config_from_checkpoint(checkpoint_path)
    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    cli_args = config.get("cli_args", {}) if isinstance(config, dict) else {}
    image_size = image_size_override or cli_args.get("image_size") or int(dataset_cfg.get("image_size", 224))
    input_channels = input_channels_override or int(dataset_cfg.get("input_channels", 1))
    return image_size, input_channels


def preprocess_image(
    image_path: str | Path,
    *,
    image_size: int,
    input_channels: int = 1,
) -> torch.Tensor:
    path = Path(image_path).expanduser().resolve()
    image = Image.open(path)
    if input_channels == 1 and image.mode != "L":
        image = image.convert("L")
    elif input_channels == 3 and image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_sonar_transforms(
        image_size=image_size,
        is_training=False,
    )
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    *,
    image_size: int,
    class_names: Optional[list[str]] = None,
    input_channels: int = 1,
    device: str = "cpu",
    topk: int = 3,
) -> dict[str, Any]:
    inputs = preprocess_image(
        image_path,
        image_size=image_size,
        input_channels=input_channels,
    ).to(device)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, int(probabilities.shape[0]))
    scores, indices = torch.topk(probabilities, k=k)
    resolved_names = class_names or [f"class_{index}" for index in range(int(probabilities.shape[0]))]
    top_predictions = [
        {
            "class_index": int(index),
            "class_name": resolved_names[int(index)] if int(index) < len(resolved_names) else str(int(index)),
            "probability": float(score),
        }
        for score, index in zip(scores.tolist(), indices.tolist())
    ]
    return {
        "image_path": str(Path(image_path).expanduser().resolve()),
        "predicted_class_index": top_predictions[0]["class_index"],
        "predicted_class_name": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["probability"],
        "topk": top_predictions,
    }


def iter_image_paths(input_path: str | Path) -> Iterable[Path]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
