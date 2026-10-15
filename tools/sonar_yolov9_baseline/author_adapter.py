"""Protocol adapter for the pinned author's YOLOv9-T implementation.

The detector, dual TaskAlignedAssigner, CIoU and DFL losses remain author code.
This module changes only input/target representation, initialization and the
frozen project postprocessing policy. It never downloads weights or installs
packages. The official tiny release provides converted COCO weights, which
are mapped explicitly into the original dual training graph's main branch.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import yaml


AUTHOR_LOSS_GAINS = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
AUTHOR_ASSIGNER_DEFAULTS = {"YOLOM": "10", "YOLOA": "0.5", "YOLOB": "6.0"}
OFFICIAL_TINY_CONVERTED_SHA256 = "61e080e964e65e32b884477c5e6344c607c7e02103d64649de810edaeb869803"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_author_modules(source_root: str | Path):
    """Import original absolute ``models``/``utils`` packages without shadows."""
    root = Path(source_root).expanduser().resolve()
    required = ("models/yolo.py", "models/detect/yolov9-t.yaml", "utils/loss_tal_dual.py")
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Incomplete YOLOv9 author source: {root / relative}")
    for name in ("models", "utils"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            locations = list(getattr(loaded, "__path__", []))
            module_file = getattr(loaded, "__file__", None)
            if module_file:
                locations.append(str(Path(module_file).parent))
            if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError(f"YOLOv9 absolute import would reuse another {name!r} package")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Author check_requirements may install packages by default. Missing packages
    # must instead be handled explicitly by the experiment environment owner.
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ.setdefault(
        "YOLOV5_CONFIG_DIR",
        str(Path(__file__).resolve().parents[2] / "results/sonar_yolov9_baseline/author_config"),
    )
    Path(os.environ["YOLOV5_CONFIG_DIR"]).expanduser().mkdir(parents=True, exist_ok=True)
    try:
        model_module = importlib.import_module("models.yolo")
        loss_module = importlib.import_module("utils.loss_tal_dual")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"YOLOv9 requires missing dependency {exc.name!r}; provision the isolated "
            "author environment before constructing this model. No package was installed."
        ) from exc
    for module in (model_module, loss_module):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Wrong author module imported: {module.__file__}")
    return root, model_module, loss_module


def _validate_assigner_environment() -> None:
    # The author's loss reads these environment variables. Fail on a hidden
    # setting rather than silently changing the frozen assignment procedure.
    for name, default in AUTHOR_ASSIGNER_DEFAULTS.items():
        value = os.environ.get(name, default)
        try:
            actual = int(value) if name == "YOLOM" else float(value)
        except ValueError as exc:
            raise ValueError(f"Malformed author assignment environment: {name}={value}") from exc
        if actual != float(default):
            raise ValueError(f"Nondefault author assignment environment: {name}={value}")


def build_author_model(
    source_root: str | Path,
    init_condition: str,
    checkpoint_path: str | Path | None = None,
    num_classes: int = 2,
    device: str | torch.device = "cpu",
    *,
    expected_checkpoint_sha256: str | None = None,
) -> torch.nn.Module:
    """Build native, unconverted YOLOv9-T with explicit initialization.

    ``init_condition`` is ``scratch`` or ``coco_pretrained``. Return the original
    DetectionModel, with JSON-serializable ``author_adapter_metadata``. Use
    :func:`forward_author` for the project's one-channel tensors.

    Initialize ``nc=2`` first, as author train_dual.py does. For unconverted
    checkpoints transfer same-name/same-shape tensors. For the official tiny
    converted checkpoint, map layers 0--21 directly and map its DDetect head
    into the dual main head. Auxiliary layers remain random and this partial
    transfer is recorded explicitly. No COCO class-index slicing is performed.
    Changing nc also changes native hidden classification widths, so more
    than final logits can be reinitialized.
    """
    if init_condition not in {"scratch", "coco_pretrained"}:
        raise ValueError("init_condition must be scratch or coco_pretrained")
    if int(num_classes) != 2:
        raise ValueError("This frozen sonar adapter requires exactly two classes")
    if init_condition == "scratch" and checkpoint_path is not None:
        raise ValueError("Scratch construction rejects any checkpoint_path")
    if init_condition == "coco_pretrained" and checkpoint_path is None:
        raise ValueError("COCO initialization requires an existing explicit checkpoint")
    _validate_assigner_environment()
    root, model_module, _ = _load_author_modules(source_root)
    config_path = root / "models/detect/yolov9-t.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = model_module.DetectionModel(deepcopy(config), ch=3, nc=2).to(device)
    head = model.model[-1]
    if type(head).__name__ != "DualDDetect" or head.nl != 3 or head.nc != 2:
        raise RuntimeError("YOLOv9-T adapter requires the original three-scale DualDDetect")
    model.nc = 2
    model.names = ["cylider", "manta"]
    # train_dual.py's nc/80 gain-scaling lines are commented at the pinned source.
    # ComputeLoss hardcodes these gains; cls_pw/fl_gamma are read from hyp.
    model.hyp = {**AUTHOR_LOSS_GAINS, "cls_pw": 1.0, "fl_gamma": 0.0, "label_smoothing": 0.0}
    native_state = model.state_dict()
    metadata: dict[str, Any] = {
        "architecture": "author_yolov9_t_dual",
        "source_root": str(root),
        "config_sha256": _sha256(config_path),
        "source_sha256": {
            relative: _sha256(root / relative)
            for relative in (
                "models/yolo.py", "models/common.py", "utils/loss_tal_dual.py",
                "utils/tal/assigner.py", "utils/tal/anchor_generator.py", "train_dual.py",
            )
        },
        "init_condition": init_condition,
        "input": "dataset L grayscale + resize + augmentation, then exact channel repeat to 3",
        "native_loss": "utils.loss_tal_dual.ComputeLoss",
        "native_loss_reduction": "sum(box,cls,dfl) * batch_size, unchanged author behavior",
        "loss_gains": dict(AUTHOR_LOSS_GAINS),
        "nc_gain_scaling": False,
        "assignment": {"topk": 10, "alpha": 0.5, "beta": 6.0, "auxiliary_weight": 0.25},
        "inference_branch": 1,
        "checkpoint": None,
        "loaded_keys": [],
        "missing_keys": list(native_state),
        "shape_mismatches": {},
        "excluded_keys": [],
        "unexpected_checkpoint_keys": [],
    }
    if init_condition == "coco_pretrained":
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit COCO checkpoint does not exist: {path}")
        checkpoint_sha = _sha256(path)
        expected_sha = expected_checkpoint_sha256 or OFFICIAL_TINY_CONVERTED_SHA256
        if checkpoint_sha != expected_sha:
            raise ValueError("COCO checkpoint SHA256 mismatch before author pickle deserialization")
        # Official release checkpoints pickle model classes. Only load the
        # pinned, SHA-bound official artifact, never a user-untrusted pickle.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        pretrained = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(pretrained, torch.nn.Module):
            raise ValueError("Expected official unconverted checkpoint['model'] module")
        source_head_type = type(pretrained.model[-1]).__name__
        if pretrained.model[-1].nc != 80:
            raise ValueError("COCO transfer requires an 80-class author checkpoint")
        converted = source_head_type == "DDetect" and len(pretrained.model) == 23
        original = source_head_type == "DualDDetect" and len(pretrained.model) == 30
        if not (converted or original):
            raise ValueError("Checkpoint must be the original or officially converted YOLOv9-T")
        pretrained_state = pretrained.float().state_dict()
        common = {}
        metadata["checkpoint_architecture"] = "converted_single" if converted else "original_dual"
        metadata["transfer_scope"] = (
            "partial COCO: shared backbone/main neck and main regression head; "
            "auxiliary neck/head random; incompatible two-class classification tensors random"
            if converted else "native author same-name/same-shape transfer"
        )
        metadata["key_mapping"] = {}
        metadata["auxiliary_pretrained"] = not converted
        for key, value in pretrained_state.items():
            target_key = _converted_to_dual_key(key) if converted else key
            if "anchor" in key:  # author excludes anchors when explicit cfg is supplied
                metadata["excluded_keys"].append(key)
            elif target_key is None or target_key not in native_state:
                metadata["unexpected_checkpoint_keys"].append(key)
            elif value.shape != native_state[target_key].shape:
                metadata["shape_mismatches"][target_key] = {
                    "source_key": key,
                    "checkpoint": list(value.shape), "target": list(native_state[target_key].shape),
                }
            else:
                common[target_key] = value
                metadata["key_mapping"][key] = target_key
        incompatible = model.load_state_dict(common, strict=False)
        metadata["loaded_keys"] = list(common)
        metadata["missing_keys"] = list(incompatible.missing_keys)
        metadata["checkpoint"] = {
            "path": str(path), "sha256": checkpoint_sha, "expected_sha256": expected_sha,
            "sha256_verified_before_load": True, "source_classes": 80,
        }
        metadata["loaded_tensor_count"] = len(common)
        metadata["model_tensor_count"] = len(native_state)
        metadata["loaded_numel"] = sum(x.numel() for x in common.values())
        metadata["model_state_numel"] = sum(x.numel() for x in native_state.values())
        if not common or incompatible.unexpected_keys:
            raise RuntimeError("COCO transfer did not produce a valid nonempty shape intersection")
    model.author_adapter_metadata = metadata
    return model


def _converted_to_dual_key(key: str) -> str | None:
    """Map the official 23-layer tiny inference graph into the 30-layer graph."""
    pieces = key.split(".", 2)
    if len(pieces) != 3 or pieces[0] != "model" or not pieces[1].isdigit():
        return None
    index, suffix = int(pieces[1]), pieces[2]
    if 0 <= index <= 21:
        return key
    if index != 22:
        return None
    for source, target in (("cv2.", "cv4."), ("cv3.", "cv5."), ("dfl.", "dfl2.")):
        if suffix.startswith(source):
            return "model.29." + target + suffix[len(source):]
    return None


def build_main_inference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Remove only native auxiliary layers; retain an exact main-branch graph.

    Used for deployment parameter/MAC reporting. Training remains DualDDetect.
    Construction preserves training RNG. Callers must verify numerical parity
    of this single-head output against the original dual output's branch 1.
    """
    if type(model.model[-1]).__name__ != "DualDDetect" or len(model.model) != 30:
        raise ValueError("Expected original 30-layer YOLOv9-T training graph")
    config = deepcopy(model.yaml)
    # The main neck ends at layer 21. Layers 22--28 are auxiliary and the
    # original layer 29 joins auxiliary/main features in DualDDetect.
    config["head"] = config["head"][:13] + [[[15, 18, 21], 1, "DDetect", ["nc"]]]
    source_class = type(model)
    with torch.random.fork_rng(devices=[]):
        inference_model = source_class(config, ch=3, nc=2)
    original_state = model.state_dict()
    main_state = inference_model.state_dict()
    mapped = {}
    for key in main_state:
        source_key = _converted_to_dual_key(key)
        if source_key is None or source_key not in original_state:
            raise RuntimeError(f"Missing main-branch tensor mapping: {key}")
        if original_state[source_key].shape != main_state[key].shape:
            raise RuntimeError(f"Main-branch tensor shape mismatch: {key}")
        mapped[key] = original_state[source_key]
    inference_model.load_state_dict(mapped, strict=True)
    inference_model.to(next(model.parameters()).device).eval()
    inference_model.nc = 2
    inference_model.names = list(model.names)
    inference_model._author_adapter_main_only = True
    inference_model.author_adapter_metadata = {
        "architecture": "author_yolov9_t_main_inference",
        "derived_from": deepcopy(model.author_adapter_metadata),
        "removed_auxiliary_layers": list(range(22, 29)),
        "inference_head": "DDetect",
        "training_graph_unchanged": True,
        "parity": "requires caller numerical verification",
    }
    return inference_model


def gray_to_three_channels(images: torch.Tensor) -> torch.Tensor:
    """Repeat already prepared grayscale; never re-read original files as RGB."""
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("Expected batched Bx1xHxW grayscale images")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Expected finite normalized floating grayscale images")
    if images.numel() and (images.min() < 0 or images.max() > 1):
        raise ValueError("Expected grayscale intensities in [0,1]")
    if images.shape[1] == 3:
        if not torch.equal(images[:, 0], images[:, 1]) or not torch.equal(images[:, 0], images[:, 2]):
            raise ValueError("Three-channel input must contain identical grayscale channels")
        return images
    return images.repeat(1, 3, 1, 1)


def forward_author(model: torch.nn.Module, images: torch.Tensor):
    return model(gray_to_three_channels(images))


def targets_to_yolo(
    targets: Sequence[Mapping[str, Any]],
    *,
    image_size: int = 640,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert resized-pixel targets to native [batch, class, cx, cy, w, h]."""
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    rows = []
    for batch_index, target in enumerate(targets):
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32, device=device).reshape(-1, 4)
        labels = torch.as_tensor(target["labels"], device=device).reshape(-1)
        if len(boxes) != len(labels) or not torch.isfinite(boxes).all():
            raise ValueError("Malformed target boxes or label count")
        if labels.numel() and ((labels < 0).any() or (labels >= 2).any() or (labels != labels.long()).any()):
            raise ValueError("Targets must use integral foreground labels 0 and 1")
        if boxes.numel() and (
            (boxes < 0).any() or (boxes > image_size).any() or (boxes[:, 2:] <= boxes[:, :2]).any()
        ):
            raise ValueError("Targets must contain positive-area resized-pixel xyxy boxes")
        if len(boxes):
            centers = (boxes[:, :2] + boxes[:, 2:]) / (2.0 * image_size)
            sizes = (boxes[:, 2:] - boxes[:, :2]) / float(image_size)
            rows.append(torch.cat((
                torch.full((len(boxes), 1), float(batch_index), device=device),
                labels.float().unsqueeze(1), centers, sizes,
            ), dim=1))
    return torch.cat(rows, dim=0) if rows else torch.empty((0, 6), dtype=torch.float32, device=device)


def author_loss(
    model: torch.nn.Module,
    predictions,
    targets: Sequence[Mapping[str, Any]],
    *,
    image_size: int = 640,
):
    """Delegate unchanged native dual-head loss; support native eval output too."""
    _validate_assigner_environment()
    device = next(model.parameters()).device
    criterion = getattr(model, "_author_adapter_criterion", None)
    if criterion is None or criterion.device != device:
        loss_module = importlib.import_module("utils.loss_tal_dual")
        criterion = loss_module.ComputeLoss(model)
        model._author_adapter_criterion = criterion
    native_targets = targets_to_yolo(targets, image_size=image_size, device=device)
    loss, items = criterion(predictions, native_targets)
    if not torch.isfinite(loss) or not torch.isfinite(items).all():
        raise ValueError("Nonfinite author YOLOv9 dual loss")
    return loss, {name: items[index] for index, name in enumerate(("box", "cls", "dfl"))}


def decode_author(
    model: torch.nn.Module,
    predictions,
    *,
    score_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    pre_nms_topk: int = 500,
    max_detections: int = 100,
    image_size: int = 640,
) -> list[dict[str, torch.Tensor]]:
    """Native main-head decoded xywh -> project frozen postprocessing.

    The native head already applies DFL, strides and class sigmoid. Choose its
    main branch, take its best class per location, and run exactly one project
    class-wise NMS after global top-k. No author default NMS is run first.
    """
    from hwnas_fpga.training.detection_trainer import _nms_xyxy

    if model.training:
        raise ValueError("decode_author requires model.eval() without switching BatchNorm state")
    decoded = predictions[0] if isinstance(predictions, tuple) else predictions
    if not getattr(model, "_author_adapter_main_only", False):
        if not isinstance(decoded, (list, tuple)) or len(decoded) != 2:
            raise ValueError("Expected native DualDDetect decoded auxiliary/main branches")
        decoded = decoded[1]
    if decoded.ndim != 3 or decoded.shape[1] != 6:
        raise ValueError("Expected Bx(4+2)xN native main-head output")
    if not torch.isfinite(decoded).all():
        raise ValueError("Nonfinite native main-head output")
    if not (0 <= score_threshold <= 1 and 0 <= nms_iou_threshold <= 1):
        raise ValueError("Invalid decode score or NMS threshold")
    if int(max_detections) <= 0:
        raise ValueError("max_detections must be positive")
    outputs = []
    for prediction in decoded:
        xywh = prediction[:4].T
        boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), dim=1)
        boxes = boxes.clamp(min=0.0, max=float(image_size))
        scores, labels = prediction[4:].max(dim=0)
        if (scores < 0).any() or (scores > 1).any():
            raise ValueError("Native class probabilities must already be sigmoid scores")
        selected = scores >= float(score_threshold)
        boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
        if pre_nms_topk > 0 and len(scores) > pre_nms_topk:
            selected = scores.topk(int(pre_nms_topk)).indices
            boxes, scores, labels = boxes[selected], scores[selected], labels[selected]
        kept = []
        for label in labels.unique().tolist():
            indices = torch.where(labels == int(label))[0]
            kept.extend(indices[_nms_xyxy(boxes[indices], scores[indices], iou_threshold=nms_iou_threshold)].tolist())
        kept = torch.tensor(kept, dtype=torch.long, device=boxes.device)
        if kept.numel():
            kept = kept[scores[kept].argsort(descending=True)][:int(max_detections)]
        outputs.append({
            "boxes": boxes[kept].detach().cpu(),
            "scores": scores[kept].detach().cpu(),
            "labels": labels[kept].long().detach().cpu(),
        })
    return outputs
