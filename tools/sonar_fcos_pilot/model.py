"""Official FCOS with three explicitly bounded FPN convolution choices.

The dataset owns letterboxing to a square. FCOS keeps its official normalization,
assignment, head and postprocessing, with a matching 640-pixel transform.
``num_classes`` includes background; foreground labels are 1 through K.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn

from .common import ROOT, environment, seed_everything, sha256_file, write_json


COMBINATIONS = ("AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB")
DEFAULT_CACHE = ROOT / "results" / "sonar_fcos_pilot_20261008" / "cache" / "torch"


def build_model(combo, num_classes, seed, pretrained=True, image_size=640, cache_dir=None):
    """Build one candidate; initial tensors depend on seed, never on combo.

    A/B change attributes of existing convolutions, without recreating their
    tensors. All ResNet convolution parameters remain trainable. Frozen batch
    normalization in the official pretrained backbone remains the official
    behavior, rather than introducing new trainable normalization parameters.
    """
    if combo not in COMBINATIONS:
        raise ValueError(f"combo must be one of {COMBINATIONS}, got {combo!r}")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
    if image_size != 640:
        raise ValueError("The approved pilot uses an unchanged 640-pixel image size")
    seed_everything(seed)
    weights = FCOS_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    hub_dir = Path(cache_dir or DEFAULT_CACHE).resolve()
    previous_hub_dir = torch.hub.get_dir()
    try:
        if pretrained:
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))
        model = fcos_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=5,
            min_size=image_size,
            max_size=image_size,
            score_thresh=0.001,
            detections_per_img=100,
        )
    finally:
        torch.hub.set_dir(previous_hub_dir)

    # Preserve the pretrained classification tower. Only its category-specific
    # output convolution must be reinitialized for the new label set.
    classification = model.head.classification_head
    old_logits = classification.cls_logits
    logits = nn.Conv2d(old_logits.in_channels, num_classes, kernel_size=3, padding=1)
    nn.init.normal_(logits.weight, std=0.01)
    nn.init.constant_(logits.bias, -math.log(99.0))
    classification.cls_logits = logits
    classification.num_classes = num_classes

    if len(model.backbone.fpn.layer_blocks) != 3:
        raise RuntimeError("This contract requires the official three P3/P4/P5 FPN blocks")
    for index, option in enumerate(combo):
        conv = model.backbone.fpn.layer_blocks[index][0]
        if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3):
            raise RuntimeError(f"Unexpected official FPN block {index}: {conv}")
        dilation = 1 if option == "A" else 2
        conv.dilation = (dilation, dilation)
        conv.padding = (dilation, dilation)

    checkpoint = hub_dir / "checkpoints" / Path(weights.url).name if weights else None
    model._sonar_pilot_contract = {
        "combo": combo,
        "seed": seed,
        "num_classes_including_background": num_classes,
        "foreground_label_range": [1, num_classes - 1],
        "image_size": image_size,
        "pretrained": {
            "enabled": pretrained,
            "name": "FCOS_ResNet50_FPN_Weights.COCO_V1" if weights else None,
            "source_url": weights.url if weights else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            "classification_logits": "reinitialized for foreground classes plus background",
        },
    }
    return model


def model_contract(model):
    """JSON-serializable structure/provenance for a formal run manifest."""
    metadata = dict(model._sonar_pilot_contract)
    metadata["positions"] = [
        {
            "level": f"P{index + 3}",
            "module": f"backbone.fpn.layer_blocks.{index}.0",
            "option": metadata["combo"][index],
            "kernel": list(conv.kernel_size),
            "dilation": list(conv.dilation),
            "padding": list(conv.padding),
            "weight_shape": list(conv.weight.shape),
            "feature_shape_at_640": [1, conv.out_channels, 640 // (2 ** (index + 3)), 640 // (2 ** (index + 3))],
        }
        for index, block in enumerate(model.backbone.fpn.layer_blocks)
        for conv in [block[0]]
    ]
    metadata.update({
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "all_backbone_convolutions_trainable": all(
            parameter.requires_grad
            for module in model.backbone.body.modules()
            if isinstance(module, nn.Conv2d)
            for parameter in module.parameters()
        ),
        "unchanged_extra_levels": ["P6", "P7"],
        "extra_levels_shape_at_640": [[1, 256, 10, 10], [1, 256, 5, 5]],
        "score_threshold": model.score_thresh,
        "detections_per_image": model.detections_per_img,
        "transform_min_size": list(model.transform.min_size),
        "transform_max_size": model.transform.max_size,
        "backbone_batch_norm": sorted({type(module).__name__ for module in model.backbone.body.modules() if "BatchNorm" in type(module).__name__}),
        "activation_checkpointing": False,
    })
    return metadata


def state_digest(model):
    """Digest tensors, excluding the intentional non-tensor A/B attributes."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dilation_witness(device="cpu"):
    """An impulse verifies that A and B actually sample different locations."""
    conv_a = nn.Conv2d(1, 1, 3, padding=1, bias=False).to(device)
    conv_b = nn.Conv2d(1, 1, 3, padding=2, dilation=2, bias=False).to(device)
    with torch.no_grad():
        conv_a.weight.fill_(1)
        conv_b.weight.copy_(conv_a.weight)
    sample = torch.zeros(1, 1, 9, 9, device=device)
    sample[0, 0, 4, 4] = 1
    sample.requires_grad_(True)
    output_a, output_b = conv_a(sample), conv_b(sample)
    difference = (output_a - output_b).abs().sum()
    gradient_a, = torch.autograd.grad(output_a[0, 0, 4, 4], sample, retain_graph=True)
    gradient_b, = torch.autograd.grad(output_b[0, 0, 4, 4], sample)
    return {
        "device": str(device),
        "shape": list(output_a.shape),
        "output_difference_l1": float(difference.detach()),
        "input_gradient_difference_l1": float((gradient_a - gradient_b).abs().sum()),
        "finite": bool(torch.isfinite(output_a).all() and torch.isfinite(output_b).all()),
    }


def initialization_audit(cache_dir=None, pretrained=True):
    """Compare all candidates, and AAA against an independent official FCOS."""
    torch.set_num_threads(4)
    rows = []
    baseline = None
    for combo in COMBINATIONS:
        candidate = build_model(combo, 11, 42, pretrained=pretrained, cache_dir=cache_dir)
        rows.append({"combo": combo, "state_sha256": state_digest(candidate), "parameter_count": model_contract(candidate)["parameter_count"]})
        if baseline is None:
            baseline = candidate
        else:
            del candidate
    expected = rows[0]
    parity = all(row["state_sha256"] == expected["state_sha256"] and row["parameter_count"] == expected["parameter_count"] for row in rows)

    # Build from torchvision directly rather than from this module. A new class
    # output layer is the only adaptation required in both models.
    weights = FCOS_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    old_hub = torch.hub.get_dir()
    seed_everything(42)
    try:
        if pretrained:
            torch.hub.set_dir(str(Path(cache_dir or DEFAULT_CACHE).resolve()))
        reference = fcos_resnet50_fpn(weights=weights, weights_backbone=None, trainable_backbone_layers=5, min_size=640, max_size=640, score_thresh=0.001, detections_per_img=100)
    finally:
        torch.hub.set_dir(old_hub)
    reference.head.classification_head.cls_logits = copy.deepcopy(baseline.head.classification_head.cls_logits)
    reference.head.classification_head.num_classes = 11
    baseline.eval()
    reference.eval()
    seed_everything(19)
    image = torch.rand(3, 640, 640)
    with torch.no_grad():
        actual = baseline([image])[0]
        expected_output = reference([image])[0]
    differences = {name: float((actual[name] - expected_output[name]).abs().max()) if actual[name].numel() else 0.0 for name in actual}
    official_parity = all(torch.equal(actual[name], expected_output[name]) for name in actual)
    return {
        "status": "PASS" if parity and official_parity else "FAIL",
        "pretrained": pretrained,
        "candidates": rows,
        "all_initial_tensor_states_identical": parity,
        "AAA_matches_independent_official_eval": official_parity,
        "official_eval_max_absolute_differences": differences,
        "official_eval_detection_count": len(actual["boxes"]),
    }


def cuda_preflight(output_dir, cache_dir=None, pretrained=True):
    """Synthetic kernel check, not an epoch, accuracy result or formal run."""
    output_dir = Path(output_dir)
    report = {"environment": environment(), "kind": "synthetic_forward_backward_preflight", "formal_training": False}
    if not torch.cuda.is_available():
        report["status"] = "NO_CUDA"
        write_json(output_dir / "preflight.json", report)
        return report
    torch.set_num_threads(4)
    report["kernel_witness"] = dilation_witness("cuda")
    free, total = torch.cuda.mem_get_info()
    report["cuda_free_bytes_at_start"] = free
    report["cuda_total_bytes"] = total
    report["runs"] = []
    for combo in ["AAA", "BBB"]:
        model = None
        optimizer = None
        losses = loss = image = target = scaler = None
        start = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run = {"combo": combo, "batch_size": 1, "image_size": 640, "amp": "float16"}
        try:
            model = build_model(combo, 11, 42, pretrained=pretrained, cache_dir=cache_dir).cuda().train()
            run["contract"] = model_contract(model)
            image = torch.rand(3, 640, 640, device="cuda")
            target = {"boxes": torch.tensor([[160., 150., 230., 240.], [350., 350., 510., 520.]], device="cuda"), "labels": torch.tensor([1, 2], device="cuda")}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scaler = torch.amp.GradScaler("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                losses = model([image], [target])
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite synthetic loss: {losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = {
                f"P{index + 3}": float(block[0].weight.grad.detach().abs().sum())
                for index, block in enumerate(model.backbone.fpn.layer_blocks)
            }
            if not all(math.isfinite(value) and value > 0 for value in gradients.values()):
                raise RuntimeError(f"Invalid FPN gradients: {gradients}")
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            run.update({"status": "PASS", "losses": {key: float(value.detach()) for key, value in losses.items()}, "fpn_gradient_l1": gradients})
        except torch.OutOfMemoryError as exc:
            run.update({"status": "CUDA_OOM", "error": str(exc)})
        except Exception as exc:
            run.update({"status": "FAIL", "error": repr(exc)})
        finally:
            run["elapsed_seconds"] = time.perf_counter() - start
            run["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            run["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["runs"].append(run)
            del model, optimizer
            losses = loss = image = target = scaler = None
            gc.collect()
            torch.cuda.empty_cache()
            write_json(output_dir / "preflight.json", report)
        if run["status"] != "PASS":
            break
    report["status"] = "PASS" if len(report["runs"]) == 2 and all(run["status"] == "PASS" for run in report["runs"]) else "FAIL"
    write_json(output_dir / "preflight.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCOS 640-pixel synthetic GPU preflight")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "sonar_fcos_pilot_20261008" / "model_preflight")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    report = cuda_preflight(args.output_dir, args.cache_dir, pretrained=not args.no_pretrained)
    audit = initialization_audit(args.cache_dir, pretrained=not args.no_pretrained)
    write_json(args.output_dir / "initialization_audit.json", audit)
    print("FCOS_KERNEL_WITNESS", report.get("kernel_witness"))
    print("FCOS_PREFLIGHT", report["status"])
    print("FCOS_INITIALIZATION_AUDIT", audit["status"])
    raise SystemExit(0 if report["status"] == "PASS" and audit["status"] == "PASS" else 1)
