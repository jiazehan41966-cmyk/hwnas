"""Deterministic sonar-classification robustness evaluation.

The protocol perturbs only the inner-validation inputs and keeps their class
labels unchanged.  It measures classification robustness; it is not an image
restoration or PSNR/SSIM protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .trainer import evaluate_classifier


DEFAULT_SONAR_ROBUSTNESS_CONDITIONS: tuple[dict[str, Any], ...] = (
    {"name": "speckle_var_0p01", "kind": "speckle", "variance": 0.01},
    {"name": "speckle_var_0p04", "kind": "speckle", "variance": 0.04},
    {"name": "contrast_0p70", "kind": "contrast", "factor": 0.70},
    {"name": "blur_3x3", "kind": "blur", "kernel_size": 3},
)


def resolve_sonar_robustness_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and canonicalize the frozen robustness protocol."""

    raw = dict(config or {})
    enabled = bool(raw.get("enabled", False))
    conditions_raw = raw.get("conditions", DEFAULT_SONAR_ROBUSTNESS_CONDITIONS)
    if not isinstance(conditions_raw, Sequence) or isinstance(conditions_raw, (str, bytes)):
        raise ValueError("robustness.conditions must be a non-empty sequence")

    conditions: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(conditions_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"robustness condition {index} must be a mapping")
        condition = dict(item)
        name = str(condition.get("name") or f"condition_{index}").strip()
        kind = str(condition.get("kind") or "").strip().lower()
        if not name or name in names:
            raise ValueError("robustness condition names must be non-empty and unique")
        if kind not in {"speckle", "contrast", "blur"}:
            raise ValueError(f"unsupported sonar robustness condition kind: {kind!r}")
        names.add(name)
        normalized: dict[str, Any] = {"name": name, "kind": kind}
        if kind == "speckle":
            variance = float(condition.get("variance", 0.04))
            if not math.isfinite(variance) or variance <= 0.0:
                raise ValueError("speckle variance must be positive and finite")
            normalized["variance"] = variance
        elif kind == "contrast":
            factor = float(condition.get("factor", 0.70))
            if not math.isfinite(factor) or factor <= 0.0:
                raise ValueError("contrast factor must be positive and finite")
            normalized["factor"] = factor
        else:
            kernel_size = int(condition.get("kernel_size", 3))
            if kernel_size <= 1 or kernel_size % 2 == 0:
                raise ValueError("blur kernel_size must be an odd integer greater than one")
            normalized["kernel_size"] = kernel_size
        conditions.append(normalized)

    if enabled and not conditions:
        raise ValueError("enabled robustness evaluation requires at least one condition")

    aggregation = str(raw.get("aggregation", "mean_macro_f1")).strip().lower()
    if aggregation != "mean_macro_f1":
        raise ValueError("robustness aggregation must be 'mean_macro_f1'")
    mean = [float(value) for value in raw.get("normalization_mean", [0.5])]
    std = [float(value) for value in raw.get("normalization_std", [0.5])]
    if not mean or len(mean) != len(std) or any(value <= 0.0 for value in std):
        raise ValueError("robustness normalization mean/std must have equal non-zero length")

    resolved = {
        "schema_version": 1,
        "protocol": str(raw.get("protocol", "sonar_corruption_v1")),
        "enabled": enabled,
        "seed": int(raw.get("seed", 314159)),
        "aggregation": aggregation,
        "normalization_mean": mean,
        "normalization_std": std,
        "conditions": conditions,
        "claim_boundary": (
            "Inner-validation classification robustness under deterministic synthetic "
            "sonar perturbations; not clean-reference image-restoration quality."
        ),
    }
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    resolved["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return resolved


def _channel_tensor(values: Sequence[float], inputs: torch.Tensor) -> torch.Tensor:
    if len(values) == 1:
        values = list(values) * int(inputs.shape[1])
    if len(values) != int(inputs.shape[1]):
        raise ValueError(
            "robustness normalization channel count does not match the input tensor"
        )
    return torch.tensor(values, dtype=inputs.dtype, device=inputs.device).view(1, -1, 1, 1)


def apply_sonar_corruption(
    inputs: torch.Tensor,
    *,
    condition: Mapping[str, Any],
    seed: int,
    batch_index: int,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
) -> torch.Tensor:
    """Apply one deterministic corruption in de-normalized intensity space."""

    if inputs.ndim != 4:
        raise ValueError("sonar robustness inputs must have shape [N, C, H, W]")
    mean = _channel_tensor(normalization_mean, inputs)
    std = _channel_tensor(normalization_std, inputs)
    intensity = torch.clamp(inputs * std + mean, 0.0, 1.0)
    kind = str(condition["kind"])

    if kind == "speckle":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(batch_index))
        noise = torch.randn(
            intensity.shape,
            generator=generator,
            dtype=intensity.dtype,
            device="cpu",
        ).to(intensity.device)
        intensity = intensity + intensity * noise * math.sqrt(float(condition["variance"]))
    elif kind == "contrast":
        intensity = (intensity - 0.5) * float(condition["factor"]) + 0.5
    elif kind == "blur":
        kernel_size = int(condition["kernel_size"])
        intensity = F.avg_pool2d(
            intensity,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
    else:  # pragma: no cover - validated by resolve_sonar_robustness_config
        raise ValueError(f"unsupported sonar corruption kind: {kind!r}")

    intensity = torch.clamp(intensity, 0.0, 1.0)
    return (intensity - mean) / std


@torch.no_grad()
def evaluate_sonar_robustness(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: str,
    num_classes: int,
    class_weights: torch.Tensor | None,
    config: Mapping[str, Any] | None,
    topk: int = 5,
) -> dict[str, Any]:
    """Evaluate a restored clean-selected model on the fixed corruption suite."""

    resolved = resolve_sonar_robustness_config(config)
    if not resolved["enabled"]:
        return {**resolved, "status": "disabled", "condition_results": []}

    weights = None if class_weights is None else class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    condition_results: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(resolved["conditions"]):
        condition_seed = int(resolved["seed"]) + condition_index * 1_000_003

        def transform(inputs: torch.Tensor, batch_index: int) -> torch.Tensor:
            return apply_sonar_corruption(
                inputs,
                condition=condition,
                seed=condition_seed,
                batch_index=batch_index,
                normalization_mean=resolved["normalization_mean"],
                normalization_std=resolved["normalization_std"],
            )

        summary = evaluate_classifier(
            model,
            data_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            topk=topk,
            input_transform=transform,
        )
        condition_results.append(
            {
                "name": condition["name"],
                "kind": condition["kind"],
                "parameters": {
                    key: value
                    for key, value in condition.items()
                    if key not in {"name", "kind"}
                },
                "seed": condition_seed,
                "metrics": summary,
            }
        )

    macro_f1_values = [
        float(result["metrics"]["macro_f1"]) for result in condition_results
    ]
    return {
        **resolved,
        "status": "complete",
        "condition_results": condition_results,
        "f_robust": sum(macro_f1_values) / len(macro_f1_values),
        "robust_worst_macro_f1": min(macro_f1_values),
    }
