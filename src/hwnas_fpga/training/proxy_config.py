"""Explicit configuration for the independent, inexpensive RL evaluator."""
from typing import Any, Mapping
import math


def resolve_proxy_training(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = {
        "engine": "independent_simple_v1", "optimizer": "adamw",
        "lr": 0.001, "weight_decay": 0.0001,
        "loss": "class_weighted_cross_entropy", "scheduler": "none",
        "amp": False, "gradient_accumulation_steps": 1,
        "warmup_epochs": 0, "label_smoothing": 0.0, "logit_adjust_tau": 0.0,
    }
    supplied = dict(config or {})
    unknown = set(supplied) - set(resolved) - {"batch_size"}
    if unknown:
        raise ValueError(f"unsupported search.proxy_training fields: {sorted(unknown)}")
    for name, value in supplied.items():
        if name not in {"lr", "weight_decay", "batch_size"} and value != resolved[name]:
            raise ValueError(f"independent_simple_v1 does not support proxy setting {name}={value!r}")
        resolved[name] = value
    for name in ("lr", "weight_decay"):
        value = float(resolved[name])
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"invalid proxy {name}")
        resolved[name] = value
    if "batch_size" in resolved:
        size = resolved["batch_size"]
        if isinstance(size, bool) or int(size) != size or size <= 0:
            raise ValueError("proxy batch_size must be a positive integer")
        resolved["batch_size"] = int(size)
    return resolved
