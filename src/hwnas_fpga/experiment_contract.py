"""Versioned experiment contracts shared by selection and formal evaluation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


DATASET_CONTRACT_FIELDS = (
    "image_size",
    "input_channels",
    "geometry_mode",
    "fixed_scale_factor",
    "geometry_padding_value",
    "augmentation_profile",
    "inner_val_fraction",
)

TRAINING_CONTRACT_FIELDS = (
    "epochs",
    "optimizer",
    "lr",
    "weight_decay",
    "scheduler",
    "warmup_epochs",
    "min_lr_ratio",
    "label_smoothing",
    "logit_adjust_tau",
    "batch_size",
    "gradient_accumulation_steps",
    "amp",
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_experiment_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    contract_path = Path(path).expanduser()
    if not contract_path.is_absolute():
        contract_path = (Path.cwd() / contract_path).resolve()
    else:
        contract_path = contract_path.resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"experiment contract does not exist: {contract_path}")
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"experiment contract must contain a mapping: {contract_path}")
    if not payload.get("protocol"):
        raise ValueError(f"experiment contract is missing protocol: {contract_path}")
    if not isinstance(payload.get("dataset"), dict):
        raise ValueError(f"experiment contract is missing dataset mapping: {contract_path}")
    if not isinstance(payload.get("training_recipe"), dict):
        raise ValueError(
            f"experiment contract is missing training_recipe mapping: {contract_path}"
        )
    return contract_path, payload


def experiment_contract_provenance(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": canonical_sha256(payload),
        "protocol": payload.get("protocol"),
    }


def _validate_section(
    *,
    contract: Mapping[str, Any],
    observed: Mapping[str, Any],
    fields: tuple[str, ...],
    section: str,
) -> None:
    for field in fields:
        if field not in contract:
            raise ValueError(f"experiment contract is missing {section}.{field}")
        expected = contract[field]
        actual = observed.get(field)
        if actual != expected:
            raise ValueError(
                f"{section}.{field} conflicts with the experiment contract: "
                f"{actual!r} != {expected!r}"
            )


def validate_formal_values_against_contract(
    *,
    contract: Mapping[str, Any],
    observed_dataset: Mapping[str, Any],
    observed_recipe: Mapping[str, Any],
) -> None:
    _validate_section(
        contract=contract["dataset"],
        observed=observed_dataset,
        fields=DATASET_CONTRACT_FIELDS,
        section="dataset",
    )
    _validate_section(
        contract=contract["training_recipe"],
        observed=observed_recipe,
        fields=TRAINING_CONTRACT_FIELDS,
        section="training_recipe",
    )


def bind_contract_sections(
    config: Mapping[str, Any],
    contract_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind preprocessing and recipe fields without changing search budgets."""

    path, contract = load_experiment_contract(contract_path)
    resolved = deepcopy(dict(config))
    dataset = resolved.setdefault("dataset", {})
    training = resolved.setdefault("training", {})
    for field in DATASET_CONTRACT_FIELDS:
        value = contract["dataset"][field]
        if field in dataset and dataset[field] != value:
            raise ValueError(
                f"dataset.{field} conflicts with experiment contract: "
                f"{dataset[field]!r} != {value!r}"
            )
        dataset[field] = deepcopy(value)
    for field in TRAINING_CONTRACT_FIELDS:
        value = contract["training_recipe"][field]
        if field in training and training[field] != value:
            raise ValueError(
                f"training.{field} conflicts with experiment contract: "
                f"{training[field]!r} != {value!r}"
            )
        training[field] = deepcopy(value)
    dataset["split_mode"] = "frozen_inner"
    resolved["experiment_contract"] = experiment_contract_provenance(path, contract)
    return resolved, contract
