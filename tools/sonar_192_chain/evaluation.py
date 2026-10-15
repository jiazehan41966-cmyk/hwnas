"""Source-bound full-valid replay for a selected EMA and saved PTQ exports."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from .contract import digest, sha256_file
from .host_decode import decode_raw, native_predictions_from_raw


def load_fake_export(candidate, manifest_path):
    """Validate all exported bytes, then reconstruct the independent fake runtime."""
    from .handoff import inspect_export
    from .quantization import FakeReference
    checked = inspect_export(candidate, manifest_path)
    parameters = {}
    for row in checked["parameter_index"]["parameters"]:
        if row["kind"] != "lut":
            parameters.setdefault(row["node"], {})[row["kind"]] = checked["arrays"][row["key"]]
    return FakeReference(checked["graph"], parameters).eval(), checked


def frozen_validation_loader(execution_manifest, checkpoint_binding, dataset_root):
    """Verify all 99 valid image/label bytes and derive the immutable GT contract."""
    from tools.sonar_yolov9_baseline import protocol as p
    from .confirmation import validate_validation_contract
    path, dataset_root = Path(execution_manifest).resolve(), Path(dataset_root).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(path) != checkpoint_binding["binding"]["execution_manifest_sha256"]:
        raise ValueError("Validation manifest differs from selected checkpoint")
    baseline = p.load_baseline()
    registered_data = (p.ROOT / baseline["dataset"]["data_dir"]).resolve()
    if dataset_root != registered_data:
        raise ValueError("Validation dataset root differs from frozen loader")
    loader = p.frozen.make_loader(baseline, dataset_root, "valid", checkpoint_binding["seed"], 0)
    actual = p.frozen.bind_dataset(loader)
    actual["entries"] = p._normalize_entries(actual["entries"])
    actual["sha256"] = digest(actual["entries"])
    expected = manifest["inputs"]["datasets"]["valid"]
    if actual != expected or actual["images"] != 99:
        raise ValueError("Validation image/annotation inventory differs")
    normalized = [{"image": "valid/images/" + row["image"], "target": deepcopy(row["target"])}
                  for row in manifest["validation_targets"]]
    protocol = dict(schema_version="sonar192_confirmation_v1", seeds=[42, 43, 44],
                    profiles=["W8A8", "HEAD_W8A16", "HEAD_W16A16"],
                    metric="coco_101point_AP50_95_max100_per_class", selection_policy="epoch80_ema",
                    validation_count=99, validation_images_sha256=digest(sorted(row["image"] for row in normalized)),
                    validation_targets_sha256=digest(sorted(normalized, key=lambda row: row["image"])),
                    num_classes=2, absolute_drop=.003, relative_drop=.20, improved_seeds=2,
                    evidence_kind="real_validation")
    validate_validation_contract(protocol, path, checkpoint_binding)
    # Batch1 reduces the retained full FP64 graph's memory; preprocessing and
    # input order remain exactly the frozen dataset and collate function.
    single = torch.utils.data.DataLoader(loader.dataset, batch_size=1, shuffle=False,
                                         num_workers=0, collate_fn=loader.collate_fn)
    return single, protocol


@torch.no_grad()
def evaluate_records(model, loader, dataset_root, *, raw_source=None, limit=None, progress=None):
    """Produce the confirmation reader's record schema using the same host path.

    ``raw_source=None`` evaluates native FP32 EMA and verifies the raw adapter
    against its native decoded tensor. A quantized source is the independent
    FP64 fake graph; this function makes no whole-valid integer parity claim.
    """
    from tools.sonar_yolov9_baseline import protocol as p
    if model.training or (raw_source is not None and raw_source.training):
        raise ValueError("Evaluation requires inference mode")
    records, maximum, batches = [], 0.0, 0
    for images, targets in loader:
        if limit is not None and len(records) >= limit:
            break
        if raw_source is None:
            native = model(images)["predictions"]
            decoded, raw = native
            replayed = native_predictions_from_raw(model, raw)[0]
            maximum = max(maximum, float((decoded - replayed).abs().max()))
            if not torch.equal(decoded, replayed):
                raise ValueError("Host raw decode differs from native direct-head output")
        else:
            raw = raw_source(images)["raw_predictions"]
        predictions = decode_raw(model, raw)
        records.extend(p.records_from_batch(predictions, targets, dataset_root))
        batches += 1
        if progress is not None:
            progress(len(records))
    return records, dict(images=len(records), batches=batches,
                         native_raw_decode_compared=raw_source is None,
                         native_raw_decode_max_abs_error=maximum if raw_source is None else None,
                         runtime="native_FP32" if raw_source is None else raw_source.execution_kind,
                         integer_full_valid_replayed=False, test_evaluated=False)
