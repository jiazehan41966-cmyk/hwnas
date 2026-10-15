"""Thirty-epoch, resumable training; test data never enters this module."""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import environment, read_json, require_manifest, resolve_path, seed_everything, sha256_file, source_binding, utc_now, write_json
from .data import SonarDataset, predictions_to_original
from .metrics import calibrate_threshold, evaluate_predictions, write_error_examples
from .model import build_model, model_contract
from .numerics import AccumulationGroup, numerical_policy, restore_rng as _restore_rng, rng_state as _rng_state


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def loader(dataset, config, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["workers"], collate_fn=collate,
                      generator=generator, pin_memory=config["device"].startswith("cuda"))


def run_id(combo, seed):
    return f"{combo}_seed{seed}"


def run_directory(config, combo, seed):
    return resolve_path(config["output_dir"]) / "runs" / run_id(combo, seed)


def learning_rate(config, epoch, step, steps):
    drops = sum(epoch >= boundary for boundary in config["learning_rate_drop_epochs"])
    rate = config["learning_rate"] * 0.1 ** drops
    warmup = config["warmup_epochs"] * steps
    position = epoch * steps + step
    if position < warmup:
        factor = config["warmup_start_factor"] + (1 - config["warmup_start_factor"]) * position / max(1, warmup)
        rate *= factor
    return rate


@torch.inference_mode()
def predict(model, dataset, config):
    model.eval()
    predictions = []
    device = torch.device(config["device"])
    use_amp = config["amp"] and device.type == "cuda"
    for images, targets in loader(dataset, config):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model([image.to(device, non_blocking=True) for image in images])
        predictions.extend(predictions_to_original(target, output) for target, output in zip(targets, outputs))
    return predictions


def _save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_one(config, combo, seed):
    """Resume only a matching run; completed results are reused, never replaced."""
    manifest_path, manifest = require_manifest(config)
    if combo not in config["combinations"] or seed not in config["seeds"]:
        raise ValueError("Run is outside the approved combination/seed budget.")
    torch.set_num_threads(config["torch_threads"])
    seed_everything(seed)
    directory = run_directory(config, combo, seed)
    current_environment = environment()
    binding = {**source_binding(config), "data_manifest_sha256": sha256_file(manifest_path),
               "combo": combo, "seed": seed, "environment": current_environment}
    # Verify raw inputs before returning a cached result, as well as before training.
    for sample in manifest["samples"]:
        if sample["split"] in {"train", "val"} and sha256_file(sample["image_path"]) != sample["sha256"]:
            raise RuntimeError("Raw development image changed: " + sample["image_path"])
    summary_path = directory / "summary.json"
    if summary_path.exists():
        result = read_json(summary_path)
        if result["binding"] != binding:
            raise RuntimeError("Completed run has different source/config/data; preserve it and use a new campaign.")
        for artifact in result["best_artifacts"].values():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("Completed run artifact changed; refusing to reuse an outdated summary.")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "run_manifest.json"
    if receipt_path.exists() and read_json(receipt_path)["binding"] != binding:
        raise RuntimeError("Interrupted run binding changed; unsafe to resume.")
    train_data = SonarDataset(manifest_path, "train", config["image_size"])
    val_data = SonarDataset(manifest_path, "val", config["image_size"])
    model = build_model(combo, len(manifest["classes"]) + 1, seed,
                        pretrained=config["pretrained"], image_size=config["image_size"],
                        cache_dir=resolve_path(config["weight_cache"]))
    device = torch.device(config["device"])
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=config["learning_rate"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["amp_initial_scale"], enabled=config["amp"] and device.type == "cuda")
    checkpoint_dir = resolve_path(config["checkpoint_root"]) / run_id(combo, seed)
    last_path = checkpoint_dir / "last.pt"
    history, best_map, start_epoch, best_artifacts = [], -math.inf, 0, None
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["binding"] != binding:
            raise RuntimeError("Checkpoint does not belong to this source/config/data.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scaler.load_state_dict(saved["scaler"])
        history, best_map, start_epoch = saved["history"], saved["best_map"], saved["epoch"] + 1
        best_artifacts = saved["best_artifacts"]
        for artifact in best_artifacts.values():
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("Committed validation/checkpoint artifact changed; cannot safely resume.")
        _restore_rng(saved["rng"])
    receipt = {"run_id": run_id(combo, seed), "combo": combo, "seed": seed,
               "binding": binding, "environment": current_environment, "model": model_contract(model),
               "epochs": config["epochs"], "training_images": len(train_data), "validation_images": len(val_data),
               "test_used": False, "created_at": utc_now(), "effective_batch_size": config["batch_size"] * config["gradient_accumulation"],
               "numerical_policy": numerical_policy()}
    if not receipt_path.exists():
        started_runs = list((resolve_path(config["output_dir"]) / "runs").glob("*/run_manifest.json"))
        if len(started_runs) >= config["max_formal_runs"]:
            raise RuntimeError("The fourteen-run budget is already committed.")
        write_json(receipt_path, receipt)
    elif read_json(receipt_path)["model"] != model_contract(model):
        raise RuntimeError("Model/pretraining provenance changed since this run began.")
    numerical_events_path = directory / "numerical_events.json"
    numerical_events = read_json(numerical_events_path) if numerical_events_path.exists() else []

    def record_numerical_event(event):
        numerical_events.append({"run_id": run_id(combo, seed), **event})
        write_json(numerical_events_path, numerical_events)
        print(f"{run_id(combo, seed)} numerical_event={event['event']} "
              f"epoch={event['epoch']} group={event['group']} "
              f"scale={event.get('failed_scale', event.get('successful_scale', event.get('scale')))} "
              f"retry={event.get('retry', event.get('retries', 0))}", flush=True)

    try:
        for epoch in range(start_epoch, config["epochs"]):
            started = time.monotonic()
            model.train()
            data_loader = loader(train_data, config, shuffle=True, seed=seed + epoch * 100003)
            steps = len(data_loader)
            accumulation = config["gradient_accumulation"]
            optimizer.zero_grad(set_to_none=True)
            total_loss, updates, overflow_retries = 0., 0, 0
            recovery_group = None
            for step, (images, targets) in enumerate(data_loader):
                rate = learning_rate(config, epoch, step, steps)
                group_start = (step // accumulation) * accumulation
                divisor = min(accumulation, steps - group_start)
                if recovery_group is None:
                    recovery_group = AccumulationGroup(model, optimizer, scaler, device, divisor,
                                                       epoch=epoch + 1, group_index=step // accumulation + 1,
                                                       use_amp=config["amp"] and device.type == "cuda",
                                                       event_callback=record_numerical_event)
                loss_value = recovery_group.backward(images, targets, step=step, learning_rate=rate)
                if (step + 1) % accumulation == 0 or step + 1 == steps:
                    completed_group = recovery_group.finish()
                    for value in completed_group["loss_values"]:
                        total_loss += value
                    loss_value = completed_group["loss_values"][-1]
                    overflow_retries += completed_group["retries"]
                    updates += completed_group["optimizer_steps"]
                    recovery_group = None
                if step % 100 == 0 or step + 1 == steps:
                    write_json(directory / "status.json", {"status": "TRAINING", "epoch": epoch + 1,
                               "step": step + 1, "steps": steps, "loss": loss_value, "optimizer_steps": updates,
                               "overflow_retries": overflow_retries, "amp_scale": float(scaler.get_scale()),
                               "updated_at": utc_now()})
            expected_updates = math.ceil(steps / accumulation)
            if updates != expected_updates:
                raise RuntimeError(f"Completed epoch has {updates} optimizer updates; expected {expected_updates}.")
            predictions = predict(model, val_data, config)
            metrics = evaluate_predictions(val_data.records, predictions, len(manifest["classes"]),
                                           manifest["small_area_threshold"], config["diagnostic_score_threshold"])
            if metrics["map"] is None:
                raise RuntimeError("Validation mAP is undefined; cannot select a checkpoint.")
            improved = metrics["map"] > best_map
            if improved:
                best_map = metrics["map"]
            history.append({"epoch": epoch + 1, "loss": total_loss / steps, "validation_map": metrics["map"],
                            "learning_rate": rate, "optimizer_steps": updates, "overflow_retries": overflow_retries,
                            "amp_scale": float(scaler.get_scale()), "seconds": time.monotonic() - started})
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            # Epoch-addressed best artifacts are written before the resume commit.
            # An interruption cannot replace the best referenced by the older last.pt.
            if improved:
                best_path = checkpoint_dir / f"best_epoch_{epoch + 1:03d}.pt"
                prediction_path = directory / "validation" / f"epoch_{epoch + 1:03d}_predictions.json"
                metric_path = directory / "validation" / f"epoch_{epoch + 1:03d}_metrics.json"
                _save_checkpoint(best_path, {"model": state, "binding": binding, "epoch": epoch})
                write_json(prediction_path, predictions)
                write_json(metric_path, metrics)
                best_artifacts = {name: {"path": str(path), "sha256": sha256_file(path)}
                                  for name, path in [("checkpoint", best_path), ("predictions", prediction_path), ("metrics", metric_path)]}
            payload = {"model": state,
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(),
                       "history": history, "best_map": best_map, "epoch": epoch, "binding": binding,
                       "best_artifacts": best_artifacts}
            _save_checkpoint(last_path, payload)
            write_json(directory / "history.json", history)
            print(f"{run_id(combo, seed)} epoch={epoch+1}/30 loss={total_loss/steps:.5f} validation_mAP={metrics['map']:.3f}", flush=True)
        best_path = Path(best_artifacts["checkpoint"]["path"])
        predictions = read_json(best_artifacts["predictions"]["path"])
        calibration = calibrate_threshold(val_data.records, predictions, len(manifest["classes"]),
                                          manifest["small_area_threshold"], config["validation_fp_budget_per_image"])
        metrics = read_json(best_artifacts["metrics"]["path"])
        examples = write_error_examples(val_data.records, predictions, directory / "error_examples",
                                        manifest["small_area_threshold"], config["diagnostic_score_threshold"])
        result = {**receipt, "status": "COMPLETE", "completed_at": utc_now(), "binding": binding,
                  "validation": metrics, "validation_calibration": calibration,
                  "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256_file(best_path),
                  "best_artifacts": best_artifacts,
                  "trained_epochs": len(history), "error_examples": examples}
        write_json(summary_path, result)
        write_json(directory / "status.json", {"status": "COMPLETE", "trained_epochs": len(history), "updated_at": utc_now()})
        return result
    except BaseException as error:
        write_json(directory / "status.json", {"status": "ERROR", "error": str(error), "updated_at": utc_now(), "resume_from_last_complete_epoch": last_path.exists()})
        raise
    finally:
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
