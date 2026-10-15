"""Replay an overflowing AMP accumulation group without losing an update.

Only overflow attempts are replayed. A successful first attempt follows the
ordinary GradScaler unscale/step/update path without restoring RNG or buffers.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from .common import utc_now, write_json


MAX_OVERFLOW_RETRIES = 8


def numerical_policy():
    return {"version": "AMP_GROUP_REPLAY_V1", "maximum_overflow_retries": MAX_OVERFLOW_RETRIES,
            "overflow_action": "official_scaler_backoff_and_replay_same_group",
            "failed_attempts_update_parameters": False, "discard_training_batches": False,
            "float32_diagnostic_on_exhaustion": True}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _finite_gradients(optimizer):
    return all(bool(torch.isfinite(parameter.grad).all())
               for group in optimizer.param_groups for parameter in group["params"]
               if parameter.grad is not None)


def _image_id(target):
    value = target.get("image_id")
    if value is None:
        return None  # Synthetic fixtures may omit an image ID; real data has it.
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0])
    return int(value)


class AccumulationGroup:
    """One optimizer update, with a bounded replay of its retained minibatches.

    A minibatch is submitted at the original point in the DataLoader loop. Its
    RNG snapshot is taken after data loading and immediately before the model,
    so even a stochastic dataset does not change the original normal path.
    """

    def __init__(self, model, optimizer, scaler, device, divisor, *, epoch,
                 group_index, use_amp, event_callback=None,
                 max_retries=MAX_OVERFLOW_RETRIES):
        self.model, self.optimizer, self.scaler = model, optimizer, scaler
        self.device = torch.device(device)
        self.divisor, self.epoch, self.group_index = divisor, epoch, group_index
        self.use_amp, self.event_callback = use_amp, event_callback
        self.max_retries = max_retries
        self.batches, self.loss_values, self.events = [], [], []
        self.buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        self.initial_scale = float(scaler.get_scale())
        self.original_end_rng = None

    def _context(self):
        return {"epoch": self.epoch, "group": self.group_index,
                "steps": [batch["step"] + 1 for batch in self.batches],
                "image_ids": [image_id for batch in self.batches for image_id in batch["image_ids"]]}

    def _event(self, kind, **details):
        event = {"event": kind, **self._context(), **details, "recorded_at": utc_now()}
        self.events.append(event)
        if self.event_callback is not None:
            self.event_callback(event)

    def _restore_buffers(self):
        current = dict(self.model.named_buffers())
        if current.keys() != self.buffers.keys():
            raise RuntimeError("Model buffer registration changed inside an accumulation group.")
        with torch.no_grad():
            for name, value in self.buffers.items():
                current[name].copy_(value)

    def _backward(self, batch, *, mixed_precision, scaled):
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = batch["learning_rate"]
        images = [image.to(self.device, non_blocking=True) for image in batch["images"]]
        targets = [{key: value.to(self.device) for key, value in target.items()}
                   for target in batch["targets"]]
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=mixed_precision):
            loss = sum(self.model(images, targets).values())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training loss at epoch {self.epoch}, step {batch['step'] + 1}.")
        if scaled:
            self.scaler.scale(loss / self.divisor).backward()
        else:
            (loss / self.divisor).backward()
        return float(loss.detach())

    def backward(self, images, targets, *, step, learning_rate):
        if len(self.batches) >= self.divisor:
            raise RuntimeError("Too many minibatches in the accumulation group.")
        batch = {"images": [image.detach().clone() for image in images],
                 "targets": [{key: target[key].detach().clone() for key in ("boxes", "labels")}
                             for target in targets],
                 "image_ids": [_image_id(target) for target in targets],
                 "step": step, "learning_rate": learning_rate, "rng": rng_state()}
        self.batches.append(batch)
        value = self._backward(batch, mixed_precision=self.use_amp, scaled=True)
        self.loss_values.append(value)
        self.original_end_rng = rng_state()
        return value

    def _replay(self, *, mixed_precision, scaled):
        self.optimizer.zero_grad(set_to_none=True)
        self._restore_buffers()
        values = []
        for batch in self.batches:
            restore_rng(batch["rng"])
            values.append(self._backward(batch, mixed_precision=mixed_precision, scaled=scaled))
        return values

    def _exhausted(self, retries):
        """Run a diagnostic only; even finite float32 gradients are not applied."""
        detail = "finite"
        try:
            self._replay(mixed_precision=False, scaled=False)
            if not _finite_gradients(self.optimizer):
                detail = "nonfinite_gradients"
        except RuntimeError as error:
            detail = f"nonfinite_or_failed_forward_backward: {error}"
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._restore_buffers()
            restore_rng(self.original_end_rng)
        self._event("AMP_RETRY_LIMIT", retries=retries, final_scale=float(self.scaler.get_scale()),
                    float32_diagnostic=detail)
        raise RuntimeError(f"Nonfinite gradients: AMP overflow retry limit ({self.max_retries}) "
                           f"reached at {self._context()}; float32 diagnostic={detail}. "
                           "No optimizer update was applied to this group.")

    def finish(self):
        if len(self.batches) != self.divisor:
            raise RuntimeError("Cannot update an incomplete accumulation group.")
        retries = 0
        while True:
            self.scaler.unscale_(self.optimizer)
            if _finite_gradients(self.optimizer):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if retries:
                    self._event("AMP_GROUP_RECOVERED", retries=retries,
                                successful_scale=self.successful_scale,
                                scale_after_update=float(self.scaler.get_scale()), optimizer_steps=1)
                return {"loss_values": self.loss_values, "retries": retries, "optimizer_steps": 1,
                        "scale": float(self.scaler.get_scale())}
            if not self.scaler.is_enabled():
                self._event("NONFINITE_FLOAT32_GRADIENTS", scale=1.0, optimizer_steps=0)
                raise RuntimeError(f"Nonfinite gradients in float32 at {self._context()}; "
                                   "no optimizer update was applied.")
            failed_scale = float(self.scaler.get_scale())
            # unscale_ has registered the nonfinite gradients with GradScaler.
            # Its public step API skips the optimizer entirely, and update()
            # applies the configured backoff and clears its unscale/step state.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            reduced_scale = float(self.scaler.get_scale())
            retries += 1
            self._event("AMP_GRADIENT_OVERFLOW", failed_scale=failed_scale,
                        reduced_scale=reduced_scale, retry=retries, optimizer_steps=0)
            if not math.isfinite(reduced_scale) or not 0 < reduced_scale < failed_scale:
                raise RuntimeError("GradScaler did not produce a finite, smaller scale after overflow.")
            if retries > self.max_retries:
                self._exhausted(retries)
            self.successful_scale = reduced_scale
            self.loss_values = self._replay(mixed_precision=self.use_amp, scaled=True)


def recovery_witness(output_dir, device="cuda:0"):
    """Tiny real CUDA/fp16 kernel witness, independent of research data/models."""
    chosen_device = torch.device(device)
    if chosen_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The recovery witness requires an actual CUDA device.")

    class WitnessModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((1, 1), device=chosen_device))
            self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64, device=chosen_device))

        def forward(self, images, targets):
            self.forward_count.add_(1)
            return {"loss": (self.weight @ (images[0] * 128.)).sum()}

    def execute(scale):
        model = WitnessModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        scaler = torch.amp.GradScaler("cuda", init_scale=scale, growth_interval=2000)
        events, update_counts, skipped_update_proofs = [], [], []
        optimizer_updates = [0]
        optimizer.register_step_post_hook(lambda *args: optimizer_updates.__setitem__(0, optimizer_updates[0] + 1))

        def event_callback(event):
            events.append(event)
            if event["event"] == "AMP_GRADIENT_OVERFLOW":
                skipped_update_proofs.append({"weights_unchanged": bool(torch.equal(model.weight.detach(), before_weight)),
                                              "momentum_unchanged": bool(torch.equal(optimizer.state[model.weight].get("momentum_buffer", torch.zeros_like(model.weight)), before_momentum)),
                                              "optimizer_calls_unchanged": optimizer_updates[0] == before_updates})

        losses = []
        for group_index, size in enumerate([4, 4, 1], 1):
            before_weight = model.weight.detach().clone()
            before_momentum = optimizer.state[model.weight].get("momentum_buffer", torch.zeros_like(model.weight)).clone()
            before_updates = optimizer_updates[0]
            group = AccumulationGroup(model, optimizer, scaler, chosen_device, size, epoch=1,
                                      group_index=group_index, use_amp=True, event_callback=event_callback)
            for index in range(size):
                loss = group.backward([torch.ones((1, 1))],
                                      [{"boxes": torch.zeros((1, 4)), "labels": torch.ones(1, dtype=torch.int64),
                                        "image_id": torch.tensor([(group_index - 1) * 4 + index + 1])}],
                                      step=(group_index - 1) * 4 + index, learning_rate=.001)
                losses.append(loss)
            update_counts.append(group.finish()["optimizer_steps"])
        torch.cuda.synchronize(chosen_device)
        return {"model": model, "optimizer": optimizer, "scaler": scaler, "events": events,
                "optimizer_calls": optimizer_updates[0], "updates": update_counts,
                "skipped_update_proofs": skipped_update_proofs, "losses": losses}

    recovered, reference = execute(4096.), execute(128.)
    weight_parity = torch.equal(recovered["model"].weight, reference["model"].weight)
    momentum_parity = torch.equal(recovered["optimizer"].state[recovered["model"].weight]["momentum_buffer"],
                                  reference["optimizer"].state[reference["model"].weight]["momentum_buffer"])
    overflow_events = [event for event in recovered["events"] if event["event"] == "AMP_GRADIENT_OVERFLOW"]
    checks = {"finite_forward_losses": all(math.isfinite(loss) for loss in recovered["losses"]),
              "overflow_observed": bool(overflow_events),
              "official_backoff_observed": all(event["reduced_scale"] < event["failed_scale"] for event in overflow_events),
              "failed_attempts_preserved_weight_and_momentum": bool(recovered["skipped_update_proofs"]) and
              all(all(proof.values()) for proof in recovered["skipped_update_proofs"]),
              "exact_optimizer_update_count": recovered["optimizer_calls"] == 3 and recovered["updates"] == [1, 1, 1],
              "buffer_replay_did_not_double_count": int(recovered["model"].forward_count) == 9,
              "weights_match_finite_reference": weight_parity, "momentum_matches_finite_reference": momentum_parity}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "created_at": utc_now(),
              "device": str(chosen_device), "gpu": torch.cuda.get_device_name(chosen_device),
              "torch_version": torch.__version__, "policy": numerical_policy(), "checks": checks,
              "training_images": 9, "effective_batch_size": 4, "optimizer_updates": recovered["optimizer_calls"],
              "overflow_events": overflow_events, "skipped_update_proofs": recovered["skipped_update_proofs"],
              "final_scale_state": recovered["scaler"].state_dict(), "formal_training": False}
    destination = Path(output_dir) / "amp_recovery_witness.json"
    write_json(destination, result)
    if result["status"] != "PASS":
        raise RuntimeError(f"AMP recovery kernel witness failed: {destination}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the tiny CUDA/fp16 overflow replay witness.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    result = recovery_witness(arguments.output_dir, arguments.device)
    print(f"AMP recovery witness: {result['status']}; optimizer updates={result['optimizer_updates']}", flush=True)


if __name__ == "__main__":
    main()
