"""Local MODNAS/new-neck preflight and bounded real-data integration smoke.

Hardware feedback fails closed without new-space, source-bound measurements.
The Params smoke is deliberately separate from hardware-aware NAS results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
               str(ROOT / "tools/sonar_analysis")]

import torch
from torch import nn

import baseline as frozen
from tools.modnas_elastic.protocol import CONFIG, canonical_sha, dump, load_config, prepare, runtime, sha


def unpack_choices(choices):
    if choices.shape != (7, 2):
        raise ValueError("The new space requires seven binary choice rows")
    return dict(pan_weights=choices[0], td_kernel_weights=choices[1:4], pan_kernel_weights=choices[4:7])


def discrete_architecture(weights):
    pan = bool(weights["pan_weights"].detach().argmax().item())
    kernels = lambda tensor: [3 if int(value) == 0 else 5 for value in tensor.detach().argmax(dim=-1).tolist()]
    return dict(pan=pan, td_kernels=kernels(weights["td_kernel_weights"]),
                pan_kernels=kernels(weights["pan_kernel_weights"]) if pan else [])


def tensor_digest(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def finite_gradients(gradients):
    if any(not torch.isfinite(value).all() for value in gradients):
        raise ValueError("Nonfinite gradients")
    norm = math.sqrt(sum(float(value.detach().square().sum()) for value in gradients))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Architecture gradients must be finite and nonzero")
    return norm


def source_and_interfaces():
    from tools.modnas_elastic.author import source_provenance
    from tools.modnas_elastic.model import ElasticNeckSupernet, enumerate_architectures, architecture_id
    from tools.modnas_elastic.hardware import DifferentiableParamsTable, build_hardware_gap_report
    author = source_provenance()
    architectures = list(enumerate_architectures())
    if len(architectures) != 72 or len({architecture_id(a) for a in architectures}) != 72:
        raise ValueError("New-space cardinality/ID mismatch")
    return author, architectures, ElasticNeckSupernet, DifferentiableParamsTable, build_hardware_gap_report


def preflight(output, device="cpu", threads=4):
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    author, arches, _model, _params, build_gap = source_and_interfaces()
    manifest, _baseline = prepare(output, device)
    gap = build_gap()
    dump(Path(output) / "hardware_gap.json", gap)
    report = dict(status="PASS_LOCAL_SOFTWARE_PREFLIGHT", namespace=manifest["namespace"],
        execution_manifest_sha256=sha(Path(output) / "execution_manifest.json"), author=author,
        candidates=len(arches), candidates_without_pan=sum(not a["pan"] for a in arches),
        candidates_with_pan=sum(a["pan"] for a in arches),
        hardware_search_status="BLOCKED_MISSING_NEW_SPACE_COST",
        reason="New-neck measured cost labels are required; old-space aggregate costs are not reused",
        formal_eligible=False, test_accessed=False, test_evaluated=False)
    dump(Path(output) / "preflight.json", report)
    return report


def make_loaders(baseline, seed, batch_size):
    value = deepcopy(baseline)
    value["training"]["batch_size"] = batch_size
    data = ROOT / value["dataset"]["data_dir"]
    train = frozen.make_loader(value, data, "train", seed, 0, train=True)
    valid = frozen.make_loader(value, data, "valid", seed, 0)
    calibration = frozen.make_loader(value, data, "train", seed, 0)
    return value, train, valid, calibration


def calibration_smoke(candidate, loader, device):
    """One train batch checks the calibration path; it is not formal calibration."""
    candidate.eval()
    layers = [layer for layer in candidate.modules() if isinstance(layer, nn.BatchNorm2d)]
    previous = [layer.momentum for layer in layers]
    before = frozen.state_hash(candidate)
    parameter_before = {name: value.detach().clone() for name, value in candidate.named_parameters()}
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    images, _ = next(iter(loader))
    with torch.no_grad():
        candidate(images.to(device))
    for layer, momentum in zip(layers, previous):
        layer.momentum = momentum
    candidate.eval()
    unchanged = all(torch.equal(value, dict(candidate.named_parameters())[name])
                    for name, value in parameter_before.items())
    if not unchanged:
        raise ValueError("BN calibration changed trainable weights")
    return dict(split="train", batches=1, images=int(images.shape[0]),
                trainable_weights_unchanged=True, state_before=before,
                state_after=frozen.state_hash(candidate), formal_calibration=False)


def smoke(output, steps=1, device="cpu", threads=4):
    from tools.modnas_elastic.author import ConditionalHypernetwork, AuthorSampler, combine_gradients
    from tools.modnas_elastic.model import architecture_id
    from tools.modnas_elastic.hardware import architecture_probabilities
    from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss

    output = Path(output)
    if (output / "smoke_report.json").exists() or (output / "smoke_checkpoint.pt").exists():
        raise FileExistsError("Smoke already exists; use a new directory to preserve results")
    cfg, _ = load_config()
    if not 1 <= steps <= cfg["smoke"]["maximum_optimizer_steps"]:
        raise ValueError("This command supports only 1..3 bounded integration steps")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(threads)
    seed = cfg["smoke"]["seed"]
    frozen.seed_all(seed)
    author, arches, model_type, params_type, _gap = source_and_interfaces()
    manifest, baseline = prepare(output, device)
    baseline, train_loader, valid_loader, calibration_loader = make_loaders(
        baseline, seed, cfg["smoke"]["batch_size"])
    # The first real train/valid batches are replayed only for an interface smoke.
    train_images, train_targets = next(iter(train_loader))
    valid_images, valid_targets = next(iter(valid_loader))
    model = model_type(seed=seed, num_classes=2).to(device)
    hypernetwork = ConditionalHypernetwork().to(device)
    sampler = AuthorSampler(total_epochs=max(steps, 2))
    params_table = params_type().to(device)
    with torch.no_grad():
        probe = {"pan_weights": torch.tensor([0., 1.], device=device),
                 "td_kernel_weights": torch.tensor([[1., 0.]] * 3, device=device),
                 "pan_kernel_weights": torch.tensor([[1., 0.]] * 3, device=device)}
        # A fixed common-space maximum scales the cost objective, not any accuracy score.
        params_normalizer = int(params_table.values.max().item())
        probe_params = float(params_table(architecture_probabilities(probe)))
        actual_params = sum(p.numel() for p in model.export_subnet(discrete_architecture(probe)).parameters())
        if probe_params != actual_params:
            raise ValueError("Differentiable Params objective differs from the static exported model")
    criterion = DenseAnchorDetectionLoss(model.loss_config(baseline)).to(device)
    weight_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["algorithm"]["weight_lr"],
                                        weight_decay=cfg["algorithm"]["weight_decay"])
    architecture_optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=cfg["algorithm"]["hypernetwork_lr"])
    hardware = torch.zeros(1, 10, device=device)
    preferences = [torch.tensor([value], device=device) for value in cfg["algorithm"]["preference_vectors"]]
    initial_model = frozen.state_hash(model)
    initial_hypernetwork = frozen.state_hash(hypernetwork)
    records = []
    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step_index in range(steps):
        sampler.before_epoch()
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        weight_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            weight_choices = sampler.sample(hypernetwork(preferences[step_index % len(preferences)], hardware))
        weights = unpack_choices(weight_choices)
        prediction = model(train_images.to(device), weights)
        train_loss, train_parts = criterion(prediction, train_targets)
        if not torch.isfinite(train_loss):
            raise ValueError("Nonfinite real-batch detection weight loss")
        train_loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        weight_norm = finite_gradients(gradients)
        weight_optimizer.step()
        # Validation only updates architecture parameters; BN and network weights stay fixed.
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_fixed_state = frozen.state_hash(model)
        architecture_optimizer.zero_grad(set_to_none=True)
        hp = list(hypernetwork.parameters())
        gradient_groups = []
        conditions = []
        for preference in preferences:
            logits = hypernetwork(preference, hardware)
            choice = sampler.sample(logits)
            weights = unpack_choices(choice)
            predictions = model.forward_combinations(valid_images.to(device), weights)
            loss_e0, _ = criterion(predictions["E0"], valid_targets)
            loss_e1, _ = criterion(predictions["E1"], valid_targets)
            detection_loss = weights["pan_weights"][0] * loss_e0 + weights["pan_weights"][1] * loss_e1
            expected_params = params_table(architecture_probabilities(weights))
            objective = preference[0, 0] * detection_loss + preference[0, 1] * expected_params / params_normalizer
            raw = torch.autograd.grad(objective, hp, allow_unused=True)
            group = [torch.zeros_like(parameter) if gradient is None else gradient.detach()
                     for parameter, gradient in zip(hp, raw)]
            norm = finite_gradients(group)
            gradient_groups.append(group)
            arch = discrete_architecture(weights)
            conditions.append(dict(preference=preference.detach().cpu().tolist()[0],
                architecture=arch, architecture_id=architecture_id(arch),
                validation_detection_loss=float(detection_loss.detach()),
                raw_expected_Params=float(expected_params.detach()),
                normalized_Params=float(expected_params.detach()) / params_normalizer,
                scalarized_objective=float(objective.detach()), hypernetwork_gradient_norm=norm,
                hardware_status="UNKNOWN_NEW_SPACE_MEASUREMENT", evaluation_kind="real_batch_loss_not_final_accuracy"))
        combined, aggregation_weights, solver_cost = combine_gradients(gradient_groups)
        for parameter, gradient in zip(hp, combined):
            parameter.grad = gradient
        finite_gradients(combined)
        architecture_optimizer.step()
        if frozen.state_hash(model) != model_fixed_state:
            raise ValueError("Validation architecture update modified network weights or BN state")
        records.append(dict(step=step_index + 1, train_loss=train_parts, weight_gradient_norm=weight_norm,
            train_architecture=discrete_architecture(unpack_choices(weight_choices)),
            conditions=conditions, MGD_weights=aggregation_weights.tolist(), MGD_cost=solver_cost,
            architecture_validation_bn_unchanged=True,
            sampler_temperature=float(sampler.tau_curr.item())))
        print(json.dumps(dict(step=step_index + 1, train_loss=float(train_loss.detach()),
                             architecture_conditions=len(conditions), hardware_feedback="BLOCKED")), flush=True)
    if initial_model == frozen.state_hash(model) or initial_hypernetwork == frozen.state_hash(hypernetwork):
        raise ValueError("Both network weights and conditional architecture parameters must update")
    exports = []
    model.eval()
    with torch.no_grad():
        for preference in preferences:
            logits = hypernetwork(preference, hardware)
            indices = logits.argmax(dim=-1)
            choice = torch.nn.functional.one_hot(indices, num_classes=2).to(logits.dtype)
            weights = unpack_choices(choice)
            arch = discrete_architecture(weights)
            candidate = model.export_subnet(arch).to(device).eval()
            onehot_output = model(valid_images.to(device), weights)
            independent_output = candidate(valid_images.to(device))
            maximum = max(float((a - b).abs().max()) for a, b in zip(
                onehot_output["predictions"], independent_output["predictions"]))
            if maximum != 0.0:
                raise ValueError(f"Exported candidate differs from one-hot supernet: {maximum}")
            count = sum(p.numel() for p in candidate.parameters())
            expected = float(params_table(architecture_probabilities(weights)))
            if count != expected:
                raise ValueError("Candidate Params mismatch")
            exports.append(dict(preference=preference.cpu().tolist()[0], architecture=arch,
                architecture_id=architecture_id(arch), architecture_sha256=canonical_sha(arch),
                Params=count, prediction_shapes=[list(value.shape) for value in independent_output["predictions"]],
                onehot_export_max_abs_error=maximum, hardware_feasibility="UNKNOWN"))
    candidate = model.export_subnet(exports[0]["architecture"]).to(device).eval()
    calibration = calibration_smoke(candidate, calibration_loader, device)
    valid_smoke, _ = frozen.evaluate(candidate, valid_loader,
        DenseAnchorDetectionLoss(candidate.loss_config(baseline)).to(device), baseline, device, max_batches=1)
    checkpoint = output / "smoke_checkpoint.pt"
    state = dict(namespace=cfg["namespace"], mode="integration_smoke", seed=seed, optimizer_steps=steps,
        model_state_dict=model.state_dict(), hypernetwork_state_dict=hypernetwork.state_dict(),
        weight_optimizer_state_dict=weight_optimizer.state_dict(),
        architecture_optimizer_state_dict=architecture_optimizer.state_dict(),
        source_sha256=manifest["source_sha256"], execution_manifest_sha256=sha(output / "execution_manifest.json"),
        space=cfg["space"], author=author, objective="Params_diagnostic_not_FPGA_hardware_feedback",
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
        formal_eligible=False, test_accessed=False, test_evaluated=False)
    torch.save(state, checkpoint)
    export_path = output / "exported_candidate_smoke.pt"
    torch.save(dict(mode="integration_smoke", architecture=exports[0]["architecture"],
                    model_state_dict=candidate.state_dict(), calibration=calibration,
                    formal_eligible=False, test_evaluated=False), export_path)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    proposals = {architecture_id(r["train_architecture"]) for r in records}
    proposals.update(condition["architecture_id"] for record in records for condition in record["conditions"])
    report = dict(status="PASS_NEW_SPACE_LOCAL_INTEGRATION", author=author, space=cfg["space"],
        environment=runtime(device), seed=seed, optimizer_steps=steps, records=records,
        train_batch=dict(split="train", images=int(train_images.shape[0]), augmented_images_sha256=tensor_digest(train_images)),
        architecture_batch=dict(split="valid", images=int(valid_images.shape[0]), images_sha256=tensor_digest(valid_images)),
        criterion=asdict(criterion.config), Params_normalizer=params_normalizer,
        objective="detection_loss_and_exact_Params_DIAGNOSTIC", hardware_embedding_kind="diagnostic_ten_zeros_not_measured",
        hardware_search_status="BLOCKED_MISSING_NEW_SPACE_COST", actual_FPGA_metrics=None,
        exports=exports, calibration_smoke=calibration, validation_smoke=valid_smoke,
        validation_smoke_scope="one_valid_image_interface_check_not_final_accuracy",
        network_weights_changed=True, hypernetwork_parameters_changed=True,
        accounting=dict(wall_seconds=elapsed, GPU_h=elapsed / 3600 if device == "cuda" else 0.0,
            timed_scope="optimizer_steps_export_calibration_one_batch_validation_and_checkpoint_IO",
            excluded_setup="source_data_validation_model_HPN_initialization_and_data_loader_first_batches",
            CPU_wall_hours=elapsed / 3600 if device == "cpu" else None,
            unique_proposed_architectures=len(proposals), condition_evaluations=steps * len(preferences),
            unknown_hardware_candidates=len(proposals), distinct_full_retrained_candidate_evaluations=0,
            measured_hardware_queries=0, final_retraining_cost=None),
        artifacts={p.name: sha(p) for p in (checkpoint, export_path, output / "execution_manifest.json")},
        formal_eligible=False, performance_claim=False, test_accessed=False, test_evaluated=False,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
    dump(output / "smoke_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "smoke", "hardware-check"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/modnas_elastic_adapter_20261007/local")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cost-table", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = preflight(args.output, args.device, args.threads)
    elif args.mode == "smoke":
        result = smoke(args.output, args.steps, args.device, args.threads)
    else:
        if args.cost_table is None:
            result = dict(status="BLOCKED_MISSING_NEW_SPACE_COST",
                          reason="Supply source-bound measurements for the new 72-candidate space")
            print(json.dumps(result), flush=True)
            return 2
        from tools.modnas_elastic.hardware import ExternalMeasuredCostTable
        table = ExternalMeasuredCostTable.from_json(args.cost_table)
        result = dict(status="NEW_SPACE_MEASURED_COST_TABLE_VALIDATED", entries=len(table.architectures))
    print(json.dumps({key: result[key] for key in ("status", "hardware_search_status", "optimizer_steps") if key in result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
