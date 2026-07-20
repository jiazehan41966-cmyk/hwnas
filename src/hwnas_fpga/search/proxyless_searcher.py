from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hwnas_fpga.hardware import CostEstimate, FPGACostEstimator
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate, SearchConstraints
from hwnas_fpga.models import ProxylessSuperNet
from hwnas_fpga.search_space import SearchSpace

from .searcher import BaseSearcher, _metric_display_name, candidate_selection_score

if TYPE_CHECKING:
    from hwnas_fpga.experiment import ExperimentTracker


@dataclass
class ProxylessLossWeights:
    accuracy: float = 1.0
    latency: float = 0.25
    energy: float = 0.05
    dsp: float = 0.12
    bram: float = 0.12
    lut: float = 0.12

    @classmethod
    def from_mapping(cls, payload: Optional[dict[str, float]]) -> "ProxylessLossWeights":
        data = dict(payload or {})
        resource = float(data.get("resource", 0.12))
        return cls(
            accuracy=float(data.get("accuracy", 1.0)),
            latency=float(data.get("latency", 0.25)),
            energy=float(data.get("energy", 0.05)),
            dsp=float(data.get("dsp", resource)),
            bram=float(data.get("bram", resource)),
            lut=float(data.get("lut", resource)),
        )


@dataclass
class ProxylessGradRegConfig:
    loss_type: Optional[str] = None
    params: dict[str, float] = None
    target_hardware: Optional[str] = "latency_ms"
    ref_value: Optional[float] = None

    @classmethod
    def from_inputs(
        cls,
        *,
        loss_type: Optional[str],
        params: Optional[dict[str, float]],
        target_hardware: Optional[str],
        ref_value: Optional[float],
    ) -> "ProxylessGradRegConfig":
        normalized_type = None if loss_type is None else str(loss_type).strip().lower()
        if normalized_type in {"none", ""}:
            normalized_type = None
        if normalized_type not in {None, "add#linear", "mul#log"}:
            raise ValueError(f"Unsupported grad_reg_loss_type: {loss_type}")
        normalized_target = str(target_hardware or "latency_ms").strip().lower()
        return cls(
            loss_type=normalized_type,
            params=dict(params or {}),
            target_hardware=normalized_target,
            ref_value=None if ref_value is None else float(ref_value),
        )


class ProxylessSearcher(BaseSearcher):
    """ProxylessNAS-style differentiable search with binary path sampling."""

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
        *,
        seed: int = 42,
        device: str = "cpu",
        warmup_epochs: int = 10,
        search_epochs: int = 120,
        weight_lr: float = 0.025,
        weight_momentum: float = 0.9,
        weight_decay: float = 3e-4,
        arch_lr: float = 1e-3,
        arch_weight_decay: float = 0.0,
        temperature: float = 1.0,
        stage_channels: Optional[list[int]] = None,
        stage_depth: Optional[int] = None,
        max_eval_batches: int = 20,
        objective_weights: Optional[dict[str, float]] = None,
        grad_reg_loss_type: Optional[str] = None,
        grad_reg_loss_params: Optional[dict[str, float]] = None,
        target_hardware: Optional[str] = "latency_ms",
        ref_value: Optional[float] = None,
        selection_metric: str = "macro_f1",
    ) -> None:
        super().__init__(search_space, cost_estimator, constraints)
        self.seed = int(seed)
        self.device = device
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.search_epochs = max(1, int(search_epochs))
        self.max_eval_batches = max(1, int(max_eval_batches))
        self.selection_metric = str(selection_metric or "macro_f1")
        self.loss_weights = ProxylessLossWeights.from_mapping(objective_weights)
        self.grad_reg = ProxylessGradRegConfig.from_inputs(
            loss_type=grad_reg_loss_type,
            params=grad_reg_loss_params,
            target_hardware=target_hardware,
            ref_value=ref_value,
        )
        self.stage_channels = stage_channels
        self.stage_depth = stage_depth
        self.temperature = float(temperature)

        self.weight_lr = float(weight_lr)
        self.weight_momentum = float(weight_momentum)
        self.weight_decay = float(weight_decay)
        self.arch_lr = float(arch_lr)
        self.arch_weight_decay = float(arch_weight_decay)

        self._rng = Random(self.seed)
        torch.manual_seed(self.seed)

        resolved_classes = search_space.config.num_classes
        if resolved_classes is None:
            resolved_classes = 8

        self.supernet = ProxylessSuperNet(
            search_space=search_space,
            num_classes=int(resolved_classes),
            cost_estimator=cost_estimator,
            stage_channels=self.stage_channels,
            stage_depth=self.stage_depth,
            temperature=self.temperature,
        ).to(self.device)
        self.last_trained_model = self.supernet

        self._reset_optimizers()

    def _reset_optimizers(self) -> None:
        self.weight_optimizer = torch.optim.SGD(
            self.supernet.weight_parameters(),
            lr=self.weight_lr,
            momentum=self.weight_momentum,
            weight_decay=self.weight_decay,
        )
        self.arch_optimizer = torch.optim.Adam(
            self.supernet.arch_parameters(),
            lr=self.arch_lr,
            weight_decay=self.arch_weight_decay,
        )

    def _effective_limit(
        self,
        *,
        constraint_value: Optional[float],
        hardware_value: Optional[float],
        default_value: float = 1.0,
    ) -> float:
        values = [float(v) for v in (constraint_value, hardware_value) if v is not None and v > 0]
        if values:
            return max(1e-6, min(values))
        return max(1e-6, float(default_value))

    def _build_criterion(self, class_weights: Optional[torch.Tensor]) -> nn.Module:
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        return nn.CrossEntropyLoss(weight=class_weights).to(self.device)

    def _compute_hardware_penalty(self) -> tuple[torch.Tensor, dict[str, float]]:
        expected = self.supernet.expected_hardware_metrics()
        hardware_spec = self.estimator.hardware_spec
        constraints = self.constraints

        latency_ref = self._effective_limit(
            constraint_value=(constraints.max_latency_ms if constraints else None),
            hardware_value=None,
            default_value=50.0,
        )
        energy_ref = self._effective_limit(
            constraint_value=(constraints.max_energy_mj if constraints else None),
            hardware_value=None,
            default_value=10.0,
        )
        dsp_ref = self._effective_limit(
            constraint_value=(constraints.max_dsp if constraints else None),
            hardware_value=hardware_spec.max_dsp,
            default_value=220.0,
        )
        bram_ref = self._effective_limit(
            constraint_value=(constraints.max_bram if constraints else None),
            hardware_value=hardware_spec.max_bram,
            default_value=140.0,
        )
        lut_ref = self._effective_limit(
            constraint_value=(constraints.max_lut if constraints else None),
            hardware_value=hardware_spec.max_lut,
            default_value=53_200.0,
        )

        latency_norm = expected["latency_ms"] / latency_ref
        energy_norm = expected["energy_proxy_mj"] / energy_ref
        dsp_norm = expected["dsp"] / dsp_ref
        bram_norm = expected["bram"] / bram_ref
        lut_norm = expected["lut"] / lut_ref

        penalty = (
            self.loss_weights.latency * latency_norm
            + self.loss_weights.energy * energy_norm
            + self.loss_weights.dsp * dsp_norm
            + self.loss_weights.bram * bram_norm
            + self.loss_weights.lut * lut_norm
        )
        components = {
            "latency_norm": float(latency_norm.detach().item()),
            "energy_norm": float(energy_norm.detach().item()),
            "dsp_norm": float(dsp_norm.detach().item()),
            "bram_norm": float(bram_norm.detach().item()),
            "lut_norm": float(lut_norm.detach().item()),
        }
        return penalty, components

    @staticmethod
    def _resolve_metric_key(name: Optional[str]) -> str:
        alias = {
            "latency": "latency_ms",
            "latency_ms": "latency_ms",
            "dsp": "dsp",
            "bram": "bram",
            "lut": "lut",
            "energy": "energy_proxy_mj",
            "energy_proxy_mj": "energy_proxy_mj",
        }
        key = str(name or "latency_ms").strip().lower()
        if key not in alias:
            raise ValueError(f"Unsupported target_hardware metric: {name}")
        return alias[key]

    def _resolve_reg_ref_value(self, metric_key: str) -> float:
        if self.grad_reg.ref_value is not None and self.grad_reg.ref_value > 0:
            return float(self.grad_reg.ref_value)
        constraints = self.constraints
        hardware_spec = self.estimator.hardware_spec
        if metric_key == "latency_ms":
            return self._effective_limit(
                constraint_value=(constraints.max_latency_ms if constraints else None),
                hardware_value=None,
                default_value=50.0,
            )
        if metric_key == "energy_proxy_mj":
            return self._effective_limit(
                constraint_value=(constraints.max_energy_mj if constraints else None),
                hardware_value=None,
                default_value=10.0,
            )
        if metric_key == "dsp":
            return self._effective_limit(
                constraint_value=(constraints.max_dsp if constraints else None),
                hardware_value=hardware_spec.max_dsp,
                default_value=220.0,
            )
        if metric_key == "bram":
            return self._effective_limit(
                constraint_value=(constraints.max_bram if constraints else None),
                hardware_value=hardware_spec.max_bram,
                default_value=140.0,
            )
        if metric_key == "lut":
            return self._effective_limit(
                constraint_value=(constraints.max_lut if constraints else None),
                hardware_value=hardware_spec.max_lut,
                default_value=53_200.0,
            )
        return 1.0

    def _compose_arch_loss(self, ce_loss: torch.Tensor) -> tuple[torch.Tensor, float, dict[str, float]]:
        if self.grad_reg.loss_type is None:
            hw_penalty, components = self._compute_hardware_penalty()
            return self.loss_weights.accuracy * ce_loss + hw_penalty, float(hw_penalty.detach().item()), components

        expected = self.supernet.expected_hardware_metrics()
        metric_key = self._resolve_metric_key(self.grad_reg.target_hardware)
        if metric_key not in expected:
            raise KeyError(f"Expected metric '{metric_key}' is unavailable")
        expected_value = torch.clamp(expected[metric_key], min=1e-6)
        ref_value = max(1e-6, self._resolve_reg_ref_value(metric_key))

        if self.grad_reg.loss_type == "add#linear":
            reg_lambda = float(self.grad_reg.params.get("lambda", 0.1))
            reg_term = reg_lambda * (expected_value - ref_value) / ref_value
            arch_loss = ce_loss + reg_term
            reg_scalar = float(reg_term.detach().item())
        elif self.grad_reg.loss_type == "mul#log":
            alpha = float(self.grad_reg.params.get("alpha", 0.2))
            beta = float(self.grad_reg.params.get("beta", 0.3))
            log_ref = math.log(max(ref_value, 1.000001))
            if abs(log_ref) < 1e-6:
                log_ref = 1.0
            reg_ratio = torch.log(expected_value) / log_ref
            reg_factor = torch.pow(torch.clamp(reg_ratio, min=1e-6), beta)
            arch_loss = alpha * ce_loss * reg_factor
            reg_scalar = float(reg_factor.detach().item())
        else:
            raise ValueError(f"Unsupported grad_reg_loss_type: {self.grad_reg.loss_type}")

        components = {
            "target_metric": float(expected_value.detach().item()),
            "ref_value": float(ref_value),
            "reg_type": self.grad_reg.loss_type,
        }
        return arch_loss, reg_scalar, components

    def _set_weight_requires_grad(self, enabled: bool) -> None:
        for parameter in self.supernet.weight_parameters():
            parameter.requires_grad_(enabled)

    def _set_arch_requires_grad(self, enabled: bool) -> None:
        for parameter in self.supernet.arch_parameters():
            parameter.requires_grad_(enabled)

    def _next_batch(self, iterator, loader: DataLoader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    @staticmethod
    def _resolve_selection_score(summary: dict[str, float], selection_metric: str) -> float:
        normalized = str(selection_metric or "macro_f1").strip().lower()
        aliases = {
            "accuracy": "top1",
            "top1": "top1",
            "macro_f1": "macro_f1",
            "f1": "macro_f1",
            "weighted_f1": "weighted_f1",
        }
        key = aliases.get(normalized, normalized)
        if key not in summary:
            raise ValueError(f"Unsupported selection_metric: {selection_metric}")
        return float(summary[key])

    @torch.no_grad()
    def _evaluate_argmax_summary(
        self,
        data_loader: DataLoader,
        *,
        criterion: nn.Module,
        num_classes: int,
    ) -> dict[str, float]:
        self.supernet.eval()
        self.supernet.activate_argmax_paths()

        total_loss = 0.0
        total_samples = 0
        total_topk = 0
        confusion = torch.zeros(int(num_classes), int(num_classes), dtype=torch.long)
        for step, (inputs, targets) in enumerate(data_loader):
            if step >= self.max_eval_batches:
                break
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = self.supernet(inputs)
            loss = criterion(outputs, targets)

            predictions = outputs.argmax(dim=1)
            topk = max(1, min(5, outputs.shape[1]))
            topk_indices = outputs.topk(topk, dim=1).indices

            total_loss += float(loss.item()) * inputs.size(0)
            total_samples += int(targets.size(0))
            total_topk += int(topk_indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item())

            for target, prediction in zip(targets.view(-1).tolist(), predictions.view(-1).tolist()):
                confusion[int(target), int(prediction)] += 1

        if total_samples <= 0:
            return {
                "loss": 0.0,
                "top1": 0.0,
                "top5": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "num_samples": 0.0,
            }

        supports = confusion.sum(dim=1)
        macro_f1 = 0.0
        weighted_f1 = 0.0
        for class_index in range(int(confusion.shape[0])):
            tp = float(confusion[class_index, class_index].item())
            fp = float(confusion[:, class_index].sum().item() - tp)
            fn = float(confusion[class_index, :].sum().item() - tp)
            support = int(supports[class_index].item())
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            macro_f1 += f1
            weighted_f1 += f1 * support

        return {
            "loss": total_loss / max(1, total_samples),
            "top1": float(confusion.diag().sum().item()) / max(1, total_samples),
            "top5": total_topk / max(1, total_samples),
            "macro_f1": macro_f1 / max(1, int(confusion.shape[0])),
            "weighted_f1": weighted_f1 / max(1, total_samples),
            "num_samples": float(total_samples),
        }

    def _build_candidate_from_supernet(
        self,
        *,
        arch_id: str,
        summary: dict[str, float],
    ) -> tuple[SearchCandidate, bool, CostEstimate]:
        architecture = self.supernet.extract_architecture()
        cost_estimate = self.estimator.estimate(architecture, self.search_space)
        feasible = self.check_feasibility(cost_estimate)
        selection_score = self._resolve_selection_score(summary, self.selection_metric)
        metrics = cost_estimate.to_candidate_metrics()
        metrics = CandidateMetrics(
            selection_score=float(selection_score),
            accuracy=summary.get("top1"),
            macro_f1=summary.get("macro_f1"),
            weighted_f1=summary.get("weighted_f1"),
            top1=summary.get("top1"),
            top5=summary.get("top5"),
            latency_ms=metrics.latency_ms,
            dsp=metrics.dsp,
            bram=metrics.bram,
            lut=metrics.lut,
            power_w=metrics.power_w,
            energy_mj=metrics.energy_mj,
            memory_bandwidth_gbps=metrics.memory_bandwidth_gbps,
            offchip_mem_mb=metrics.offchip_mem_mb,
        )
        candidate = SearchCandidate(
            arch_id=arch_id,
            encoding=architecture.to_dict(),
            metrics=metrics,
        )
        return candidate, feasible, cost_estimate

    def _record_candidate(
        self,
        *,
        candidate: SearchCandidate,
        feasible: bool,
    ) -> None:
        self.evaluated_candidates.append(candidate)
        if feasible:
            self.feasible_candidates.append(candidate)
        else:
            self.infeasible_candidates.append(candidate)

    def search(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        num_candidates: int = 100,
        train_epochs: int = 3,
        device: str = "cpu",
        verbose: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        artifact_tracker: Optional["ExperimentTracker"] = None,
    ) -> Optional[SearchCandidate]:
        del num_candidates  # Unused for Proxyless search.
        del device

        if int(num_classes) != int(self.supernet.num_classes):
            self.supernet = ProxylessSuperNet(
                search_space=self.search_space,
                num_classes=int(num_classes),
                cost_estimator=self.estimator,
                stage_channels=self.stage_channels,
                stage_depth=self.stage_depth,
                temperature=self.temperature,
            ).to(self.device)
            self.last_trained_model = self.supernet
            self._reset_optimizers()

        criterion = self._build_criterion(class_weights)
        val_loader = val_loader or train_loader

        if verbose:
            metric_label = _metric_display_name(self.selection_metric)
            print(
                "Proxyless search setup: "
                f"warmup={self.warmup_epochs}, search={self.search_epochs}, "
                f"device={self.device}, mixed_ops={len(self.supernet.mixed_ops)}, "
                f"selection_metric={metric_label}"
            )

        # Warmup: optimize only weights with single-path sampling.
        if self.warmup_epochs > 0:
            self._set_weight_requires_grad(True)
            self._set_arch_requires_grad(False)
            for epoch in range(self.warmup_epochs):
                self.supernet.train()
                total_loss = 0.0
                total_batches = 0
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    self.supernet.sample_single_paths(self._rng)
                    self.weight_optimizer.zero_grad(set_to_none=True)
                    logits = self.supernet(inputs)
                    loss = criterion(logits, targets)
                    loss.backward()
                    self.weight_optimizer.step()

                    total_loss += float(loss.detach().item())
                    total_batches += 1

                if verbose:
                    avg_loss = total_loss / max(1, total_batches)
                    print(f"[Warmup {epoch + 1}/{self.warmup_epochs}] loss={avg_loss:.4f}")

        # Alternating optimization.
        self._set_arch_requires_grad(True)
        best_candidate: Optional[SearchCandidate] = None
        best_score = float("-inf")
        metric_label = _metric_display_name(self.selection_metric)
        train_epoch_history: list[dict[str, Any]] = []

        for epoch in range(self.search_epochs):
            self.supernet.train()
            train_iter = iter(train_loader)
            val_iter = iter(val_loader)
            num_steps = max(1, min(len(train_loader), len(val_loader)))

            epoch_weight_loss = 0.0
            epoch_arch_loss = 0.0
            epoch_hw_penalty = 0.0

            for _step in range(num_steps):
                (x_train, y_train), train_iter = self._next_batch(train_iter, train_loader)
                (x_val, y_val), val_iter = self._next_batch(val_iter, val_loader)

                x_train = x_train.to(self.device, non_blocking=True)
                y_train = y_train.to(self.device, non_blocking=True)
                x_val = x_val.to(self.device, non_blocking=True)
                y_val = y_val.to(self.device, non_blocking=True)

                # Step A: update weight parameters.
                self._set_weight_requires_grad(True)
                self._set_arch_requires_grad(False)
                self.supernet.sample_single_paths(self._rng)
                self.weight_optimizer.zero_grad(set_to_none=True)
                logits_train = self.supernet(x_train)
                weight_loss = criterion(logits_train, y_train)
                weight_loss.backward()
                self.weight_optimizer.step()
                epoch_weight_loss += float(weight_loss.detach().item())

                # Step B: update architecture parameters.
                self._set_weight_requires_grad(False)
                self._set_arch_requires_grad(True)
                self.supernet.sample_pair_paths(self._rng)
                self.arch_optimizer.zero_grad(set_to_none=True)
                logits_val = self.supernet(x_val)
                ce_loss = criterion(logits_val, y_val)
                arch_loss, reg_scalar, _hw_items = self._compose_arch_loss(ce_loss)
                arch_loss.backward()
                self.arch_optimizer.step()
                epoch_arch_loss += float(arch_loss.detach().item())
                epoch_hw_penalty += reg_scalar

                self._set_weight_requires_grad(True)

            val_summary = self._evaluate_argmax_summary(
                val_loader,
                criterion=criterion,
                num_classes=int(num_classes),
            )
            val_score = self._resolve_selection_score(val_summary, self.selection_metric)
            arch_id = f"proxyless_epoch_{len(self.evaluated_candidates)}"
            candidate, feasible, cost_estimate = self._build_candidate_from_supernet(
                arch_id=arch_id,
                summary=val_summary,
            )
            self._record_candidate(candidate=candidate, feasible=feasible)

            if feasible and candidate.metrics.accuracy is not None and candidate.metrics.accuracy > best_score:
                best_score = float(candidate.metrics.accuracy)
                best_candidate = candidate

            epoch_record = {
                "epoch": epoch + 1,
                "weight_loss": epoch_weight_loss / max(1, num_steps),
                "arch_loss": epoch_arch_loss / max(1, num_steps),
                "hw_penalty": epoch_hw_penalty / max(1, num_steps),
                "selection_metric": self.selection_metric,
                "val_score": float(val_score),
                "val_top1": float(val_summary["top1"]),
                "val_top5": float(val_summary["top5"]),
                "val_macro_f1": float(val_summary["macro_f1"]),
                "val_weighted_f1": float(val_summary["weighted_f1"]),
                "feasible": bool(feasible),
            }
            train_epoch_history.append(epoch_record)
            self.last_training_history = {
                "epochs": train_epoch_history,
                "best_eval": dict(val_summary),
                "selection_metric": self.selection_metric,
            }

            if artifact_tracker is not None:
                artifact_tracker.record_candidate(
                    candidate,
                    feasible=feasible,
                    cost_estimate=cost_estimate,
                    history={"epoch_summary": epoch_record, "best_eval": dict(val_summary)},
                    extra={"search_method": "proxyless", "epoch": epoch + 1},
                )
                if best_candidate is not None:
                    artifact_tracker.save_best_candidate(
                        best_candidate,
                        model_state_dict=self.supernet.state_dict(),
                        history=self.last_training_history,
                        extra={
                            "selection_metric": self.selection_metric,
                            "epoch": epoch + 1,
                            "best_score": best_score,
                        },
                    )
                artifact_tracker.update_search_state(
                    total_evaluated=len(self.evaluated_candidates),
                    feasible=len(self.feasible_candidates),
                    infeasible=len(self.infeasible_candidates),
                    best_candidate=best_candidate,
                    extra={"epoch": epoch + 1, "mode": "proxyless"},
                )

            if verbose:
                print(
                    f"[Search {epoch + 1}/{self.search_epochs}] "
                    f"w_loss={epoch_record['weight_loss']:.4f}, "
                    f"a_loss={epoch_record['arch_loss']:.4f}, "
                    f"{metric_label}={val_score:.4f}, feasible={feasible}"
                )

        if verbose:
            print("\nProxyless search completed!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Infeasible: {len(self.infeasible_candidates)}")
            if best_candidate is not None:
                print(
                    f"Best {metric_label}: "
                    f"{candidate_selection_score(best_candidate, self.selection_metric):.4f}"
                )
            elif self.evaluated_candidates:
                print(f"No feasible architecture found among {len(self.evaluated_candidates)} epochs")

        return best_candidate

    def get_search_summary(self) -> dict[str, int]:
        return {
            "total_evaluated": len(self.evaluated_candidates),
            "feasible": len(self.feasible_candidates),
            "infeasible": len(self.infeasible_candidates),
        }
