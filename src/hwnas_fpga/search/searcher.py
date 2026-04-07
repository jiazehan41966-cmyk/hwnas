"""Search algorithms and candidate selection."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional, List
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from hwnas_fpga.interfaces import (
    CandidateMetrics,
    SearchCandidate,
    SearchConstraints,
)
from hwnas_fpga.search_space import SearchSpace, ArchitectureSpec
from hwnas_fpga.hardware import FPGACostEstimator, CostEstimate
from hwnas_fpga.models import build_model
from hwnas_fpga.search.pareto import compute_pareto_front
from hwnas_fpga.training import train_model

if TYPE_CHECKING:
    from hwnas_fpga.experiment import ExperimentTracker


def _metric_display_name(selection_metric: str) -> str:
    normalized = str(selection_metric or "macro_f1").strip().lower()
    if normalized in {"macro_f1", "f1"}:
        return "MacroF1"
    if normalized in {"accuracy", "top1"}:
        return "Top1"
    if normalized == "weighted_f1":
        return "WeightedF1"
    return selection_metric


class BaseSearcher:
    """搜索器基类"""

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
    ):
        self.search_space = search_space
        self.estimator = cost_estimator
        self.constraints = constraints

        # 搜索历史
        self.evaluated_candidates: List[SearchCandidate] = []
        self.feasible_candidates: List[SearchCandidate] = []
        self.infeasible_candidates: List[SearchCandidate] = []
        self.last_trained_model = None
        self.last_training_history = None
        self.last_cost_estimate: Optional[CostEstimate] = None

    def check_feasibility(self, cost_estimate: CostEstimate) -> bool:
        """检查架构是否满足约束"""
        if self.constraints:
            # 检查 SearchConstraints
            if (
                self.constraints.max_latency_ms is not None
                and cost_estimate.latency_ms > self.constraints.max_latency_ms
            ):
                return False
            if (
                self.constraints.max_energy_mj is not None
                and cost_estimate.energy_mj > self.constraints.max_energy_mj
            ):
                return False
            if (
                self.constraints.max_model_size_mb is not None
                and cost_estimate.model_size_mb > self.constraints.max_model_size_mb
            ):
                return False
            if (
                self.constraints.max_dsp is not None
                and cost_estimate.resource_dsp > self.constraints.max_dsp
            ):
                return False
            if (
                self.constraints.max_bram is not None
                and cost_estimate.resource_bram > self.constraints.max_bram
            ):
                return False
            if (
                self.constraints.max_lut is not None
                and cost_estimate.resource_lut > self.constraints.max_lut
            ):
                return False
            if (
                self.constraints.max_power_w is not None
                and cost_estimate.power_w > self.constraints.max_power_w
            ):
                return False
            if (
                self.constraints.max_memory_bandwidth_gbps is not None
                and cost_estimate.memory_bandwidth_gbps > self.constraints.max_memory_bandwidth_gbps
            ):
                return False
            if (
                self.constraints.max_offchip_mem_mb is not None
                and cost_estimate.offchip_mem_mb > self.constraints.max_offchip_mem_mb
            ):
                return False

        # 检查硬件规格约束
        hardware_spec = self.estimator.hardware_spec
        if (
            hardware_spec.max_dsp is not None
            and cost_estimate.resource_dsp > hardware_spec.max_dsp
        ):
            return False
        if (
            hardware_spec.max_bram is not None
            and cost_estimate.resource_bram > hardware_spec.max_bram
        ):
            return False
        if (
            hardware_spec.max_lut is not None
            and cost_estimate.resource_lut > hardware_spec.max_lut
        ):
            return False
        if (
            hardware_spec.max_power_w is not None
            and cost_estimate.power_w > hardware_spec.max_power_w
        ):
            return False
        if (
            hardware_spec.memory_bandwidth_gbps is not None
            and cost_estimate.memory_bandwidth_gbps > hardware_spec.memory_bandwidth_gbps
        ):
            return False
        if (
            hardware_spec.offchip_mem_mb is not None
            and cost_estimate.offchip_mem_mb > hardware_spec.offchip_mem_mb
        ):
            return False

        return len(cost_estimate.violations) == 0


class RandomSearcher(BaseSearcher):
    """随机搜索器"""

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
        seed: int = 42,
        eval_early_stopping_patience: Optional[int] = 2,
        selection_metric: str = "macro_f1",
    ):
        super().__init__(search_space, cost_estimator, constraints)
        self.rng = random.Random(seed)
        self.eval_early_stopping_patience = eval_early_stopping_patience
        self.selection_metric = selection_metric

    def sample_candidate(self) -> ArchitectureSpec:
        """采样一个候选架构"""
        return self.search_space.sample(
            seed=None,
            rng=self.rng,
            cost_estimator=self.estimator,
            apply_pruning=True,
            require_feasible=True,
            max_feasible_attempts=32,
            prefer_lightweight=True,
        )

    def evaluate_candidate(
        self,
        architecture: ArchitectureSpec,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        train_epochs: int = 3,
        device: str = "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ) -> SearchCandidate:
        """评估单个候选架构"""
        arch_id = f"arch_{len(self.evaluated_candidates)}"
        self.last_trained_model = None
        self.last_training_history = None

        # 1. 硬件代价估计
        cost_estimate = self.estimator.estimate(architecture, self.search_space)
        self.last_cost_estimate = cost_estimate

        # 2. 检查可行性
        is_feasible = self.check_feasibility(cost_estimate)

        candidate = SearchCandidate(
            arch_id=arch_id,
            encoding=architecture.to_dict(),
            metrics=cost_estimate.to_candidate_metrics(),
        )

        # 3. 如果可行，训练并评估精度
        if is_feasible:
            try:
                model = build_model(architecture, num_classes=num_classes)
                accuracy, history = train_model(
                    model=model,
                    train_loader=train_loader,
                    num_classes=num_classes,
                    epochs=train_epochs,
                    device=device,
                    val_loader=val_loader,
                    class_weights=class_weights,
                    early_stopping_patience=(
                        self.eval_early_stopping_patience if val_loader is not None else None
                    ),
                    selection_metric=self.selection_metric,
                )
                candidate.metrics.accuracy = accuracy
                best_eval = dict(history.get("best_eval") or {})
                candidate.metrics.macro_f1 = best_eval.get("macro_f1")
                candidate.metrics.weighted_f1 = best_eval.get("weighted_f1")
                candidate.metrics.top1 = best_eval.get("top1")
                candidate.metrics.top5 = best_eval.get("top5")
                self.last_trained_model = model
                self.last_training_history = history
            except Exception as e:
                print(f"Error training {arch_id}: {e}")
                is_feasible = False

        # 4. 记录候选
        self.evaluated_candidates.append(candidate)
        if is_feasible:
            self.feasible_candidates.append(candidate)
        else:
            self.infeasible_candidates.append(candidate)

        return candidate, is_feasible

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
    ) -> SearchCandidate:
        """执行随机搜索"""
        best_candidate = None
        best_accuracy = 0.0
        metric_label = _metric_display_name(self.selection_metric)

        for i in range(num_candidates):
            # 采样架构
            architecture = self.sample_candidate()

            # 评估
            candidate, is_feasible = self.evaluate_candidate(
                architecture,
                train_loader,
                val_loader,
                num_classes,
                train_epochs=train_epochs,
                device=device,
                class_weights=class_weights,
            )

            if artifact_tracker:
                artifact_tracker.record_candidate(
                    candidate,
                    feasible=is_feasible,
                    cost_estimate=self.last_cost_estimate,
                    history=self.last_training_history,
                    extra={"iteration": i + 1, "search_method": "random"},
                )

            if is_feasible and verbose:
                acc = candidate.metrics.accuracy
                lat = candidate.metrics.latency_ms
                print(
                    f"[{i+1}/{num_candidates}] {candidate.arch_id}: "
                    f"{metric_label}={acc:.4f}, Lat={lat:.2f}ms"
                )

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_candidate = candidate
                    if artifact_tracker and self.last_trained_model is not None:
                        artifact_tracker.save_best_candidate(
                            candidate,
                            model_state_dict=self.last_trained_model.state_dict(),
                            history=self.last_training_history,
                            extra={
                                "selection_metric": self.selection_metric,
                                "best_accuracy": best_accuracy,
                                "iteration": i + 1,
                            },
                        )

            if artifact_tracker:
                artifact_tracker.update_search_state(
                    total_evaluated=len(self.evaluated_candidates),
                    feasible=len(self.feasible_candidates),
                    infeasible=len(self.infeasible_candidates),
                    best_candidate=best_candidate,
                    extra={"iteration": i + 1, "mode": "count"},
                )

        if verbose:
            print(f"\nSearch completed!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Infeasible: {len(self.infeasible_candidates)}")
            if best_candidate:
                print(f"Best {metric_label}: {best_accuracy:.4f}")

        return best_candidate

    def search_with_timeout(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        timeout_minutes: int = 60,
        train_epochs: int = 3,
        device: str = "cpu",
        verbose: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        artifact_tracker: Optional["ExperimentTracker"] = None,
    ) -> SearchCandidate:
        """带时间限制的搜索（参考TinyTNAS）"""
        start_time = datetime.now()
        timeout_seconds = timeout_minutes * 60

        best_candidate = None
        best_accuracy = 0.0
        iteration = 0
        metric_label = _metric_display_name(self.selection_metric)

        while True:
            # 检查是否超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= timeout_seconds:
                break

            # 采样架构
            architecture = self.sample_candidate()

            # 评估
            candidate, is_feasible = self.evaluate_candidate(
                architecture,
                train_loader,
                val_loader,
                num_classes,
                train_epochs=train_epochs,
                device=device,
                class_weights=class_weights,
            )

            iteration += 1

            if artifact_tracker:
                artifact_tracker.record_candidate(
                    candidate,
                    feasible=is_feasible,
                    cost_estimate=self.last_cost_estimate,
                    history=self.last_training_history,
                    extra={"iteration": iteration, "search_method": "random", "mode": "timeout"},
                )

            if is_feasible and verbose:
                acc = candidate.metrics.accuracy
                lat = candidate.metrics.latency_ms
                remaining = timeout_seconds - elapsed
                print(
                    f"[{iteration}] {candidate.arch_id}: "
                    f"{metric_label}={acc:.4f}, Lat={lat:.2f}ms, Remaining={remaining:.1f}s"
                )

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_candidate = candidate
                    if artifact_tracker and self.last_trained_model is not None:
                        artifact_tracker.save_best_candidate(
                            candidate,
                            model_state_dict=self.last_trained_model.state_dict(),
                            history=self.last_training_history,
                            extra={
                                "selection_metric": self.selection_metric,
                                "best_accuracy": best_accuracy,
                                "iteration": iteration,
                                "mode": "timeout",
                            },
                        )

            if artifact_tracker:
                artifact_tracker.update_search_state(
                    total_evaluated=len(self.evaluated_candidates),
                    feasible=len(self.feasible_candidates),
                    infeasible=len(self.infeasible_candidates),
                    best_candidate=best_candidate,
                    extra={
                        "iteration": iteration,
                        "remaining_seconds": max(0.0, timeout_seconds - elapsed),
                        "mode": "timeout",
                    },
                )

        if verbose:
            print(f"\nSearch completed after {iteration} iterations!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Infeasible: {len(self.infeasible_candidates)}")
            if best_candidate:
                print(f"Best {metric_label}: {best_accuracy:.4f}")

        return best_candidate

    def get_pareto_front(self, k: int = 10) -> List[SearchCandidate]:
        """获取Pareto前沿（多目标优化）"""
        if not self.feasible_candidates:
            return []

        pareto_front = compute_pareto_front(
            self.feasible_candidates,
            objectives=["accuracy", "latency_ms", "dsp", "bram", "lut"],
            directions=["max", "min", "min", "min", "min"],
        )
        sorted_candidates = sorted(
            pareto_front,
            key=lambda c: (c.metrics.accuracy or 0.0, -(c.metrics.latency_ms or float("inf"))),
            reverse=True,
        )
        return sorted_candidates[:k]

    def get_search_summary(self) -> dict[str, int]:
        return {
            "total_evaluated": len(self.evaluated_candidates),
            "feasible": len(self.feasible_candidates),
            "infeasible": len(self.infeasible_candidates),
        }


def create_searcher(
    search_space: SearchSpace,
    cost_estimator: FPGACostEstimator,
    constraints: Optional[SearchConstraints] = None,
    method: str = "random",
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """创建搜索器工厂"""
    if method == "random":
        return RandomSearcher(
            search_space=search_space,
            cost_estimator=cost_estimator,
            constraints=constraints,
            seed=seed,
            eval_early_stopping_patience=kwargs.get("eval_early_stopping_patience", 2),
            selection_metric=kwargs.get("selection_metric", "macro_f1"),
        )
    if method == "rl":
        from .rl_searcher import RLSearcher

        return RLSearcher(
            search_space=search_space,
            cost_estimator=cost_estimator,
            constraints=constraints,
            controller_hidden_dim=kwargs.get("controller_hidden_dim", 64),
            controller_lr=kwargs.get("controller_lr", 0.01),
            train_epochs_per_arch=kwargs.get("train_epochs_per_arch", 3),
            device=kwargs.get("device", "cpu"),
            seed=seed,
            reward_weights=kwargs.get("reward_weights"),
            eval_early_stopping_patience=kwargs.get("eval_early_stopping_patience", 2),
            selection_metric=kwargs.get("selection_metric", "macro_f1"),
        )
    if method == "proxyless":
        from .proxyless_searcher import ProxylessSearcher

        proxyless_cfg = dict(kwargs.get("proxyless_cfg") or {})
        stage_channels = proxyless_cfg.get("stage_channels")
        if isinstance(stage_channels, tuple):
            stage_channels = list(stage_channels)
        stage_depth = proxyless_cfg.get("stage_depth")
        grad_reg_loss_type = proxyless_cfg.get("grad_reg_loss_type")
        grad_reg_loss_params = proxyless_cfg.get("grad_reg_loss_params")
        if grad_reg_loss_type == "add#linear":
            grad_reg_loss_params = dict(grad_reg_loss_params or {})
            if "lambda" not in grad_reg_loss_params:
                grad_reg_loss_params["lambda"] = float(proxyless_cfg.get("grad_reg_loss_lambda", 0.1))
        elif grad_reg_loss_type == "mul#log":
            grad_reg_loss_params = dict(grad_reg_loss_params or {})
            if "alpha" not in grad_reg_loss_params:
                grad_reg_loss_params["alpha"] = float(proxyless_cfg.get("grad_reg_loss_alpha", 0.2))
            if "beta" not in grad_reg_loss_params:
                grad_reg_loss_params["beta"] = float(proxyless_cfg.get("grad_reg_loss_beta", 0.3))

        return ProxylessSearcher(
            search_space=search_space,
            cost_estimator=cost_estimator,
            constraints=constraints,
            seed=seed,
            device=kwargs.get("device", "cpu"),
            warmup_epochs=proxyless_cfg.get("warmup_epochs", 10),
            search_epochs=proxyless_cfg.get("search_epochs", kwargs.get("train_epochs_per_arch", 120)),
            weight_lr=proxyless_cfg.get("weight_lr", 0.025),
            weight_momentum=proxyless_cfg.get("weight_momentum", 0.9),
            weight_decay=proxyless_cfg.get("weight_decay", 3e-4),
            arch_lr=proxyless_cfg.get("arch_lr", 1e-3),
            arch_weight_decay=proxyless_cfg.get("arch_weight_decay", 0.0),
            temperature=proxyless_cfg.get("temperature", 1.0),
            stage_channels=stage_channels,
            stage_depth=stage_depth,
            max_eval_batches=proxyless_cfg.get("max_eval_batches", 20),
            objective_weights=kwargs.get("reward_weights"),
            grad_reg_loss_type=grad_reg_loss_type,
            grad_reg_loss_params=grad_reg_loss_params,
            target_hardware=proxyless_cfg.get("target_hardware", "latency_ms"),
            ref_value=proxyless_cfg.get("ref_value"),
        )
    raise ValueError(f"Unsupported search method: {method}")
