"""Constrained searcher with operator-specific control.

This module provides searchers that allow fine-grained control over
the search space, enabling ablation studies such as:
- Fused MBConv vs Standard MBConv comparison
- With/Without early pruning comparison (FNAS acceleration effects)
- Different search strategies (RL vs Differentiable)
"""

from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from hwnas_fpga.interfaces import SearchCandidate, SearchConstraints
from hwnas_fpga.search_space import SearchSpace, ArchitectureSpec, SearchSpaceConfig
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.models import build_model
from hwnas_fpga.training import train_model

# Import base searcher
from hwnas_fpga.search.searcher import BaseSearcher, RandomSearcher


# ============================================================================
# Search Configuration for Ablation Studies
# ============================================================================


@dataclass
class SearchConfig:
    """搜索配置

    用于控制搜索空间和行为，支持对比实验。

    Attributes:
        enable_fused_mbconv: 是否启用 Fused MBConv
        enable_standard_mbconv: 是否启用标准 MBConv
        enable_early_pruning: 是否启用早期剪枝（违反约束立即终止）
        enable_weight_sharing: 是否启用权重共享（未实现，预留）
        search_method: 搜索方法 (random, evolution, rl, differentiable)
        max_search_time_minutes: 最大搜索时间
        use_lut_only: 是否仅使用 LUT 估计（不回退到分析模型）
    """

    # 算子控制
    enable_fused_mbconv: bool = False
    enable_standard_mbconv: bool = True
    enable_dw_pw_conv: bool = True
    enable_conv: bool = True
    enable_skip: bool = True
    enable_mixconv: bool = False
    enable_denoise: bool = True
    enable_edge: bool = True

    # 搜索行为控制
    enable_early_pruning: bool = True
    enable_weight_sharing: bool = False
    use_lut_only: bool = False

    # 搜索方法
    search_method: str = "random"

    # 时间限制
    max_search_time_minutes: Optional[int] = None

    def get_enabled_ops(self) -> Set[str]:
        """获取启用的算子集合"""
        ops = set()
        if self.enable_fused_mbconv:
            ops.add("fused_mbconv")
        if self.enable_standard_mbconv:
            ops.add("mbconv")
        if self.enable_dw_pw_conv:
            ops.add("dw_pw_conv")
        if self.enable_conv:
            ops.add("conv")
        if self.enable_skip:
            ops.add("skip")
        if self.enable_mixconv:
            ops.add("mixconv")
        if self.enable_denoise:
            ops.add("denoise")
        if self.enable_edge:
            ops.add("edge")
        return ops

    @classmethod
    def create_fused_only(cls) -> "SearchConfig":
        """仅使用 Fused MBConv 的配置"""
        return cls(
            enable_fused_mbconv=True,
            enable_standard_mbconv=False,
            enable_dw_pw_conv=True,
            enable_conv=False,
            enable_skip=True,
        )

    @classmethod
    def create_standard_only(cls) -> "SearchConfig":
        """仅使用标准 MBConv 的配置"""
        return cls(
            enable_fused_mbconv=False,
            enable_standard_mbconv=True,
            enable_dw_pw_conv=True,
            enable_conv=False,
            enable_skip=True,
        )

    @classmethod
    def create_with_early_pruning(cls) -> "SearchConfig":
        """启用早期剪枝的配置"""
        return cls(
            enable_fused_mbconv=True,
            enable_standard_mbconv=True,
            enable_early_pruning=True,
        )

    @classmethod
    def create_without_early_pruning(cls) -> "SearchConfig":
        """禁用早期剪枝的配置"""
        return cls(
            enable_fused_mbconv=True,
            enable_standard_mbconv=True,
            enable_early_pruning=False,
        )


# ============================================================================
# Configurable Searcher
# ============================================================================


class ConfigurableSearcher(BaseSearcher):
    """可配置的约束搜索器

    支持基于配置的搜索空间控制和对比实验。

    Example:
        # Fused MBConv vs Standard MBConv 对比
        config = SearchConfig.create_fused_only()
        searcher = ConfigurableSearcher(search_space, estimator, config)
        best = searcher.search(train_loader, num_classes)

        # With/Without Early Pruning 对比
        config1 = SearchConfig.create_with_early_pruning()
        config2 = SearchConfig.create_without_early_pruning()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
        config: Optional[SearchConfig] = None,
        seed: int = 42,
    ):
        super().__init__(search_space, cost_estimator, constraints)

        self.config = config or SearchConfig()
        self.rng = __import__("random").Random(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

        # 创建受限的搜索空间
        self._restricted_space = self._create_restricted_search_space()

    def _create_restricted_search_space(self) -> SearchSpace:
        """根据配置创建受限的搜索空间"""
        enabled_ops = self.config.get_enabled_ops()

        # 获取原始配置
        original_config = self.search_space.config

        # 过滤算子选择
        filtered_ops = tuple(
            op for op in original_config.op_choices if op in enabled_ops
        )

        if not filtered_ops:
            raise ValueError(
                f"No operators enabled! Enabled ops: {enabled_ops}, "
                f"Available ops: {original_config.op_choices}"
            )

        # 创建新的搜索空间配置
        new_config = SearchSpaceConfig(
            input_channels=original_config.input_channels,
            image_size=original_config.image_size,
            stem_channels=original_config.stem_channels,
            stem_stride=original_config.stem_stride,
            stage_strides=original_config.stage_strides,
            channel_choices=original_config.channel_choices,
            depth_choices=original_config.depth_choices,
            kernel_choices=original_config.kernel_choices,
            expand_choices=original_config.expand_choices,
            op_choices=filtered_ops,
            head_channels=original_config.head_channels,
            num_classes=original_config.num_classes,
        )

        return SearchSpace(new_config)

    def sample_candidate(self) -> ArchitectureSpec:
        """从受限的搜索空间采样候选架构"""
        return self._restricted_space.sample(
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
        """评估单个候选架构（支持早期剪枝）"""
        arch_id = f"arch_{len(self.evaluated_candidates)}"

        # 1. 硬件代价估计
        cost_estimate = self.estimator.estimate(architecture, self._restricted_space)

        # 2. 检查可行性
        is_feasible = self.check_feasibility(cost_estimate)

        # 3. 早期剪枝（如果启用）
        if self.config.enable_early_pruning and not is_feasible:
            candidate = SearchCandidate(
                arch_id=arch_id,
                encoding=architecture.to_dict(),
                metrics=cost_estimate.to_candidate_metrics(),
            )
            self.evaluated_candidates.append(candidate)
            self.infeasible_candidates.append(candidate)
            return candidate, False

        # 4. 如果可行或未启用早期剪枝，训练并评估精度
        candidate = SearchCandidate(
            arch_id=arch_id,
            encoding=architecture.to_dict(),
            metrics=cost_estimate.to_candidate_metrics(),
        )

        if is_feasible:
            try:
                model = build_model(architecture, num_classes=num_classes)
                accuracy, _ = train_model(
                    model=model,
                    train_loader=train_loader,
                    num_classes=num_classes,
                    epochs=train_epochs,
                    device=device,
                    val_loader=val_loader,
                    class_weights=class_weights,
                    early_stopping_patience=2 if val_loader is not None else None,
                )
                candidate.metrics.accuracy = accuracy
            except Exception as e:
                print(f"Error training {arch_id}: {e}")
                is_feasible = False

        # 5. 记录候选
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
    ) -> SearchCandidate:
        """执行搜索"""
        if verbose:
            enabled_ops = ", ".join(self.config.get_enabled_ops())
            print(f"Search config:")
            print(f"  Enabled ops: {enabled_ops}")
            print(f"  Early pruning: {self.config.enable_early_pruning}")
            print(f"  Search method: {self.config.search_method}")
            print(f"  Number of candidates: {num_candidates}")

        best_candidate = None
        best_accuracy = 0.0

        for i in range(num_candidates):
            # 检查时间限制
            if self.config.max_search_time_minutes:
                # TODO: 添加时间检查
                pass

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

            if is_feasible and verbose:
                acc = candidate.metrics.accuracy
                lat = candidate.metrics.latency_ms
                print(
                    f"[{i+1}/{num_candidates}] {candidate.arch_id}: "
                    f"Acc={acc:.4f}, Lat={lat:.2f}ms"
                )

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_candidate = candidate

        if verbose:
            print(f"\nSearch completed!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Infeasible: {len(self.infeasible_candidates)}")

            if hasattr(self.estimator, "get_lut_stats"):
                lut_stats = self.estimator.get_lut_stats()
                print(f"LUT stats: {lut_stats}")

            if best_candidate:
                print(f"Best accuracy: {best_accuracy:.4f}")

        return best_candidate


# ============================================================================
# Ablation Study Helper
# ============================================================================


def run_ablation_study(
    search_configs: List[SearchConfig],
    search_space: SearchSpace,
    cost_estimator: FPGACostEstimator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_classes: int,
    num_candidates: int = 20,
    train_epochs: int = 3,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, SearchCandidate]:
    """运行消融实验

    Args:
        search_configs: 搜索配置列表
        search_space: 搜索空间
        cost_estimator: 硬件估计器
        train_loader: 训练数据加载器
        num_classes: 类别数
        num_candidates: 每个配置评估的候选数量
        train_epochs: 每个候选的训练轮数
        device: 设备

    Returns:
        字典：配置名称 -> 最佳候选
    """
    results = {}

    for i, config in enumerate(search_configs):
        config_name = f"config_{i}"

        print(f"\n{'='*60}")
        print(f"Running ablation study: {config_name}")
        print(f"{'='*60}")
        print(f"Config: {config}")

        # 为每个配置创建新的估计器（重置 LUT 统计）
        # 注意：实际可能需要克隆估计器
        searcher = ConfigurableSearcher(
            search_space=search_space,
            cost_estimator=cost_estimator,
            config=config,
            seed=42 + i,  # 不同配置使用不同种子
        )

        best = searcher.search(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_candidates=num_candidates,
            train_epochs=train_epochs,
            device=device,
            verbose=True,
            class_weights=class_weights,
        )

        results[config_name] = best

    return results


def run_fused_vs_standard_mbconv(
    search_space: SearchSpace,
    cost_estimator: FPGACostEstimator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_classes: int,
    num_candidates: int = 20,
    train_epochs: int = 3,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """运行 Fused MBConv vs 标准 MBConv 对比实验

    Returns:
        字典：包含两个配置的结果
    """
    configs = [
        SearchConfig.create_fused_only(),
        SearchConfig.create_standard_only(),
    ]

    results = run_ablation_study(
        search_configs=configs,
        search_space=search_space,
        cost_estimator=cost_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        num_candidates=num_candidates,
        train_epochs=train_epochs,
        device=device,
        class_weights=class_weights,
    )

    # 分析结果
    print(f"\n{'='*60}")
    print("Fused MBConv vs Standard MBConv Comparison")
    print(f"{'='*60}")

    fused_best = results["config_0"]
    standard_best = results["config_1"]

    print(f"\nFused MBConv:")
    if fused_best and fused_best.metrics.accuracy is not None:
        print(f"  Best Accuracy: {fused_best.metrics.accuracy:.4f}")
        print(f"  Latency: {fused_best.metrics.latency_ms:.2f}ms")
        print(f"  DSP: {fused_best.metrics.dsp}")
        print(f"  BRAM: {fused_best.metrics.bram}")
        print(f"  LUT: {fused_best.metrics.lut}")

    print(f"\nStandard MBConv:")
    if standard_best and standard_best.metrics.accuracy is not None:
        print(f"  Best Accuracy: {standard_best.metrics.accuracy:.4f}")
        print(f"  Latency: {standard_best.metrics.latency_ms:.2f}ms")
        print(f"  DSP: {standard_best.metrics.dsp}")
        print(f"  BRAM: {standard_best.metrics.bram}")
        print(f"  LUT: {standard_best.metrics.lut}")

    # 对比
    if (
        fused_best
        and standard_best
        and fused_best.metrics.accuracy is not None
        and standard_best.metrics.accuracy is not None
    ):
        acc_diff = fused_best.metrics.accuracy - standard_best.metrics.accuracy
        lat_diff = fused_best.metrics.latency_ms - standard_best.metrics.latency_ms
        print(f"\nDifference (Fused - Standard):")
        print(f"  Accuracy: {acc_diff:+.4f}")
        print(f"  Latency: {lat_diff:+.2f}ms")

    return {
        "fused_mbconv": fused_best,
        "standard_mbconv": standard_best,
        "configs": configs,
    }


def run_early_pruning_comparison(
    search_space: SearchSpace,
    cost_estimator: FPGACostEstimator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_classes: int,
    num_candidates: int = 20,
    train_epochs: int = 3,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """运行有/无早期剪枝的对比实验（复现 FNAS 加速效果）

    Returns:
        字典：包含两个配置的结果
    """
    configs = [
        SearchConfig.create_with_early_pruning(),
        SearchConfig.create_without_early_pruning(),
    ]

    results = run_ablation_study(
        search_configs=configs,
        search_space=search_space,
        cost_estimator=cost_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        num_candidates=num_candidates,
        train_epochs=train_epochs,
        device=device,
        class_weights=class_weights,
    )

    # 分析结果
    print(f"\n{'='*60}")
    print("Early Pruning vs No Early Pruning Comparison")
    print(f"{'='*60}")

    with_pruning = results["config_0"]
    without_pruning = results["config_1"]

    print(f"\nWith Early Pruning:")
    if with_pruning and with_pruning.metrics.accuracy is not None:
        print(f"  Best Accuracy: {with_pruning.metrics.accuracy:.4f}")
        print(f"  Latency: {with_pruning.metrics.latency_ms:.2f}ms")

    print(f"\nWithout Early Pruning:")
    if without_pruning and without_pruning.metrics.accuracy is not None:
        print(f"  Best Accuracy: {without_pruning.metrics.accuracy:.4f}")
        print(f"  Latency: {without_pruning.metrics.latency_ms:.2f}ms")

    return {
        "with_pruning": with_pruning,
        "without_pruning": without_pruning,
        "configs": configs,
    }
