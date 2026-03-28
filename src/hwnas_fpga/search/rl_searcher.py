"""Reinforcement Learning-based Neural Architecture Search.

This module implements RL-based NAS using a controller network to
predict architecture decisions, inspired by ENAS and ProxylessNAS.

Key components:
- Controller: MLP-based controller for architecture prediction
- Action Space: Discrete action space for architecture decisions
- Reward Function: Combined accuracy and hardware cost reward
- Training: REINFORCE-style policy gradient with baseline
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from hwnas_fpga.interfaces import SearchCandidate, SearchConstraints
from hwnas_fpga.search_space import SearchSpace, ArchitectureSpec, SearchSpaceConfig
from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.models import build_model
from hwnas_fpga.search.pareto import compute_pareto_front
from hwnas_fpga.training import train_model

if TYPE_CHECKING:
    from hwnas_fpga.experiment import ExperimentTracker


# ============================================================================
# Action Space Definition
# ============================================================================


@dataclass(frozen=True)
class ActionSpace:
    """动作空间定义

    定义 RL 控制器可以采取的所有离散动作。

    Attributes:
        channel_actions: 可选通道数
        depth_actions: 可选深度
        kernel_actions: 可选卷积核大小
        expand_actions: 可选扩展比
        op_actions: 可选算子类型
    """

    channel_actions: Tuple[int, ...]
    depth_actions: Tuple[int, ...]
    kernel_actions: Tuple[int, ...]
    expand_actions: Tuple[int, ...]
    op_actions: Tuple[str, ...]

    @property
    def num_channels(self) -> int:
        return len(self.channel_actions)

    @property
    def num_depths(self) -> int:
        return len(self.depth_actions)

    @property
    def num_kernels(self) -> int:
        return len(self.kernel_actions)

    @property
    def num_expands(self) -> int:
        return len(self.expand_actions)

    @property
    def num_ops(self) -> int:
        return len(self.op_actions)

    @property
    def total_actions(self) -> int:
        return self.num_channels + self.num_depths + self.num_kernels + self.num_expands + self.num_ops

    def channel_to_index(self, channel: int) -> int:
        return self.channel_actions.index(channel)

    def depth_to_index(self, depth: int) -> int:
        return self.depth_actions.index(depth)

    def kernel_to_index(self, kernel: int) -> int:
        return self.kernel_actions.index(kernel)

    def expand_to_index(self, expand: int) -> int:
        return self.expand_actions.index(expand)

    def op_to_index(self, op: str) -> int:
        return self.op_actions.index(op)

    @classmethod
    def from_search_config(cls, config: SearchSpaceConfig) -> "ActionSpace":
        """从 SearchSpaceConfig 创建动作空间"""
        return cls(
            channel_actions=tuple(config.channel_choices),
            depth_actions=tuple(config.depth_choices),
            kernel_actions=tuple(config.kernel_choices),
            expand_actions=tuple(config.expand_choices),
            op_actions=tuple(config.op_choices),
        )


@dataclass
class ArchitectureAction:
    """架构动作

    表示一个候选架构的完整决策序列。

    Attributes:
        channel_indices: 各stage的通道索引
        depth_indices: 各stage的深度索引
        kernel_indices: 各block的核索引
        expand_indices: 各block的扩展比索引
        op_indices: 各block的算子索引
    """

    channel_indices: Tuple[int, ...]
    depth_indices: Tuple[int, ...]
    kernel_indices: Tuple[int, ...]
    expand_indices: Tuple[int, ...]
    op_indices: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_indices": self.channel_indices,
            "depth_indices": self.depth_indices,
            "kernel_indices": self.kernel_indices,
            "expand_indices": self.expand_indices,
            "op_indices": self.op_indices,
        }


# ============================================================================
# Controller Network
# ============================================================================


class Controller(nn.Module):
    """RL 控制器网络

    使用 MLP 架构预测架构决策。
    每个决策类型（通道、深度、核、扩展比、算子）都有独立的输出头。

    Architecture:
        Embedding -> Shared Hidden Layers -> Multiple Output Heads
    """

    def __init__(
        self,
        action_space: ActionSpace,
        hidden_dim: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 16,
        max_stages: int = 8,
        max_blocks_per_stage: int = 8,
    ):
        super().__init__()

        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_stages = max_stages
        self.max_blocks_per_stage = max_blocks_per_stage

        # 共享隐藏层
        self.shared_layers = nn.ModuleList()
        input_dim = embedding_dim * 5  # 5种决策类型

        for i in range(num_layers):
            if i == 0:
                self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出头
        self.channel_head = nn.Linear(hidden_dim, action_space.num_channels)
        self.depth_head = nn.Linear(hidden_dim, action_space.num_depths)
        self.kernel_head = nn.Linear(hidden_dim, action_space.num_kernels)
        self.expand_head = nn.Linear(hidden_dim, action_space.num_expands)
        self.op_head = nn.Linear(hidden_dim, action_space.num_ops)

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for layer in self.shared_layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for head in [self.channel_head, self.depth_head, self.kernel_head, self.expand_head, self.op_head]:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        stage_idx: int,
        block_idx: int,
        prev_choices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播，返回各决策类型的logits

        Args:
            stage_idx: 当前stage索引
            block_idx: 当前block索引
            prev_choices: 之前的决策（用于建模依赖关系）

        Returns:
            各决策类型的logits字典
        """
        # 构建输入特征
        # 简化：使用stage和block索引的embedding
        stage_emb = self._get_embedding(stage_idx, self.max_stages)
        block_emb = self._get_embedding(block_idx, self.max_blocks_per_stage)

        x = torch.cat([stage_emb, block_emb, stage_emb, block_emb, stage_emb])  # 5个embedding

        # 共享隐藏层
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        # 输出头
        logits = {
            "channel": self.channel_head(x),
            "depth": self.depth_head(x),
            "kernel": self.kernel_head(x),
            "expand": self.expand_head(x),
            "op": self.op_head(x),
        }

        return logits

    def _get_embedding(self, idx: int, max_idx: int) -> torch.Tensor:
        """获取索引的embedding"""
        # 简化：使用正弦位置编码
        # 跟随 controller 参数所在设备（兼容 MPS / CUDA / CPU）
        device = next(self.parameters()).device
        emb = torch.zeros(self.embedding_dim, device=device)
        for i in range(self.embedding_dim):
            if i % 2 == 0:
                emb[i] = math.sin(idx / (max_idx ** (i / self.embedding_dim)))
            else:
                emb[i] = math.cos(idx / (max_idx ** ((i - 1) / self.embedding_dim)))
        return emb

    def sample(self, stage_idx: int, block_idx: int) -> Dict[str, int]:
        """采样动作

        Returns:
            各决策类型的索引字典
        """
        logits = self.forward(stage_idx, block_idx)

        # Gumbel-Softmax 采样
        actions = {}
        for key, logit in logits.items():
            probs = F.softmax(logit, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
            actions[key] = action

        return actions

    def get_log_prob(
        self,
        stage_idx: int,
        block_idx: int,
        actions: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """计算动作的log概率

        Returns:
            各决策类型的log概率字典
        """
        logits = self.forward(stage_idx, block_idx)

        log_probs = {}
        for key, action_idx in actions.items():
            if key in logits:
                probs = F.softmax(logits[key], dim=-1)
                log_probs[key] = torch.log(probs[action_idx])

        return log_probs


# ============================================================================
# Reward Function
# ============================================================================


class RewardFunction:
    """奖励函数

    结合任务精度和硬件代价计算奖励。

    Reward = accuracy - lambda_latency * normalized_latency
             - lambda_dsp * normalized_dsp - lambda_bram * normalized_bram
    """

    def __init__(
        self,
        accuracy_weight: float = 1.0,
        latency_weight: float = 0.2,
        energy_weight: float = 0.0,
        dsp_weight: float = 0.1,
        bram_weight: float = 0.1,
        lut_weight: float = 0.1,
        constraint_penalty: float = 10.0,
    ):
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.dsp_weight = dsp_weight
        self.bram_weight = bram_weight
        self.lut_weight = lut_weight
        self.constraint_penalty = constraint_penalty

        # 用于归一化的统计信息
        self._stats = {
            "max_accuracy": 0.0,
            "max_latency": 1.0,
            "max_energy": 1.0,
            "max_dsp": 1.0,
            "max_bram": 1.0,
            "max_lut": 1.0,
        }

    def compute_reward(
        self,
        accuracy: float,
        latency_ms: float,
        energy_mj: float,
        dsp: int,
        bram: int,
        lut: int,
        is_feasible: bool = True,
    ) -> float:
        """计算奖励

        Args:
            accuracy: 分类精度
            latency_ms: 延迟（毫秒）
            dsp: DSP使用量
            bram: BRAM使用量
            lut: LUT使用量
            is_feasible: 是否满足约束

        Returns:
            奖励值
        """
        # 如果不可行，给一个大的惩罚
        if not is_feasible:
            return -self.constraint_penalty

        # 更新统计信息（指数移动平均）
        self._stats["max_accuracy"] = max(
            self._stats["max_accuracy"], accuracy
        )
        self._stats["max_latency"] = max(self._stats["max_latency"], latency_ms)
        self._stats["max_energy"] = max(self._stats["max_energy"], energy_mj)
        self._stats["max_dsp"] = max(self._stats["max_dsp"], dsp)
        self._stats["max_bram"] = max(self._stats["max_bram"], bram)
        self._stats["max_lut"] = max(self._stats["max_lut"], lut)

        # 归一化
        norm_accuracy = accuracy / (self._stats["max_accuracy"] + 1e-8)
        norm_latency = latency_ms / (self._stats["max_latency"] + 1e-8)
        norm_energy = energy_mj / (self._stats["max_energy"] + 1e-8)
        norm_dsp = dsp / (self._stats["max_dsp"] + 1e-8)
        norm_bram = bram / (self._stats["max_bram"] + 1e-8)
        norm_lut = lut / (self._stats["max_lut"] + 1e-8)

        # 计算奖励
        reward = (
            self.accuracy_weight * norm_accuracy
            - self.latency_weight * norm_latency
            - self.energy_weight * norm_energy
            - self.dsp_weight * norm_dsp
            - self.bram_weight * norm_bram
            - self.lut_weight * norm_lut
        )

        return reward


# ============================================================================
# RL Searcher
# ============================================================================


class RLSearcher:
    """基于强化学习的神经架构搜索器

    使用 REINFORCE 算法训练控制器网络来生成最优架构。

    Algorithm:
        1. Controller 采样一个架构
        2. 评估该架构（训练+硬件估计）
        3. 计算奖励
        4. 更新控制器参数
        5. 重复

    Args:
        search_space: 搜索空间
        cost_estimator: 硬件代价估计器
        constraints: 搜索约束
        controller_hidden_dim: 控制器隐藏层维度
        controller_lr: 控制器学习率
        train_epochs_per_arch: 每个架构的训练轮数
        device: 设备
    """

    def __init__(
        self,
        search_space: SearchSpace,
        cost_estimator: FPGACostEstimator,
        constraints: Optional[SearchConstraints] = None,
        *,
        controller_hidden_dim: int = 64,
        controller_lr: float = 0.01,
        train_epochs_per_arch: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.search_space = search_space
        self.estimator = cost_estimator
        self.constraints = constraints
        self.train_epochs_per_arch = train_epochs_per_arch
        self.device = device
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 创建动作空间
        self.action_space = ActionSpace.from_search_config(search_space.config)

        # 创建控制器
        self.controller = Controller(
            self.action_space,
            hidden_dim=controller_hidden_dim,
        ).to(device)

        # 优化器
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(),
            lr=controller_lr,
        )

        # 奖励函数
        reward_weights = reward_weights or {}
        resource_weight = float(reward_weights.get("resource", 0.1))
        self.reward_function = RewardFunction(
            accuracy_weight=float(reward_weights.get("accuracy", 1.0)),
            latency_weight=float(reward_weights.get("latency", 0.2)),
            energy_weight=float(reward_weights.get("energy", 0.0)),
            dsp_weight=float(reward_weights.get("dsp", resource_weight)),
            bram_weight=float(reward_weights.get("bram", resource_weight)),
            lut_weight=float(reward_weights.get("lut", resource_weight)),
        )

        # Baseline（用于减少方差）
        self.baseline = 0.0
        self.baseline_momentum = 0.9

        # 搜索历史
        self.evaluated_candidates: List[SearchCandidate] = []
        self.feasible_candidates: List[SearchCandidate] = []
        self.infeasible_candidates: List[SearchCandidate] = []
        self.best_candidate: Optional[SearchCandidate] = None
        self.best_reward = float("-inf")
        self.last_trained_model = None
        self.last_training_history = None
        self.last_cost_estimate = None

    def generate_architecture(self) -> ArchitectureSpec:
        """使用控制器生成一个架构"""
        stage_count = self.search_space.config.stage_count
        depth_choices = self.action_space.depth_actions

        stage_strides = self.search_space.config.stage_strides

        stages = []
        current_channels = self.search_space.config.stem_channels

        for stage_idx in range(stage_count):
            # 采样通道数
            channel_idx = self.controller.sample(stage_idx, 0)["channel"]
            channels = self.action_space.channel_actions[channel_idx]

            # 采样深度
            depth_idx = self.controller.sample(stage_idx, 0)["depth"]
            depth = self.action_space.depth_actions[depth_idx]

            # 采样该stage的所有block
            blocks = []
            for block_idx in range(depth):
                block_idx_global = stage_idx * 4 + block_idx
                stride = stage_strides[stage_idx] if block_idx == 0 else 1

                # 确定输入和输出通道（用于skip合法性检查）
                block_in_channels = current_channels if block_idx == 0 else channels

                # 采样算子类型（考虑约束）
                valid_ops = []
                for op in self.action_space.op_actions:
                    if op == "skip":
                        # skip 需要：stride == 1 且 in_channels == out_channels
                        if stride == 1 and block_in_channels == channels:
                            valid_ops.append(op)
                    else:
                        valid_ops.append(op)

                if not valid_ops:
                    # 如果没有合法的op，使用第一个非skip的op
                    for op in self.action_space.op_actions:
                        if op != "skip":
                            valid_ops = [op]
                            break

                # 采样算子类型
                if len(valid_ops) == 1:
                    op = valid_ops[0]
                    op_idx = self.action_space.op_to_index(op)
                else:
                    op_idx = self.controller.sample(stage_idx, block_idx_global)["op"]
                    # 如果采样到不合法的op，从valid_ops中随机选择
                    if self.action_space.op_actions[op_idx] not in valid_ops:
                        op = random.choice(valid_ops)
                        op_idx = self.action_space.op_to_index(op)
                    else:
                        op = self.action_space.op_actions[op_idx]

                # 采样核大小（根据op调整）
                if op == "skip":
                    # skip 必须用 kernel_size=1
                    kernel_size = 1
                else:
                    kernel_idx = self.controller.sample(stage_idx, block_idx_global)["kernel"]
                    kernel_size = self.action_space.kernel_actions[kernel_idx]

                # 采样扩展比（根据op调整）
                if op in ("skip", "conv", "dw_pw_conv", "mixconv", "denoise", "edge"):
                    # 这些操作不需要 expand_ratio
                    expand_ratio = 1
                else:
                    expand_idx = self.controller.sample(stage_idx, block_idx_global)["expand"]
                    expand_ratio = self.action_space.expand_actions[expand_idx]

                # 构建block
                from hwnas_fpga.search_space import BlockSpec

                block = BlockSpec(
                    op=op,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    stride=stride,
                )
                blocks.append(block)

            # 构建stage
            from hwnas_fpga.search_space import StageSpec

            stages.append(
                StageSpec(
                    channels=channels,
                    depth=depth,
                    stride=stage_strides[stage_idx],
                    blocks=tuple(blocks),
                )
            )

            current_channels = channels

        # 构建完整架构
        from hwnas_fpga.search_space import ArchitectureSpec

        architecture = ArchitectureSpec(
            input_channels=self.search_space.config.input_channels,
            stem_channels=self.search_space.config.stem_channels,
            stages=tuple(stages),
            stem_stride=self.search_space.config.stem_stride,
            head_channels=self.search_space.config.head_channels,
            num_classes=self.search_space.config.num_classes,
        )

        return architecture

    def evaluate_architecture(
        self,
        architecture: ArchitectureSpec,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, float], bool, SearchCandidate]:
        """评估一个架构

        Returns:
            (accuracy, metrics_dict, is_feasible, candidate)
        """
        arch_id = f"rl_arch_{len(self.evaluated_candidates)}"
        self.last_trained_model = None
        self.last_training_history = None

        # 1. 硬件代价估计
        cost_estimate = self.estimator.estimate(architecture, self.search_space)
        self.last_cost_estimate = cost_estimate

        # 2. 检查可行性
        is_feasible = self.check_feasibility(cost_estimate)

        metrics = {
            "latency_ms": cost_estimate.latency_ms,
            "dsp": cost_estimate.peak_dsp,
            "bram": cost_estimate.peak_bram,
            "lut": cost_estimate.peak_lut,
            "power_w": cost_estimate.power_w,
            "energy_mj": cost_estimate.energy_mj,
            "memory_bandwidth_gbps": cost_estimate.memory_bandwidth_gbps,
            "offchip_mem_mb": cost_estimate.offchip_mem_mb,
        }

        # 3. 如果可行，训练并评估精度
        accuracy = 0.0
        if is_feasible:
            try:
                model = build_model(architecture, num_classes=num_classes)
                accuracy, history = train_model(
                    model=model,
                    train_loader=train_loader,
                    num_classes=num_classes,
                    epochs=self.train_epochs_per_arch,
                    device=self.device,
                    val_loader=val_loader,
                    class_weights=class_weights,
                    early_stopping_patience=2 if val_loader is not None else None,
                )
                self.last_trained_model = model
                self.last_training_history = history
            except Exception as e:
                print(f"Error training {arch_id}: {e}")
                is_feasible = False

        # 4. 记录候选
        from hwnas_fpga.interfaces import SearchCandidate, CandidateMetrics

        candidate = SearchCandidate(
            arch_id=arch_id,
            encoding=architecture.to_dict(),
            metrics=CandidateMetrics(
                accuracy=accuracy,
                latency_ms=metrics["latency_ms"],
                dsp=metrics["dsp"],
                bram=metrics["bram"],
                lut=metrics["lut"],
                power_w=metrics["power_w"],
                energy_mj=metrics["energy_mj"],
                memory_bandwidth_gbps=metrics["memory_bandwidth_gbps"],
                offchip_mem_mb=metrics["offchip_mem_mb"],
            ),
        )
        self.evaluated_candidates.append(candidate)
        if is_feasible:
            self.feasible_candidates.append(candidate)
        else:
            self.infeasible_candidates.append(candidate)

        # 更新最佳候选
        if is_feasible:
            reward = self.reward_function.compute_reward(
                accuracy=accuracy,
                latency_ms=metrics["latency_ms"],
                energy_mj=metrics["energy_mj"],
                dsp=metrics["dsp"],
                bram=metrics["bram"],
                lut=metrics["lut"],
                is_feasible=is_feasible,
            )
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_candidate = candidate

        return accuracy, metrics, is_feasible, candidate

    def check_feasibility(self, cost_estimate) -> bool:
        """检查架构是否满足约束"""
        if self.constraints:
            if (
                self.constraints.max_latency_ms is not None
                and cost_estimate.latency_ms > self.constraints.max_latency_ms
            ):
                return False
            if (
                self.constraints.max_dsp is not None
                and cost_estimate.peak_dsp > self.constraints.max_dsp
            ):
                return False
            if (
                self.constraints.max_bram is not None
                and cost_estimate.peak_bram > self.constraints.max_bram
            ):
                return False
            if (
                self.constraints.max_lut is not None
                and cost_estimate.peak_lut > self.constraints.max_lut
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

        hardware_spec = self.estimator.hardware_spec
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

    def update_controller(
        self,
        architecture: ArchitectureSpec,
        reward: float,
    ) -> float:
        """更新控制器参数

        Returns:
            损失值
        """
        self.controller_optimizer.zero_grad()

        # 计算策略梯度（简化版本）
        total_log_prob = 0.0

        # 对每个stage和block的决策计算log_prob
        for stage_idx, stage in enumerate(architecture.stages):
            stage_strides = self.search_space.config.stage_strides

            # Channel决策
            channel_idx = self.action_space.channel_to_index(stage.channels)
            channel_log_prob = self.controller.get_log_prob(stage_idx, 0, {"channel": channel_idx})["channel"]
            total_log_prob += channel_log_prob

            # Depth决策
            depth_idx = self.action_space.depth_to_index(stage.depth)
            depth_log_prob = self.controller.get_log_prob(stage_idx, 0, {"depth": depth_idx})["depth"]
            total_log_prob += depth_log_prob

            # Block决策
            for block_idx, block in enumerate(stage.blocks):
                block_idx_global = stage_idx * 4 + block_idx

                # Op决策
                op_idx = self.action_space.op_to_index(block.op)
                op_log_prob = self.controller.get_log_prob(stage_idx, block_idx_global, {"op": op_idx})["op"]
                total_log_prob += op_log_prob

                # Kernel决策（跳过操作的特殊处理）
                if block.op != "skip":
                    kernel_idx = self.action_space.kernel_to_index(block.kernel_size)
                    kernel_log_prob = self.controller.get_log_prob(stage_idx, block_idx_global, {"kernel": kernel_idx})["kernel"]
                    total_log_prob += kernel_log_prob

                # Expand决策（某些操作不需要expand）
                if block.op in ("mbconv", "fused_mbconv"):
                    expand_idx = self.action_space.expand_to_index(block.expand_ratio)
                    expand_log_prob = self.controller.get_log_prob(stage_idx, block_idx_global, {"expand": expand_idx})["expand"]
                    total_log_prob += expand_log_prob

        # REINFORCE损失
        advantage = reward - self.baseline
        loss = -advantage * total_log_prob

        loss.backward()
        self.controller_optimizer.step()

        # 更新baseline
        self.baseline = self.baseline * self.baseline_momentum + reward * (1 - self.baseline_momentum)

        return loss.item()

    def search(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        num_episodes: int = 100,
        timeout_minutes: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        artifact_tracker: Optional["ExperimentTracker"] = None,
    ) -> SearchCandidate:
        """执行RL搜索

        Args:
            train_loader: 训练数据加载器
            num_classes: 类别数
            num_episodes: 训练回合数
            device: 设备
            verbose: 是否打印详细信息

        Returns:
            最佳候选架构
        """
        if verbose:
            print(f"Starting RL NAS with {num_episodes} episodes...")
            print(f"Device: {self.device}")
            if timeout_minutes is not None:
                print(f"Timeout: {timeout_minutes} minutes")

        start_time = datetime.now()
        timeout_seconds = None if timeout_minutes is None else timeout_minutes * 60

        for episode in range(num_episodes):
            if timeout_seconds is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    if verbose:
                        print(f"Stopping RL search after timeout at episode {episode}")
                    break

            # 1. 生成架构
            architecture = self.generate_architecture()
            previous_best_reward = self.best_reward

            # 2. 评估架构
            accuracy, metrics, is_feasible, candidate = self.evaluate_architecture(
                architecture,
                train_loader,
                val_loader,
                num_classes,
                class_weights=class_weights,
            )

            # 3. 计算奖励
            reward = self.reward_function.compute_reward(
                accuracy=accuracy,
                latency_ms=metrics["latency_ms"],
                energy_mj=metrics["energy_mj"],
                dsp=metrics["dsp"],
                bram=metrics["bram"],
                lut=metrics["lut"],
                is_feasible=is_feasible,
            )

            # 4. 更新控制器
            loss = self.update_controller(architecture, reward)

            if artifact_tracker:
                artifact_tracker.record_candidate(
                    candidate,
                    feasible=is_feasible,
                    cost_estimate=self.last_cost_estimate,
                    history=self.last_training_history,
                    extra={
                        "episode": episode,
                        "reward": reward,
                        "loss": loss,
                        "baseline": self.baseline,
                        "search_method": "rl",
                    },
                )
                artifact_tracker.save_named_checkpoint(
                    "controller_latest.pt",
                    {
                        "saved_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "episode": episode,
                        "reward": reward,
                        "baseline": self.baseline,
                        "controller_state_dict": self.controller.state_dict(),
                        "optimizer_state_dict": self.controller_optimizer.state_dict(),
                    },
                )
                if self.best_reward > previous_best_reward:
                    artifact_tracker.save_named_checkpoint(
                        "controller_best.pt",
                        {
                            "saved_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                            "episode": episode,
                            "reward": reward,
                            "best_reward": self.best_reward,
                            "controller_state_dict": self.controller.state_dict(),
                            "optimizer_state_dict": self.controller_optimizer.state_dict(),
                        },
                    )
                    if self.last_trained_model is not None:
                        artifact_tracker.save_best_candidate(
                            candidate,
                            model_state_dict=self.last_trained_model.state_dict(),
                            history=self.last_training_history,
                            extra={
                                "selection_metric": "reward",
                                "episode": episode,
                                "reward": reward,
                                "best_reward": self.best_reward,
                            },
                        )
                artifact_tracker.update_search_state(
                    total_evaluated=len(self.evaluated_candidates),
                    feasible=len(self.feasible_candidates),
                    infeasible=len(self.infeasible_candidates),
                    best_candidate=self.best_candidate,
                    extra={
                        "episode": episode,
                        "reward": reward,
                        "loss": loss,
                        "baseline": self.baseline,
                    },
                )

            if verbose and (episode % 10 == 0 or episode == 0):
                print(
                    f"Episode {episode}/{num_episodes}: "
                    f"Acc={accuracy:.4f}, Reward={reward:.4f}, "
                    f"Feasible={is_feasible}, Loss={loss:.4f}, "
                    f"Baseline={self.baseline:.4f}"
                )

        if verbose:
            print(f"\nRL NAS completed!")
            print(f"Total evaluated: {len(self.evaluated_candidates)}")
            print(f"Feasible: {len(self.feasible_candidates)}")
            print(f"Infeasible: {len(self.infeasible_candidates)}")
            if self.best_candidate:
                print(f"Best candidate: {self.best_candidate.arch_id}")
                print(f"  Accuracy: {self.best_candidate.metrics.accuracy:.4f}")
                print(f"  Latency: {self.best_candidate.metrics.latency_ms:.2f}ms")

        return self.best_candidate

    def search_with_timeout(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        timeout_minutes: int = 60,
        num_episodes: int = 10_000,
        device: str = "cpu",
        verbose: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        artifact_tracker: Optional["ExperimentTracker"] = None,
    ) -> SearchCandidate:
        return self.search(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_episodes=num_episodes,
            timeout_minutes=timeout_minutes,
            device=device,
            verbose=verbose,
            class_weights=class_weights,
            artifact_tracker=artifact_tracker,
        )

    def get_pareto_front(self, k: int = 10) -> List[SearchCandidate]:
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
