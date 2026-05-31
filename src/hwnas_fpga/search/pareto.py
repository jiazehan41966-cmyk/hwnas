"""Pareto Front Selection for Multi-objective Optimization.

This module implements Pareto frontier selection algorithms for HW-NAS,
enabling multi-objective optimization of accuracy, latency, and resource usage.

References:
- TinyTNAS: Hardware-aware multi-objective search
- HW-PR-NAS: Pareto ranking preservation in surrogate models
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from hwnas_fpga.interfaces import SearchCandidate, CandidateMetrics
from hwnas_fpga.interfaces import SearchConstraints


# ============================================================================
# Pareto Front Algorithms
# ============================================================================


def is_dominated(
    candidate_a: SearchCandidate,
    candidate_b: SearchCandidate,
    objectives: List[str],
    directions: List[str],
) -> bool:
    """检查 candidate_a 是否被 candidate_b 支配

    A 被 B 支配的条件：
    - 在所有目标上，A 不优于 B
    - 至少在一个目标上，A 严格劣于 B

    Args:
        candidate_a: 候选 A
        candidate_b: 候选 B
        objectives: 优化目标列表，如 ["accuracy", "latency_ms", "dsp"]
        directions: 优化方向列表，"max" 表示最大化，"min" 表示最小化

    Returns:
        True 如果 candidate_a 被 candidate_b 支配，否则 False
    """
    a_worse_in_any = False
    a_better_or_equal_in_all = True

    for obj, direction in zip(objectives, directions):
        a_val = getattr(candidate_a.metrics, obj, None)
        b_val = getattr(candidate_b.metrics, obj, None)

        # 处理 None 值（未评估的候选）
        if a_val is None or b_val is None:
            return False

        if direction == "max":
            # 目标是最大化，值越大越好
            if a_val < b_val:
                a_worse_in_any = True
            elif a_val > b_val:
                a_better_or_equal_in_all = False
        else:  # direction == "min"
            # 目标是最小化，值越小越好
            if a_val > b_val:
                a_worse_in_any = True
            elif a_val < b_val:
                a_better_or_equal_in_all = False

    return a_worse_in_any and a_better_or_equal_in_all


def compute_pareto_front(
    candidates: List[SearchCandidate],
    objectives: List[str] = ["top1", "latency_ms"],
    directions: List[str] = ["max", "min"],
) -> List[SearchCandidate]:
    """计算 Pareto 前沿

    Args:
        candidates: 候选列表
        objectives: 优化目标列表
        directions: 优化方向列表 ("max" 或 "min")

    Returns:
        Pareto 前沿候选列表

    Example:
        >>> pareto = compute_pareto_front(candidates, ["accuracy", "latency_ms"], ["max", "min"])
    """
    valid_candidates = [
        c
        for c in candidates
        if all(getattr(c.metrics, obj, None) is not None for obj in objectives)
    ]

    if not valid_candidates:
        return []

    pareto_front = []

    for candidate in valid_candidates:
        is_dominated_by_any = False
        for other in valid_candidates:
            if candidate is other:
                continue
            if is_dominated(candidate, other, objectives, directions):
                is_dominated_by_any = True
                break

        if not is_dominated_by_any:
            pareto_front.append(candidate)

    return pareto_front


def build_pareto_objectives(
    objective_weights: Optional[Dict[str, float]] = None,
    constraints: Optional[SearchConstraints] = None,
    selection_metric: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Build Pareto objectives from config weights and active constraints."""
    from hwnas_fpga.search.searcher import resolve_pareto_task_metric

    weights = objective_weights or {}
    primary_metric = resolve_pareto_task_metric(selection_metric or "macro_f1")
    objectives: List[str] = [primary_metric]
    directions: List[str] = ["max"]

    def add(metric: str, direction: str) -> None:
        if metric not in objectives:
            objectives.append(metric)
            directions.append(direction)

    if weights.get("latency", 0.0) > 0 or (constraints and constraints.max_latency_ms is not None):
        add("latency_ms", "min")
    if weights.get("energy", 0.0) > 0 or (constraints and constraints.max_energy_mj is not None):
        add("energy_mj", "min")

    resource_requested = any(
        weights.get(key, 0.0) > 0 for key in ("resource", "dsp", "bram", "lut")
    )
    if resource_requested or (constraints and constraints.max_dsp is not None):
        add("dsp", "min")
    if resource_requested or (constraints and constraints.max_bram is not None):
        add("bram", "min")
    if resource_requested or (constraints and constraints.max_lut is not None):
        add("lut", "min")
    if weights.get("power", 0.0) > 0 or (constraints and constraints.max_power_w is not None):
        add("power_w", "min")
    if (
        weights.get("memory_bandwidth", 0.0) > 0
        or (constraints and constraints.max_memory_bandwidth_gbps is not None)
    ):
        add("memory_bandwidth_gbps", "min")
    if (
        weights.get("offchip_mem", 0.0) > 0
        or (constraints and constraints.max_offchip_mem_mb is not None)
    ):
        add("offchip_mem_mb", "min")

    return objectives, directions


def compute_pareto_front_numpy(
    candidates: List[SearchCandidate],
    objectives: List[str] = ["accuracy", "latency_ms"],
    directions: List[str] = ["max", "min"],
) -> Tuple[List[SearchCandidate], np.ndarray]:
    """使用 NumPy 高效计算 Pareto 前沿

    Args:
        candidates: 候选列表
        objectives: 优化目标列表
        directions: 优化方向列表 ("max" 或 "min")

    Returns:
        (Pareto前沿候选, 目标值矩阵)
    """
    # 构建目标值矩阵
    valid_candidates = [
        c for c in candidates
        if all(getattr(c.metrics, obj, None) is not None for obj in objectives)
    ]

    if not valid_candidates:
        return [], np.array([])

    # 提取目标值
    points = np.array(
        [[getattr(c.metrics, obj) for obj in objectives] for c in valid_candidates]
    )

    # 标准化方向：都转换为"最小化"问题
    for i, direction in enumerate(directions):
        if direction == "max":
            points[:, i] = -points[:, i]  # 取负，将最大化转换为最小化

    # 计算支配关系
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # 如果 j 在所有目标上都优于或等于 i
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_pareto[i] = False
                break

    pareto_candidates = [valid_candidates[i] for i in range(n) if is_pareto[i]]
    pareto_points = points[is_pareto]

    # 恢复原始方向（如果是 max 则取负回来）
    for i, direction in enumerate(directions):
        if direction == "max":
            pareto_points[:, i] = -pareto_points[:, i]

    return pareto_candidates, pareto_points


# ============================================================================
# Pareto Front Ranking (分层 Pareto)
# ============================================================================


def compute_pareto_ranks(
    candidates: List[SearchCandidate],
    objectives: List[str] = ["accuracy", "latency_ms"],
    directions: List[str] = ["max", "min"],
) -> List[int]:
    """计算每个候选的 Pareto 分层排名

    第一层：Pareto 前沿
    第二层：移除第一层后的 Pareto 前沿
    以此类推...

    Args:
        candidates: 候选列表
        objectives: 优化目标列表
        directions: 优化方向列表

    Returns:
        每个候选的排名列表（从0开始）
    """
    if not candidates:
        return []

    ranks = [0] * len(candidates)
    unranked = list(candidates)
    current_rank = 0

    while unranked:
        # 计算当前未排名候选的 Pareto 前沿
        pareto = compute_pareto_front(unranked, objectives, directions)

        if not pareto:
            break

        # 设置排名
        for p in pareto:
            idx = candidates.index(p)
            ranks[idx] = current_rank

        # 移除已排名的候选
        unranked = [c for c in unranked if c not in pareto]
        current_rank += 1

    return ranks


# ============================================================================
# Hypervolume (超体积指标)
# ============================================================================


def compute_hypervolume(
    candidates: List[SearchCandidate],
    reference_point: Tuple[float, ...],
    objectives: List[str] = ["accuracy", "latency_ms"],
    directions: List[str] = ["max", "min"],
) -> float:
    """计算 Pareto 前沿的超体积

    Hypervolume 衡量 Pareto 前沿相对于参考点的支配区域大小。
    值越大表示解集质量越好。

    Args:
        candidates: 候选列表
        reference_point: 参考点（所有目标的最差值）
        objectives: 优化目标列表
        directions: 优化方向列表

    Returns:
        超体积值

    Example:
        >>> reference = (0.5, 100.0)  # (accuracy=0.5, latency=100ms)
        >>> hv = compute_hypervolume(pareto, reference, ["accuracy", "latency_ms"], ["max", "min"])
    """
    pareto, points = compute_pareto_front_numpy(candidates, objectives, directions)

    if len(points) == 0:
        return 0.0

    # 转换为最小化问题
    points_min = np.array(points)
    ref = np.array(reference_point)

    for i, direction in enumerate(directions):
        if direction == "max":
            # 对于最大化目标，参考点是下界
            # 需要反转计算
            points_min[:, i] = -points_min[:, i]

    # 计算超体积（2D 情况）
    if len(objectives) == 2:
        # 按 x 坐标排序
        sorted_idx = np.argsort(points_min[:, 0])
        points_min = points_min[sorted_idx]

        # 计算面积
        hv = 0.0
        prev_x = float("-inf")

        for point in points_min:
            if point[0] >= prev_x:
                width = point[0] - prev_x if prev_x != float("-inf") else 0
                height = point[1]
                hv += width * height
                prev_x = point[0]

        return abs(hv)

    # 对于更高维度，使用 WFG 算法（简化实现）
    # 这里只返回排序后的候选数量作为近似
    return len(pareto)


# ============================================================================
# Pareto Front Selector
# ============================================================================


class ParetoFrontSelector:
    """Pareto 前沿选择器

    提供多种选择策略从候选集中选择最优架构。

    Attributes:
        objectives: 优化目标列表
        directions: 优化方向列表
        selection_method: 选择方法 ("hypervolume", "rank", "knee")
    """

    def __init__(
        self,
        objectives: List[str] = ["accuracy", "latency_ms"],
        directions: List[str] = ["max", "min"],
        selection_method: str = "hypervolume",
    ):
        self.objectives = objectives
        self.directions = directions
        self.selection_method = selection_method

    def select(
        self,
        candidates: List[SearchCandidate],
        k: int = 1,
        reference_point: Optional[Tuple[float, ...]] = None,
    ) -> List[SearchCandidate]:
        """选择最优候选

        Args:
            candidates: 候选列表
            k: 选择数量
            reference_point: 参考点（用于 hypervolume）

        Returns:
            选中的候选列表
        """
        if not candidates:
            return []

        # 获取 Pareto 前沿
        pareto, _ = compute_pareto_front_numpy(
            candidates, self.objectives, self.directions
        )

        if len(pareto) <= k:
            return pareto

        # 根据选择方法排序
        if self.selection_method == "hypervolume":
            if reference_point is None:
                reference_point = self._default_reference_point(pareto)
            return self._select_by_hypervolume(pareto, k, reference_point)
        elif self.selection_method == "rank":
            return self._select_by_rank(candidates, k)
        elif self.selection_method == "knee":
            return self._select_knee_point(pareto, k)
        else:
            return pareto[:k]

    def _default_reference_point(self, candidates: List[SearchCandidate]) -> Tuple[float, ...]:
        """计算默认参考点"""
        mins = []
        maxs = []

        for obj, direction in zip(self.objectives, self.directions):
            values = [
                getattr(c.metrics, obj)
                for c in candidates
                if getattr(c.metrics, obj) is not None
            ]
            if values:
                mins.append(min(values))
                maxs.append(max(values))
            else:
                mins.append(0.0)
                maxs.append(1.0)

        # 对于最小化目标，参考点是最大值
        # 对于最大化目标，参考点是最小值
        reference = []
        for i, direction in enumerate(self.directions):
            if direction == "min":
                reference.append(maxs[i])
            else:
                reference.append(mins[i])

        return tuple(reference)

    def _select_by_hypervolume(
        self,
        candidates: List[SearchCandidate],
        k: int,
        reference_point: Tuple[float, ...],
    ) -> List[SearchCandidate]:
        """基于贡献的超体积选择"""
        if k >= len(candidates):
            return candidates

        selected = []
        remaining = list(candidates)

        # 贪心选择：每次选择贡献超体积最大的候选
        for _ in range(k):
            if not remaining:
                break

            best_candidate = None
            best_contribution = float("-inf")

            for candidate in remaining:
                # 计算加入该候选后的超体积
                test_set = selected + [candidate]
                hv = compute_hypervolume(
                    test_set, reference_point, self.objectives, self.directions
                )

                if hv > best_contribution:
                    best_contribution = hv
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def _select_by_rank(self, candidates: List[SearchCandidate], k: int) -> List[SearchCandidate]:
        """基于 Pareto 排名选择"""
        ranks = compute_pareto_ranks(candidates, self.objectives, self.directions)

        primary_metric = self.objectives[0] if self.objectives else "top1"
        sorted_candidates = sorted(
            zip(ranks, candidates),
            key=lambda x: (
                x[0],
                -(getattr(x[1].metrics, primary_metric, None) or 0),
            ),
        )

        return [c for _, c in sorted_candidates[:k]]

    def _select_knee_point(
        self, candidates: List[SearchCandidate], k: int
    ) -> List[SearchCandidate]:
        """选择 knee points（平衡点）

        Knee point 是精度和延迟的平衡点，
        即增加单位精度所需延迟变化最大的点。
        """
        if len(candidates) < 3 or k >= len(candidates):
            return candidates[:k]

        # 按延迟排序
        sorted_by_latency = sorted(
            candidates,
            key=lambda c: c.metrics.latency_ms or float("inf"),
        )

        # 计算每个点的 knee score
        knee_scores = []
        for i, c in enumerate(sorted_by_latency):
            if i == 0 or i == len(sorted_by_latency) - 1:
                knee_scores.append(0.0)
                continue

            # 计算斜率变化
            primary_metric = self.objectives[0] if self.objectives else "top1"
            prev_acc = getattr(sorted_by_latency[i - 1].metrics, primary_metric, None) or 0
            curr_acc = getattr(c.metrics, primary_metric, None) or 0
            next_acc = getattr(sorted_by_latency[i + 1].metrics, primary_metric, None) or 0

            prev_lat = sorted_by_latency[i - 1].metrics.latency_ms or 1
            curr_lat = c.metrics.latency_ms or 1
            next_lat = sorted_by_latency[i + 1].metrics.latency_ms or 1

            # 斜率
            slope_before = (curr_acc - prev_acc) / (curr_lat - prev_lat)
            slope_after = (next_acc - curr_acc) / (next_lat - curr_lat)

            # Knee score：斜率变化
            knee_scores.append(abs(slope_after - slope_before))

        # 选择 knee score 最大的点
        selected = []
        remaining = list(zip(knee_scores, sorted_by_latency))

        for _ in range(k):
            if not remaining:
                break

            remaining.sort(key=lambda x: x[0], reverse=True)
            best = remaining.pop(0)
            selected.append(best[1])

        return selected


# ============================================================================
# 可视化辅助（预留）
# ============================================================================


def plot_pareto_front(
    candidates: List[SearchCandidate],
    objectives: List[str] = ["accuracy", "latency_ms"],
    directions: List[str] = ["max", "min"],
    save_path: Optional[str] = None,
):
    """绘制 Pareto 前沿

    Args:
        candidates: 候选列表
        objectives: 优化目标列表
        directions: 优化方向列表
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plot")
        return

    pareto, points = compute_pareto_front_numpy(
        candidates, objectives, directions
    )
    non_pareto = [c for c in candidates if c not in pareto]

    # 提取坐标
    x_obj, y_obj = objectives[0], objectives[1]
    x_dir, y_dir = directions[0], directions[1]

    pareto_x = [getattr(c.metrics, x_obj) for c in pareto]
    pareto_y = [getattr(c.metrics, y_obj) for c in pareto]
    non_pareto_x = [getattr(c.metrics, x_obj) for c in non_pareto]
    non_pareto_y = [getattr(c.metrics, y_obj) for c in non_pareto]

    # 绘制
    plt.figure(figsize=(10, 6))
    plt.scatter(non_pareto_x, non_pareto_y, c="lightgray", label="All Candidates")
    plt.scatter(pareto_x, pareto_y, c="red", s=100, label="Pareto Front")
    plt.plot(pareto_x, pareto_y, "r--", alpha=0.5)

    # 标注坐标轴
    x_label = f"{x_obj} ({'↑' if x_dir == 'max' else '↓'})"
    y_label = f"{y_obj} ({'↑' if y_dir == 'max' else '↓'})"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Pareto Front")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Pareto plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
