# HW-NAS 项目进展总结

## 已完成的工作

### 1. LUT 模块 ✅ (复用 FBNet 设计)

**文件**: [hardware/lookup_table.py](src/hwnas_fpga/hardware/lookup_table.py)

**核心组件**:
- `OpSpec`: 算子规格定义
- `LutEntry`: LUT 条目
- `LutTable`: LUT 查找表（支持 pickle 序列化）
- `LutQueryEngine`: LUT 查询引擎（支持插值查询）
- `LutBuilder`: 从 profiling 数据构建 LUT

**测试结果**:
```
LUT table created: 774 entries
LUT query result: latency=0.0116ms, dsp=12
```

---

### 2. 扩展 cost.py 支持 LUT 查询 ✅

**文件**: [hardware/cost.py](src/hwnas_fpga/hardware/cost.py)

**新增功能**:
- `_block_to_op_spec()`: 将 ResolvedBlockSpec 转换为 OpSpec
- `get_lut_stats()`: 获取 LUT 命中统计
- LUT 未命中时自动回退到分析模型

---

### 3. 约束搜索器 ✅ (支持 Fused MBConv 对比)

**文件**: [search/constrained.py](src/hwnas_fpga/search/constrained.py)

**核心组件**:
- `SearchConfig`: 搜索配置类
- `ConfigurableSearcher`: 可配置的约束搜索器
- `run_fused_vs_standard_mbconv()`: Fused MBConv vs 标准 MBConv 对比
- `run_early_pruning_comparison()`: 有/无早期剪枝对比

---

### 4. Pareto 前沿选择器 ✅

**文件**: [search/pareto.py](src/hwnas_fpga/search/pareto.py)

**核心函数**:
- `compute_pareto_front()`: 计算 Pareto 前沿
- `compute_hypervolume()`: 计算超体积指标
- `ParetoFrontSelector`: Pareto 前沿选择器

**测试结果**:
```
Pareto front: 3 candidates
```

---

### 5. RL 搜索器 ✅ (强化学习搜索)

**文件**: [search/rl_searcher.py](src/hwnas_fpga/search/rl_searcher.py)
**脚本**: [run_rl_search.py](run_rl_search.py)

**核心组件**:
- `ActionSpace`: 动作空间定义
- `Controller`: MLP 控制器网络
- `RewardFunction`: 奖励函数（精度 + 硬件代价）
- `RLSearcher`: RL 搜索器（REINFORCE 算法）

**测试结果**:
```
Episode 0/2: Acc=0.0000, Reward=-10.0000, Feasible=False, Loss=-331.7963
RL NAS completed!
```

---

## 项目结构

```
src/hwnas_fpga/
├── interfaces.py              # 接口定义
├── search_space/              # 搜索空间
├── hardware/                  # 硬件估计
│   ├── cost.py               # FPGACostEstimator (支持 LUT)
│   └── lookup_table.py       # LUT查找表
├── models/                    # 模型构建
│   └── builder.py
├── data/                      # 数据加载
│   └── dataset.py
├── training/                  # 训练
│   └── trainer.py
└── search/                    # 搜索算法
    ├── searcher.py           # RandomSearcher
    ├── constrained.py        # ConfigurableSearcher
    ├── pareto.py             # ParetoFrontSelector
    └── rl_searcher.py        # RLSearcher

run_search.py                   # 随机搜索入口
run_rl_search.py                # RL 搜索入口
```

---

## 对比实验支持

### 已实现的对比实验

| 实验 | 函数 | 状态 |
|---|---|---|
| Fused MBConv vs 标准 MBConv | `run_fused_vs_standard_mbconv()` | ✅ |
| 有/无早期剪枝 | `run_early_pruning_comparison()` | ✅ |
| Pareto前沿分析 | `compute_pareto_front()` | ✅ |
| **RL 搜索** | `RLSearcher` | ✅ |

### 搜索方法对比

| 方法 | 类 | 状态 |
|---|---|---|
| 随机搜索 | `RandomSearcher` | ✅ |
| RL搜索 | `RLSearcher` | ✅ |
| 可微搜索 | - | ⏳ |

---

## 快速使用指南

### 随机搜索
```bash
python3 run_search.py --num-candidates 20 --train-epochs 3
```

### RL 搜索
```bash
python3 run_rl_search.py --episodes 50 --train-epochs 3 --controller-lr 0.01
```

### Fused MBConv 对比
```python
from hwnas_fpga.search import run_fused_vs_standard_mbconv
results = run_fused_vs_standard_mbconv(search_space, estimator, train_loader, 10, 20)
```

---

## 复用参考代码情况

| 参考库 | 复用内容 |
|---|---|
| **FBNet** | LutTable/LutItem/LutQuery 设计 |
| **TinyTNAS** | 约束检查、时间限制搜索 |
| **HW-NAS-Bench** | 硬件指标设计 |
| **HW-PR-NAS** | Pareto排名保持 |

---

## 下一步工作

| 优先级 | 功能 |
|---|---|
| P1 | 真实声呐数据加载（MARIS/UATD） |
| P1 | 权重共享超网训练 |
| P1 | 可微搜索器（DARTS/ProxylessNAS） |
| P2 | 从 HW-NAS-Bench 提取真实 FPGA 数据 |
| P2 | HLS/FPGA 工具链集成 |
