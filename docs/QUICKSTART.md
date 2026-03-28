# HW-NAS 快速开始

## 项目状态：可以开始跑搜索！✅

所有核心模块已实现，现在可以执行完整的硬件感知神经架构搜索流程。

---

## 快速运行

```bash
# 最小化测试（3个候选架构，1个epoch）
python3 run_search.py --num-candidates 3 --train-epochs 1 --batch-size 8

# 完整搜索（更多候选，更多epoch）
python3 run_search.py --num-candidates 20 --train-epochs 3 --batch-size 32

# 带时间限制的搜索（60分钟）
python3 run_search.py --timeout-minutes 60 --train-epochs 3 --batch-size 32
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num-candidates` | 20 | 评估的架构数量 |
| `--train-epochs` | 3 | 每个架构的训练轮数 |
| `--timeout-minutes` | None | 搜索超时时间（分钟） |
| `--num-classes` | 10 | 类别数 |
| `--batch-size` | 32 | 批大小 |
| `--image-size` | 224 | 输入图像大小 |
| `--device` | auto | 设备 (cuda/cpu) |

---

## 项目结构

```
src/hwnas_fpga/
├── interfaces.py              # 接口定义
├── search_space/              # 搜索空间 ✅
│   ├── __init__.py
│   └── space.py              # ArchitectureSpec, SearchSpace
├── hardware/                  # 硬件估计 ✅
│   ├── __init__.py
│   ├── cost.py               # FPGACostEstimator, CostEstimate
│   └── lookup_table.py       # LUT查找表 (NEW!)
├── models/                    # 模型构建 ✅
│   ├── __init__.py
│   └── builder.py            # HWNASModel, build_model
├── data/                      # 数据加载 ✅
│   ├── __init__.py
│   └── dataset.py            # DummySonarDataset
├── training/                  # 训练 ✅
│   ├── __init__.py
│   └── trainer.py            # Trainer, train_model
└── search/                    # 搜索算法 ✅
    ├── __init__.py
    ├── searcher.py           # RandomSearcher, BaseSearcher
    ├── constrained.py        # ConfigurableSearcher (NEW!)
    └── pareto.py             # ParetoFrontSelector (NEW!)

run_search.py                   # 搜索入口脚本 ✅
```

---

## 核心模块说明

### 1. 搜索空间 (search_space/space.py)

```python
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig

# 创建搜索空间
space = SearchSpace(
    SearchSpaceConfig(
        input_channels=1,      # 单通道声呐图像
        image_size=224,
        channel_choices=(16, 24, 32, 48, 64, 96),
        depth_choices=(1, 2, 3, 4),
        kernel_choices=(3, 5),
        op_choices=("conv", "dw_pw_conv", "mbconv", "fused_mbconv", "skip"),
    )
)

# 采样架构
arch = space.sample(seed=42)
```

### 2. 硬件估计器 (hardware/cost.py) + LUT支持

```python
from hwnas_fpga.hardware import FPGACostEstimator, LutTable, LutQueryEngine, create_dummy_fpga_lut
from hwnas_fpga.interfaces import HardwareSpec, SearchConstraints

# 创建 LUT 表（可选，用于加速）
lut_table = create_dummy_fpga_lut()  # 或从文件加载: LutTable.load("path/to/lut.pkl")
lut_query = LutQueryEngine(lut_table, enable_interpolation=True)

# 创建估计器（支持 LUT 查询）
estimator = FPGACostEstimator(
    hardware_spec=HardwareSpec(
        name="fpga",
        clock_mhz=200,
        max_dsp=512,
        max_bram=2000,
    ),
    constraints=SearchConstraints(
        max_latency_ms=50.0,
        max_dsp=500,
    ),
    lut_query_engine=lut_query,  # 可选：启用 LUT 加速
)

# 估计架构代价
estimate = estimator.estimate(arch, space)
# estimate.latency_ms, estimate.peak_dsp, estimate.peak_bram, ...

# 获取 LUT 统计信息
lut_stats = estimator.get_lut_stats()
# lut_stats: {"hits": 10, "misses": 2, "total": 12, "hit_rate": 0.833}
```

### 3. 模型构建 (models/builder.py)

```python
from hwnas_fpga.models import build_model

# 构建PyTorch模型
model = build_model(
    architecture=arch,
    num_classes=10,
    head_channels=None,
)
```

### 4. 基础搜索算法 (search/searcher.py)

```python
from hwnas_fpga.search import create_searcher

# 创建搜索器
searcher = create_searcher(
    search_space=space,
    cost_estimator=estimator,
    constraints=constraints,
    method="random",
    seed=42,
)

# 执行搜索
best_candidate = searcher.search(
    train_loader=train_loader,
    num_classes=10,
    num_candidates=20,
    train_epochs=3,
)
```

### 5. 约束搜索器 (search/constrained.py) - NEW!

支持算子级别控制和消融实验：

```python
from hwnas_fpga.search import ConfigurableSearcher, SearchConfig

# Fused MBConv vs Standard MBConv 对比
config_fused = SearchConfig.create_fused_only()     # 仅 Fused MBConv
config_standard = SearchConfig.create_standard_only()  # 仅标准 MBConv

searcher = ConfigurableSearcher(
    search_space=space,
    cost_estimator=estimator,
    config=config_fused,
    seed=42,
)

best = searcher.search(train_loader, num_classes=10, num_candidates=20)
```

**快捷消融实验：**

```python
from hwnas_fpga.search import run_fused_vs_standard_mbconv

# 自动运行对比实验
results = run_fused_vs_standard_mbconv(
    search_space=space,
    cost_estimator=estimator,
    train_loader=train_loader,
    num_classes=10,
    num_candidates=20,
)

# results: {"fused_mbconv": best_fused, "standard_mbconv": best_standard}
```

### 6. Pareto前沿选择器 (search/pareto.py) - NEW!

多目标优化和Pareto前沿分析：

```python
from hwnas_fpga.search import ParetoFrontSelector, compute_pareto_front

# 计算Pareto前沿
pareto_front = compute_pareto_front(
    candidates=searcher.evaluated_candidates,
    objectives=["accuracy", "latency_ms"],
    directions=["max", "min"],  # 最大化精度，最小化延迟
)

# 使用选择器选择最优候选
selector = ParetoFrontSelector(
    objectives=["accuracy", "latency_ms"],
    directions=["max", "min"],
    selection_method="hypervolume",  # 可选: hypervolume, rank, knee
)

top_k = selector.select(
    candidates=searcher.feasible_candidates,
    k=5,
    reference_point=(0.5, 100.0),  # (accuracy=0.5, latency=100ms)
)

# 可视化Pareto前沿
from hwnas_fpga.search import plot_pareto_front
plot_pareto_front(searcher.evaluated_candidates, save_path="pareto.png")
```

---

## 已实现的功能

### 搜索器

| 搜索器 | 状态 | 说明 |
|---|---|---|
| `RandomSearcher` | ✅ 已实现 | 随机采样 + 约束检查 |
| `ConfigurableSearcher` | ✅ 已实现 | 可配置的约束搜索器 |
| `DifferentiableSearcher` | ⏳ 待实现 | 可微搜索 |
| `EvolutionSearcher` | ⏳ 待实现 | 进化算法 |

### 硬件估计

| 功能 | 状态 | 说明 |
|---|---|---|
| 分析模型 | ✅ 已实现 | 基于 MACs/DSP/BRAM/LUT 公式 |
| LUT查找表 | ✅ 已实现 | 参考FBNet设计，支持插值 |
| LUT+分析混合 | ✅ 已实现 | LUT未命中时回退到分析模型 |

### 消融实验

| 实验 | 状态 | 说明 |
|---|---|---|
| Fused MBConv vs 标准 MBConv | ✅ 已实现 | `run_fused_vs_standard_mbconv()` |
| 有/无早期剪枝 | ✅ 已实现 | `run_early_pruning_comparison()` |
| 自定义算子配置 | ✅ 已实现 | `SearchConfig` |

---

## 消融实验示例

### Fused MBConv vs 标准 MBConv

```python
from hwnas_fpga.search import run_fused_vs_standard_mbconv

results = run_fused_vs_standard_mbconv(
    search_space=space,
    cost_estimator=estimator,
    train_loader=train_loader,
    num_classes=10,
    num_candidates=20,
)

# 输出示例:
# Fused MBConv:
#   Best Accuracy: 0.6523
#   Latency: 25.34ms
#   DSP: 256
#
# Standard MBConv:
#   Best Accuracy: 0.6489
#   Latency: 28.12ms
#   DSP: 288
#
# Difference (Fused - Standard):
#   Accuracy: +0.0034
#   Latency: -2.78ms
```

### 有/无早期剪枝

```python
from hwnas_fpga.search import run_early_pruning_comparison

results = run_early_pruning_comparison(
    search_space=space,
    cost_estimator=estimator,
    train_loader=train_loader,
    num_classes=10,
    num_candidates=20,
)

# 输出示例:
# With Early Pruning:
#   Best Accuracy: 0.6501
#   Latency: 26.50ms
#   Feasible candidates: 8
#   Infeasible candidates: 12
#
# Without Early Pruning:
#   Best Accuracy: 0.6512
#   Latency: 27.80ms
```

---

## LUT 使用指南

### 创建虚拟 LUT 表

```python
from hwnas_fpga.hardware import create_dummy_fpga_lut, LutTable, LutBuilder

# 使用预定义的虚拟表
lut_table = create_dummy_fpga_lut()

# 或从 HW-NAS-Bench 提取真实数据
# lut_table = extract_from_hw_nas_bench("HW-NAS-Bench-v1_0.pickle")
```

### 保存/加载 LUT 表

```python
# 保存
lut_table.save("fpga_lut.pkl")

# 加载
lut_table = LutTable.load("fpga_lut.pkl")
```

### 从 Profiling 数据构建

```python
from hwnas_fpga.hardware import LutBuilder

builder = LutBuilder()

# 添加 profiling 结果（来自 Vivado HLS 或板卡实测）
builder.add_profiling_result(
    op="conv",
    kernel_size=3,
    in_channels=16,
    out_channels=32,
    stride=1,
    latency_ms=1.5,
    cycles=300,
    dsp=32,
    bram=2,
    lut=1000,
    power_w=2.0,
)

# 构建并保存
lut_table = builder.build()
lut_table.save("my_fpga_lut.pkl")
```

---

## 下一步改进方向

### P1 - 权重共享超网
```
src/hwnas_fpga/training/
└── supernet.py        # 超网训练（待实现）
```

### P2 - RL 搜索器
```
src/hwnas_fpga/search/
└── rl_searcher.py     # 强化学习搜索（待实现）
```

### P2 - 真实声呐数据
```
src/hwnas_fpga/data/
└── sonar.py           # MARIS/UATD 数据集（待实现）
```

---

## 运行示例输出

```
Using device: cpu

=== Creating Search Space ===
Search space created with 4 stages

=== Creating Hardware Estimator ===
Hardware estimator created
  Constraints: max_latency=50.0ms, max_dsp=500, max_bram=1800, max_lut=100000

=== Creating Data Loaders ===
Train samples: 500, Val samples: 100

=== Creating Searcher ===
Searcher created (method=random, seed=42)

=== Testing Baseline Architecture ===
Baseline architecture:
  Params: 1,600
  MACs: 26,656,000
  Latency: 2.98ms
  Peak DSP: 32
  Peak BRAM: 349
  Peak LUT: 1228
  Violations: ()

=== Starting Search ===
Number of candidates to evaluate: 3
[2/3] arch_1: Acc=0.1080, Lat=28.48ms

Search completed!
Total evaluated: 3
Feasible: 1
Infeasible: 2
Best accuracy: 0.1080
LUT stats: {'hits': 5, 'misses': 2, 'total': 7, 'hit_rate': 0.714}

=== Search Results ===
Best architecture: arch_1
  Accuracy: 0.1080
  Latency: 28.48ms
  ...
```

---

## 复用参考代码情况

| 参考库 | 复用内容 | 位置 |
|---|---|---|
| FBNet | 模型构建器设计 | [models/builder.py](src/hwnas_fpga/models/builder.py) |
| FBNet | LUT架构设计 | [hardware/lookup_table.py](src/hwnas_fpga/hardware/lookup_table.py) |
| TinyTNAS | 约束检查逻辑 | [search/searcher.py:66-103](src/hwnas_fpga/search/searcher.py#L66-L103) |
| TinyTNAS | 时间限制搜索 | [search/searcher.py:178-225](src/hwnas_fpga/search/searcher.py#L178-L225) |
| HW-NAS-Bench | 硬件指标设计 | [hardware/cost.py](src/hwnas_fpga/hardware/cost.py) |
| HW-PR-NAS | Pareto排名保持 | [search/pareto.py](src/hwnas_fpga/search/pareto.py) |
