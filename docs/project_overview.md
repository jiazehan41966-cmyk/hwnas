# HW-NAS FPGA Sonar 椤圭洰浠嬬粛

> Current formal MobileNetV2 mainline: `mbconv`, `fused_mbconv`, `denoise`, `edge`, `skip`.
>
> `dw_pw_conv` remains available only for historical or lightweight-compatible
> profiles and no longer represents the default formal search path.


> 面向水下声呐图像分类/识别任务的硬件感知神经架构搜索系统

## 一、业务目标

### 核心目标

开发一套面向水下声呐图像分类/识别任务的硬件感知神经架构搜索（HW-NAS）系统，在 FPGA 资源约束下实现从搜索、训练到部署的完整闭环。

### 具体目标

1. **硬件感知优化**：针对 FPGA 资源（DSP、BRAM、LUT）特定约束，自动搜索最优神经网络架构
2. **多目标平衡**：在精度、延迟、功耗、资源占用之间取得 Pareto 最优
3. **声呐图像专用**：针对单通道声呐图像（边界模糊、纹理重要、噪声特性强）优化搜索空间
4. **可部署性**：确保搜索出的模型可直接量化和映射到 FPGA 硬件

## 二、核心功能

### 1. 硬件感知搜索空间定义

- 支持 stage-based 层次化结构（4 个 stage）
- 可搜索参数：通道数(width)、深度(depth)、算子类型(op)、卷积核大小(kernel)
- 当前正式 MobileNetV2 主线支持 5 种可搜索算子：
  - `mbconv`：MobileNetV2 风格 MBConv
  - `fused_mbconv`：融合 MBConv
  - `denoise`：声呐友好去噪扩展算子
  - `edge`：声呐友好边缘增强算子
  - `skip`：跳过连接
- `dw_pw_conv` 仍保留在历史/轻量兼容 profile 中，但不再代表当前正式主线。

### 2. FPGA 硬件代价建模

| 资源类型 | 估计方法 |
|---------|---------|
| **DSP** | 基于并行度和量化位宽计算 |
| **BRAM** | 基于权重和激活值存储需求 |
| **LUT** | 基于算子复杂度和控制逻辑 |
| **延迟** | 基于频率、MAC 数和流水线效率 |
| **功耗** | 基于资源占用和工作负载 |

### 3. 多目标搜索算法

| 搜索器 | 状态 | 说明 |
|--------|------|------|
| `RandomSearcher` | ✅ 已实现 | 随机采样 + 硬约束过滤 |
| `EvolutionSearcher` | ⏳ 待实现 | 进化算法搜索 |
| `DifferentiableSearcher` | ⏳ 待实现 | 可微搜索（DARTS 风格） |

### 4. 端到端验证流程

- 虚拟数据集快速验证
- 候选架构训练与精度评估
- 硬件可行性检查与约束验证
- Pareto 前沿选择

## 三、整体架构

### 分层架构（6 层设计）

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. 数据与任务层                              │
│         声呐图像输入 / 标签 / 预处理 / 数据增强                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    2. 搜索空间层                                │
│         架构编码 / 采样 / 合法性检查 / 空间配置                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  3. 硬件代价建模层                              │
│       DSP/BRAM/LUT 估计 / 延迟预测 / 功耗建模 / 约束检查         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    4. 搜索器层                                  │
│         随机搜索 / 进化搜索 / 可微搜索 / Pareto 选择             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  5. 重训练与评估层                              │
│         候选训练 / 精度验证 / 超网训练（待实现）                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  6. 部署映射层（待实现）                         │
│         模型导出 / 量化 / ONNX / HLS 映射 / FPGA 实测            │
└─────────────────────────────────────────────────────────────────┘
```

### 核心模块组成

```
src/hwnas_fpga/
├── interfaces.py              # 跨模块共享数据结构
├── data/                      # 数据集与预处理
│   ├── __init__.py
│   └── dataset.py            # DummySonarDataset, SonarImageDataset
├── search_space/              # 搜索空间定义
│   ├── __init__.py
│   └── space.py              # SearchSpaceConfig, ArchitectureSpec, SearchSpace
├── hardware/                  # 硬件代价建模
│   ├── __init__.py
│   ├── cost.py               # FPGACostEstimator, CostEstimate, LayerCost
│   └── lookup_table.py       # LUT 查找表（待实现）
├── models/                    # 模型构建
│   ├── __init__.py
│   └── builder.py            # HWNASModel, build_model, 各种 Block
├── search/                    # 搜索算法
│   ├── __init__.py
│   ├── searcher.py           # BaseSearcher, RandomSearcher
│   ├── pareto.py             # Pareto 前沿选择（待实现）
│   └── constrained.py        # 约束搜索（待实现）
├── training/                  # 训练逻辑
│   ├── __init__.py
│   └── trainer.py            # train_model
└── deploy/                    # 部署映射（待实现）
    └── __init__.py
```

## 四、数据实体设计

### 1. 架构表示实体

```python
# 搜索空间配置
@dataclass
class SearchSpaceConfig:
    input_channels: int = 1           # 输入通道（声呐为单通道）
    image_size: int = 224             # 输入图像大小
    stem_channels: int = 16           # Stem 输出通道
    stage_strides: tuple = (1,2,2,2)  # 每个 stage 的步长
    channel_choices: tuple            # 可选通道数
    depth_choices: tuple              # 可选深度
    kernel_choices: tuple             # 可选卷积核大小
    expand_choices: tuple             # 可选扩展比例
    op_choices: tuple                 # 可选算子类型

# 块规范
@dataclass
class BlockSpec:
    op: str                           # 算子类型
    kernel_size: int = 3              # 卷积核大小
    expand_ratio: int = 1             # 扩展比例
    stride: int = 1                   # 步长

# 阶段规范
@dataclass
class StageSpec:
    channels: int                     # 输出通道数
    depth: int                        # 块数量
    stride: int                       # 阶段步长
    blocks: tuple[BlockSpec, ...]     # 块列表

# 架构规范
@dataclass
class ArchitectureSpec:
    input_channels: int               # 输入通道
    stem_channels: int                # Stem 通道
    stages: tuple[StageSpec, ...]     # 阶段列表
    head_channels: Optional[int]      # 分类头通道
    num_classes: Optional[int]        # 类别数
```

### 2. 硬件实体

```python
# 硬件规格
@dataclass
class HardwareSpec:
    name: str                         # 硬件名称
    clock_mhz: int                    # 时钟频率
    max_lut: Optional[int]            # 最大 LUT 数量
    max_ff: Optional[int]             # 最大触发器数量
    max_bram: Optional[int]           # 最大 BRAM 块数
    max_dsp: Optional[int]            # 最大 DSP 数量
    max_power_w: Optional[float]      # 最大功耗

# 搜索约束
@dataclass
class SearchConstraints:
    max_latency_ms: Optional[float]   # 最大延迟
    max_energy_mj: Optional[float]    # 最大能耗
    max_model_size_mb: Optional[float]# 最大模型大小
    max_lut: Optional[int]            # 最大 LUT
    max_bram: Optional[int]           # 最大 BRAM
    max_dsp: Optional[int]            # 最大 DSP

# 候选指标
@dataclass
class CandidateMetrics:
    accuracy: Optional[float]         # 精度
    latency_ms: Optional[float]       # 延迟
    energy_mj: Optional[float]        # 能耗
    lut: Optional[int]                # LUT 占用
    bram: Optional[int]               # BRAM 占用
    dsp: Optional[int]                # DSP 占用
    power_w: Optional[float]          # 功耗
```

### 3. 代价估计实体

```python
# 层代价
@dataclass
class LayerCost:
    stage_index: int                  # 阶段索引
    block_index: int                  # 块索引
    op: str                           # 算子类型
    params: int                       # 参数数量
    macs: int                         # MAC 运算数
    weight_bytes: int                 # 权重存储
    activation_bytes: int             # 激活存储
    ideal_dsp: int                    # 理想 DSP 需求
    allocated_dsp: int                # 分配 DSP 数量
    bram_blocks: int                  # BRAM 块数
    lut: int                          # LUT 数量
    latency_cycles: int               # 延迟周期数

# 整体估计
@dataclass
class CostEstimate:
    params: int                       # 总参数
    macs: int                         # 总 MAC
    model_size_mb: float              # 模型大小
    peak_dsp: int                     # 峰值 DSP
    peak_bram: int                    # 峰值 BRAM
    peak_lut: int                     # 峰值 LUT
    latency_ms: float                 # 总延迟
    power_w: float                    # 功耗
    energy_mj: float                  # 能耗
    violations: tuple[str, ...]       # 违规项
    per_layer: tuple[LayerCost, ...]  # 逐层代价
```

## 五、业务流程

### 1. 搜索主流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       初始化阶段                                │
├─────────────────────────────────────────────────────────────────┤
│  1. 创建搜索空间 SearchSpace                                    │
│  2. 创建硬件估计器 FPGACostEstimator                            │
│  3. 准备数据集 DataLoader                                       │
│  4. 创建搜索器 RandomSearcher                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       基线评估                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. 生成基线架构（最小配置）                                     │
│  2. 评估硬件代价（验证估计器正确性）                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    搜索循环（约束驱动）                          │
├─────────────────────────────────────────────────────────────────┤
│  for i in range(num_candidates):                                │
│      1. 采样候选架构 space.sample()                              │
│      2. 硬件代价估计 estimator.estimate()                        │
│      3. 可行性检查 check_feasibility()                           │
│          ├── 可行 → 训练 → 精度评估 → 记录                       │
│          └── 不可行 → 记录违规 → 跳过                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       结果整理                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. 按精度/延迟排序                                             │
│  2. 提取 Pareto 前沿                                            │
│  3. 输出最佳候选                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 硬件代价估计流程

```
输入: ArchitectureSpec
           ↓
┌──────────────────────────────┐
│  1. 解析块级信息              │
│     resolve_blocks()         │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  2. 逐层计算复杂度            │
│  ├── 参数数量 params         │
│  ├── MAC 运算量 macs         │
│  ├── DSP 需求 ideal_dsp      │
│  ├── BRAM 需求 bram_blocks   │
│  └── LUT 需求 lut            │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  3. 聚合整体代价              │
│  ├── 总延迟 = Σ层延迟         │
│  ├── 峰值资源 = max(各层)     │
│  └── 功耗估计                 │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  4. 约束检查                  │
│  ├── 比对硬约束               │
│  └── 返回违规项列表           │
└──────────────────────────────┘
           ↓
输出: CostEstimate
```

### 3. 候选评估流程

```
输入: 可行候选架构 ArchitectureSpec
           ↓
┌──────────────────────────────┐
│  1. 构建 PyTorch 模型         │
│     build_model()            │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  2. 快速训练                  │
│     train_model()            │
│     (1-3 epochs)             │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  3. 验证集评估                │
│     计算 accuracy            │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  4. 记录指标                  │
│     CandidateMetrics         │
└──────────────────────────────┘
           ↓
输出: SearchCandidate
```

## 六、关键技术特征

### 1. 声呐优化设计

| 特征 | 设计决策 |
|------|---------|
| **输入适配** | 单通道输入（声呐灰度图像） |
| **算子选择** | FPGA 友好算子集（避免复杂注意力） |
| **量化策略** | 8-bit 量化为主（FPGA 友好） |
| **激活函数** | ReLU/ReLU6（避免复杂非线性） |

### 2. 约束驱动搜索

**硬约束（违反即淘汰）：**
- `latency <= max_latency_ms`
- `DSP <= max_dsp`
- `BRAM <= max_bram`
- `LUT <= max_lut`
- `power <= max_power_w`
- `model_size <= max_model_size_mb`

**软目标（优化方向）：**
```
score = accuracy
        - λ_lat × norm_latency
        - λ_eng × norm_energy
        - λ_res × norm_resource
```

### 3. 两级硬件评估

| 级别 | 用途 | 速度 | 精度 |
|------|------|------|------|
| **快速估计器** | 搜索阶段高频调用 | 毫秒级 | 中等 |
| **精确实测器** | Top-K 候选复验 | 分钟级 | 高 |

## 七、参考架构与复用情况

| 参考项目 | 复用内容 | 项目中的实现 |
|---------|---------|------------|
| **FBNet** | 模型构建器设计 | `models/builder.py` |
| **FBNet** | LUT 架构设计 | 参考 `reference/FBNet/mobile_cv/lut/` |
| **TinyTNAS** | 约束检查逻辑 | `search/searcher.py:check_feasibility()` |
| **TinyTNAS** | 时间限制搜索 | `search/searcher.py:search_with_timeout()` |
| **HW-NAS-Bench** | 硬件指标设计 | `interfaces.py`, `hardware/cost.py` |
| **DARTS** | 保留为基线 | 待实现 |

## 八、当前状态与路线图

### ✅ 已实现

| 模块 | 功能 | 文件 |
|------|------|------|
| 搜索空间 | 完整定义、采样、验证 | `search_space/space.py` |
| 硬件估计 | DSP/BRAM/LUT/延迟/功耗 | `hardware/cost.py` |
| 模型构建 | 5 种算子、Stem/Head | `models/builder.py` |
| 数据加载 | 虚拟数据集 | `data/dataset.py` |
| 随机搜索 | 约束过滤、时间限制 | `search/searcher.py` |
| 训练评估 | 快速训练、精度验证 | `training/trainer.py` |
| 入口脚本 | 端到端流程 | `run_search.py` |

### ⏳ 待实现

| 优先级 | 模块 | 功能 | 计划文件 |
|--------|------|------|----------|
| **P0** | LUT 加速 | 查找表代替计算 | `hardware/lookup_table.py` |
| **P1** | 超网训练 | 权重共享训练 | `training/supernet.py` |
| **P1** | Pareto 选择 | 多目标优化 | `search/pareto.py` |
| **P1** | 真实数据 | 声呐图像加载 | `data/sonar.py` |
| **P2** | 进化搜索 | 进化算法 | `search/evolution.py` |
| **P2** | 可微搜索 | DARTS 风格 | `search/differentiable.py` |
| **P3** | 模型导出 | ONNX/量化 | `deploy/export.py` |
| **P3** | FPGA 映射 | HLS 接口 | `deploy/hls.py` |

### 🔄 迭代计划

```
Milestone 1: 跑通最小链路 ✅
├── 固定小型搜索空间
├── 软件估计器代替 HLS
└── 虚拟数据集验证

Milestone 2: 接入真实约束（进行中）
├── 定义目标板卡资源上限
├── 接入 HLS 综合结果
└── 校准估计器误差

Milestone 3: 论文实验闭环
├── 与手工 CNN 基线对比
├── 与 DARTS/FBNet 基线对比
└── 输出 Pareto 曲线和消融实验
```

## 九、快速开始

### 运行搜索

```bash
# 最小化测试（3 个候选，1 个 epoch）
python3 run_search.py --num-candidates 3 --train-epochs 1 --batch-size 8

# 完整搜索（20 个候选，3 个 epoch）
python3 run_search.py --num-candidates 20 --train-epochs 3 --batch-size 32

# 带时间限制的搜索（60 分钟）
python3 run_search.py --timeout-minutes 60 --train-epochs 3 --batch-size 32
```

### 示例输出

```
Using device: cpu

=== Creating Search Space ===
Search space created with 4 stages

=== Creating Hardware Estimator ===
Hardware estimator created
  Constraints: max_latency=50.0ms, max_dsp=500, max_bram=1800, max_lut=100000

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
[1/3] arch_0: Acc=0.1200, Lat=15.32ms
[2/3] arch_1: Acc=0.1080, Lat=28.48ms

Search completed!
Total evaluated: 3
Feasible: 2
Infeasible: 1
Best accuracy: 0.1200

=== Search Results ===
Best architecture: arch_0
  Accuracy: 0.1200
  Latency: 15.32ms
  ...
```

## 十、项目价值

1. **工程价值**：为声呐图像处理提供自动化 FPGA 神经网络设计
2. **研究价值**：探索硬件约束下的 NAS 新方法
3. **部署价值**：确保搜出的模型可直接部署到资源受限平台
4. **可扩展性**：模块化设计支持扩展到其他传感器类型

---

*文档更新时间: 2024-03*
