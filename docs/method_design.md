# Hardware-Aware NAS for FPGA-based Sonar Image Processing

## 1. Problem Definition

在资源受限的嵌入式环境中，如何在保证任务精度的同时满足严格的硬件部署约束，是智能水下系统中的核心问题。针对水下声呐图像分类与识别任务，传统 CNN 架构通常针对 GPU 平台设计，缺乏对嵌入式硬件资源与实时性的系统性考虑。因此，将 Hardware-Aware Neural Architecture Search (HW-NAS) 与 FPGA 硬件平台结合，为自动设计可部署的高效神经网络提供了一种可行方案。

本研究构建一个 FPGA 约束下的 HW-NAS 优化问题，目标是在满足硬件资源限制与实时推理需求的条件下，自动搜索兼顾任务性能与硬件效率的 CNN 架构。

整个问题建模可以分为三个阶段：

```text
Stage 1  Problem Inputs
    └── Application & System Constraints

Stage 2  Hardware-Aware NAS Optimization
    ├── Search Space Design
    ├── Search Strategy
    ├── Hardware Cost Estimation
    └── Output: Hardware-constrained architecture

Stage 3  Training & Evaluation
    ├── Train network weights
    ├── Evaluate accuracy
    └── FPGA mapping (HLS / synthesis)
```

该流程最终输出满足 FPGA 部署约束的神经网络结构。

## 2. Problem Inputs

HW-NAS 问题的输入主要包括三个部分。

### 2.1 数据集

输入数据为水下声呐图像数据集及其对应标签，包括：

- 原始声呐图像
- 预处理图像，如去雾、增强
- 分类或检测标签

这些数据用于训练和评估 CNN 的任务性能。

声呐图像具有以下典型特性：

- 单通道灰度图像
- 存在明显斑点噪声
- 目标边界模糊
- 目标尺度变化较大
- 主要依赖纹理和几何结构特征

因此，相比自然图像分类任务，该问题更具挑战性。

### 2.2 目标 FPGA 硬件平台

硬件信息包括：

- FPGA 型号
- LUT 数量
- DSP 数量
- BRAM 容量
- 时钟频率
- 功耗预算

例如可考虑的 FPGA 平台包括：

| FPGA | LUT | DSP | BRAM |
| --- | ---: | ---: | ---: |
| Kintex-7 XC7K325 | 50,950 | 840 | ~16 Mb |
| Zynq-7020 XC7Z020 | 53,200 | 220 | ~4.9 Mb |

这些资源信息构成 NAS 搜索过程中的重要约束条件。

### 2.3 系统性能约束

系统设计需要满足以下约束：

- 推理延迟
- 功耗
- 模型大小
- FPGA 资源利用率

具体约束指标包括：

| 指标 | 含义 |
| --- | --- |
| Resource Utilization | LUT / DSP / BRAM |
| Power | 功耗 |
| Latency | 推理时间 |

这些指标共同构成 HW-NAS 的多目标优化约束。

## 3. Optimization Objective

HW-NAS 本质上是一个多目标优化问题。

优化目标包括：

1. 最大化任务性能
2. 最小化推理延迟
3. 最小化硬件资源消耗

任务性能指标可采用分类准确率或 mAP；硬件资源指标主要包括 LUT、DSP、BRAM 与功耗。

可形式化为：

```text
min over N in search space S:
    L(N) = L_task(N) + beta * C_hardware(N)
```

其中：

- `L_task(N)` 表示任务损失
- `C_hardware(N)` 表示硬件代价，如延迟和资源占用
- `beta` 表示精度与硬件效率之间的权衡系数

目标是在精度与硬件效率之间找到 Pareto 最优解。

## 4. Method Design

根据 HW-NAS 综述，方法设计可以从四个维度进行构建：

1. 搜索空间
2. 搜索策略
3. 加速技术
4. 硬件成本估计

这四个部分共同构成 HW-NAS 方法学框架。

### 4.1 Search Space Design

搜索空间定义了可探索的网络架构范围。本研究采用层次化搜索空间，网络结构参考 EfficientNet 风格的 stage-based 设计。

```text
Network
 ├── Stage 1
 │     └── Block
 ├── Stage 2
 │     └── Block
 ├── Stage 3
 │     └── Block
 └── Stage N
```

每个 Block 可以从以下算子中选择：

- MBConv
- Fused MBConv
- Group Convolution
- Depthwise Convolution

这些算子具有较好的 FPGA 硬件友好性。

Block 搜索参数包括：

- 卷积核大小
- 通道数
- expansion ratio
- block 数量
- stage 深度

搜索粒度包括：

- Block level
- Stage level
- Network level

#### 任务驱动的搜索空间扩展

针对声呐图像特性，搜索空间可进一步引入：

- 多尺度卷积
- 去噪卷积
- 边缘增强结构

这样可以更好地建模纹理特征、模糊边界和不同尺度目标。

### 4.2 Search Strategy

搜索策略用于在巨大搜索空间中寻找最优架构。本研究采用多目标硬件感知搜索策略。

#### 1. Reinforcement Learning

```text
Controller -> sample architecture
          -> evaluate accuracy and latency
          -> update policy
```

奖励函数可写为：

```text
Reward = Accuracy - lambda * Latency
```

超过 FPGA 资源限制的网络直接剪枝。

#### 2. Evolutionary Algorithm

```text
Population
   ↓
Mutation / Crossover
   ↓
Evaluate fitness
   ↓
Selection
```

适应度函数可定义为：

```text
Fitness = Accuracy - alpha * Latency
```

通过 Pareto frontier 保留多目标最优解。

#### 3. Gradient-based NAS

使用可微 NAS，如 DARTS：

- 构建 SuperNet
- 学习架构参数
- 同时优化权重与结构

该方法具有更高搜索效率。

### 4.3 Acceleration Techniques

HW-NAS 的计算成本通常较高，因此需要多种加速技术。

#### 1. 权重共享

使用 SuperNet：

```text
SuperNet
 ├── candidate path 1
 ├── candidate path 2
 └── candidate path N
```

避免为每个架构重新训练。

#### 2. Proxy Evaluation

使用代理数据集或较少 epoch 对候选架构进行快速评估。

#### 3. Early Stopping

提前终止性能较差的架构训练，以节省搜索预算。

### 4.4 Hardware Cost Estimation

HW-NAS 的关键是快速估计硬件性能。本研究采用 LUT-based latency estimation。

#### 1. Operator Profiling

首先对基础算子进行离线 profiling：

- Conv
- Depthwise Conv
- MBConv
- Pooling

通过 Vivado HLS 获得：

- 资源使用
- 时钟周期
- 运行延迟

#### 2. Latency Prediction

网络延迟由各层延迟累加：

```text
Latency_network = sum over all blocks of Latency_block_i
Latency_block = Cycles / Clock Frequency
```

该方法可以快速估计搜索空间中任意架构的硬件延迟。

## 5. Hardware Profiling Pipeline

硬件性能评估流程如下：

```text
NAS Architecture
        ↓
Block Decomposition
        ↓
C/C++ Operator Description
        ↓
Vivado HLS
        ↓
Resource Estimation
        ↓
Latency / Power Prediction
```

HLS 工具可以输出：

- LUT
- DSP
- BRAM
- latency
- power

从而构建硬件性能预测模型。

## 6. Expected Output

HW-NAS 系统最终输出包括：

1. 最优 CNN 网络结构
2. 训练后的模型权重
3. FPGA 可部署实现

输出网络包含：

- 网络层类型
- 卷积核大小
- 通道数
- 网络深度
- block 结构

并生成 FPGA 部署代码或配置文件。

## 7. Repository Mapping

为方便论文方法与工程实现对照，当前项目目录可与方法模块一一对应：

| 方法模块 | 当前代码位置 | 说明 |
| --- | --- | --- |
| Problem Inputs | `configs/`、`data/` | 数据、硬件约束与实验配置 |
| Search Space Design | `src/hwnas_fpga/search_space/` | block/stage/network 编码与合法性检查 |
| Hardware Cost Estimation | `src/hwnas_fpga/hardware/` | 解析式估计器、后续 LUT/HLS 接口 |
| Search Strategy | `src/hwnas_fpga/search/` | 随机搜索、后续进化/可微搜索 |
| Training & Evaluation | `src/hwnas_fpga/training/` | 候选训练、验证、早停与代理评估 |
| FPGA Deployment | `src/hwnas_fpga/deploy/` | 后续 ONNX、HLS、综合与部署接口 |
| Background References | `reference/` | FBNet、HW-NAS-Bench、TinyTNAS 等参考实现 |

## 8. Summary

本文构建了一个面向 FPGA 的 HW-NAS 框架，用于水下声呐图像识别任务。核心思想包括：

1. 构建硬件感知搜索空间
2. 使用多目标搜索策略优化精度与延迟
3. 利用硬件性能预测模型快速评估架构
4. 在 FPGA 资源约束下搜索可部署 CNN

该方法能够在任务性能与硬件效率之间实现自动化协同优化，为嵌入式水下视觉系统提供高效的深度学习模型。
