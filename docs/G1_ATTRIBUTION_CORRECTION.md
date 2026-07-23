# G1 收益归因订正与训练曲线再审计

制定日期：2026-07-21；再审计日期：2026-07-22。本文件订正分析中出现过、并被用于
推导实验优先级的错误结论。**后续引用 G1 差距及其机制时以本文件为准。**

## 一、订正：+0.238 不是「预训练收益」

权威来源：
[`stats.json`](../artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/scratch_v2_closed_set_3method_v1/stats.json)
（严格配对，5 折 × 3 seed）。

| 对比 | mean Δ | 95% CI | paired Cohen's dz |
|---|---:|---|---:|
| pretrained − scratch | **0.0217** | [0.0109, 0.0328] | 0.95 |
| **scratch − NAS** | **0.2163** | [0.2021, 0.2291] | **6.12** |
| pretrained − NAS | 0.2380 | [0.2194, 0.2568] | 5.50 |

**曾经的错误表述**：「换成 ImageNet 预训练 MNV2 = +0.238」，并据此推出
「多声呐数据集预训练的预期收益比算子高一个数量级」。

**正确表述**：0.238 中只有 **0.0217（9%）** 是同架构下 ImageNet 初始化相对
从零训练的增量；剩余 **0.2163（91%）** 是 scratch MNV2 与历史 NAS 候选之间的
架构族、规模及其在当前配方下可优化性的合成差距，不能进一步只归给「参数容量」。
因此「多数据集预训练值一个数量级」**没有直接证据支撑**，只能作为待验证假设。

## 二、再审计：存在当前配方下的欠拟合信号，但机制尚未分离

复现：`python scripts/diagnose_g1_capacity.py`
（只读现有记录与日志，无需 GPU）。机器可读产物：
[`g1_capacity_diagnostic.json`](../artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/g1_capacity_diagnostic.json)；
严格分析报告：
[`analysis-report.md`](../artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/g1_capacity_reaudit_v2/analysis-report.md)。

主比较使用两个各含 **15 × 150 个完整 epoch** 的日志：历史 NAS 候选与正式的
scratch-v2 MNV2 重跑。旧脚本误用了仅 11 次完整运行的旧 scratch 日志，并把不同
运行数的 epoch 行直接混合平均；现已按 `(fold, seed)` 先汇总、再严格配对。

| 方法 | epoch 140–150 在线增强 `train_acc` | best_epoch 中位 |
|---|---:|---:|
| **NAS rl_arch_135** | **0.6824** | 97 |
| MNV2 scratch-v2 | **0.9878** | 110 |

配对差为 **+0.3054**（scratch − NAS），配对 bootstrap 95% CI
**[0.3010, 0.3102]**，精确双侧符号翻转 `p = 6.10e-5`。这证明差距已出现在
当前训练配方下的训练侧，不只是 outer validation（外层验证集）上的泛化差距。

但日志中的 `train_acc` **不是无增强训练集准确率**：它在 `model.train()` 下，针对
随机翻转、旋转、仿射、亮度/对比度扰动及概率性斑点噪声后的在线 batch 计算。因此，
0.6824 不能直接表述成「连自己的原始训练集都拟合不了」。同一训练配方也不等于
控制了架构特异的优化难度。

当前证据边界是：

- **支持**「当前配方下存在明显欠拟合/训练侧瓶颈信号」；
- **不支持**把机制唯一归为参数容量；容量、优化器/学习率兼容性、正则与增强强度、
  以及 train/eval mode 行为仍然混杂；
- 15 次 NAS 运行只有 1 次在 epoch 150 取得最佳 inner macro_f1，中位为 97，说明
  机械延长同一 schedule 不是最高信息量的动作，但不能据此排除优化问题；
- 日志不呈现经典的「训练接近 1、验证落后」过拟合形态，但这不是对所有过拟合或
  捷径机制的完全排除。

### 已执行的机制门诊（2026-07-22）

统一审计报告：
[`g1_mechanism_triage_v3/analysis-report.md`](../artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/diagnostics/g1_mechanism_triage_v3/analysis-report.md)。

**第一门禁：15 对 best checkpoint 的 clean-train 重评。** 每次运行只在其自身
`split.train_indices` 上使用无增强 `eval_dataset`，不消费 inner/outer 索引：

| 指标 | NAS mean | scratch mean | 配对差（scratch − NAS） | 95% CI |
|---|---:|---:|---:|---|
| clean top-1 | 0.8357 | 0.9967 | **0.1610** | [0.1471, 0.1746] |
| clean macro_f1 | 0.7369 | 0.9798 | **0.2429** | [0.2294, 0.2564] |

因此 clean-fit gap（无增强训练拟合差距）是真实的，不只是在线增强指标的假象；但它
仍然没有区分容量与训练配方。

**第二门禁：96 张类平衡 clean 子集的 micro-overfit。** 新鲜初始化的 NAS 在
`lr=1e-3`（冻结式正则或 plain CE）和 `lr=3e-3` 下均达到 top-1/macro_f1=1.000；
`lr=3e-4` 最好约 0.979。它排除了「架构存在粗粒度实现故障、连小子集都不能记忆」，
并确认学习率敏感性，但不能回答完整数据容量。

**第三门禁：fold0/seed42 全训练索引的三臂因子诊断。** 所有新训练只使用 train 与
inner，**从未消费 outer validation**，因此下表是机制证据而非正式泛化性能：

| 训练臂 | best clean top-1 | best clean macro_f1 | best inner macro_f1 | 执行 epoch |
|---|---:|---:|---:|---:|
| 无增强 + 冻结 loss/正则，lr=1e-3 | 0.9394 | 0.8868 | 0.6122 | 150 |
| 无增强 + plain CE/无 weight decay，lr=1e-3 | **1.0000** | **1.0000** | **0.8266** | 90（早停） |
| 无增强 + plain CE/无 weight decay，lr=3e-3 | **1.0000** | **1.0000** | 0.7564 | 65（早停） |

历史原配方同一 fold/seed 的 best checkpoint clean top-1/macro_f1 为 0.8228/0.7476，
inner macro_f1 为 0.6733。由此可见，18.8K 参数架构在完整 clean 训练索引上**可以**
充分拟合；此前的低训练拟合不能主要归为通用表达容量上限。当前被确认的是一个
**loss/正则组合敏感性**（同时改变 label smoothing、logit adjustment 与 weight
decay，尚未分离到单个因子），随机增强也有次级影响。单折 inner 的提高很有希望，
但未经 outer-fold 验证，不能写成正式精度增益。

### 当前决策

1. **不能从这条曲线判断蒸馏上限**。蒸馏既可能改善泛化，也可能改变小模型的优化
   景观；当前应在配方因子验证之前保持 `HOLD`，而不是优先执行。
2. **不能就此断言「18.8K 参数的上限就是 0.694」**。该候选是在**有泄漏的
   fold-0 协议**下、用**无判别力的 3-epoch 代理**（前四候选 proxy macro_f1
   挤在 0.626–0.642，而 retrain 差 0.115）选出来的，它不是该预算下可达最优的证据。
3. **容量扫描为 `HOLD`**。下一门禁是把最小因子对照（无增强条件下的冻结 loss/正则
   vs plain CE/无 weight decay，均为 `lr=1e-3`）补到 fold0 的 seed43、44，共 4 个新
   单元；若方向一致，再决定是否投入正式 15-run 协议比较。只有配方验证失败，才重新
   放行容量扫描。
4. **四臂预处理实验暂不启动**。必须先具备按源图或采集场次分组的 split 合约，
   否则无法区分物理尺度信息与数据制作捷径。

## 三、订正：NKSID 原始像素几何的预测力**不能**当作可实现增益

**观测（成立）**：仅用两个特征（原生面积对数、长宽比）的 1-NN，在随机分层
留出上达到 **68.8% top-1**（多数类基线 36.3%）。各类原生中位边长从
~46px（small_propeller）到 ~289px（iron_pipeline），resize 到 224 的缩放系数
在 **0.78×–4.88×** 之间。

**曾经的错误表述**：「保住原生尺度预期可带来 +0.3 量级收益」。

**正确表述**：原始像素几何对类别具有较强预测性，但**尚不能确定它是可泛化的
物理尺度信息，还是数据制作流程（逐类裁剪/重采样）产生的捷径**。二者需要用
**按源图/采集场次分组的 split** 来判开：若保尺度的收益在随机 split 下存在、
在分组 split 下消失，即为捷径。

## 四、其他需要收窄的表述

| 曾经的表述 | 订正 |
|---|---|
| 用「每 MAC 精度」作搜索目标 | **不采用**。比值会奖励「极小但很差」的模型，且 MAC ≠ FPGA 代价。维持硬约束下的多目标 Pareto（macro_f1 / 实测 latency / LUT / DSP / BRAM / power / feasibility） |
| 「119× 参数差 = 部署优势」 | 降级为**部署优势假设**。参数量不等于 latency/LUT/DSP/BRAM/功耗，且 MNV2 与小模型尚未走同一完整板级协议 |
| Figshare「866 张纯背景」 | **违反边界文档**。空 YOLO 文件语义未知（见 `EXTERNAL_SONAR_DATASETS.md`）。自监督可用；监督分类前必须先确认语义 |
| 「文献里没有声呐 NAS」 | 过头。已有面向声呐目标检测的 NAS-DETR；条形核在声学场景 NAS 有先例（间接证据，不能替代侧扫验证） |
| 三个数据集「合并预训练」 | 标签空间不同（NKSID 8 类切片分类 / Figshare MILCO-NOMBO / Roboflow 三标签检测），**只能**用于自监督或多任务分头，不能共用监督标签 |
| 空洞卷积「零额外成本」 | 参数不增，但会**扩大 FPGA 行缓冲与访存**，需单独成本证据 |
| SE 是「强正则」 | SE 是注意力模块，不是正则；其 sigmoid/FC 仍需独立 INT8/HLS 证据 |
| rl_arch_135 是「NAS 冠军」 | 应称**冻结的历史 NAS 候选**（`legacy_fold0_selected`） |

## 五、跨年份评估的可行性限制

Figshare 按组的框数分布（本地清点）：

| 组 | 图像 | 有框图 | MILCO | NOMBO | 框数 |
|---|---:|---:|---:|---:|---:|
| 2010 | 345 | 28 | 22 | 12 | 34 |
| 2015 | 120 | 118 | 242 | 171 | 413 |
| 2017 | 93 | 19 | 28 | **2** | 30 |
| 2018 | 564 | 112 | 96 | 46 | 142 |
| 2021 | 48 | 27 | 49 | **0** | 49 |

**2021 组 NOMBO=0、2017 组 NOMBO=2**，简单的留一年二分类 macro_f1 在这些折上
直接退化；2015 一组占 413/668 框，与 2010（28/345 图有框）不是同一采集体制。
跨年实验必须报告 **per-class recall/AP**，不能机械套用二分类 macro_f1。
