# 当前方法说明与实现依据

核对日期：2026-09-11。本文说明当前工作区的实际方法；配置、源码和绑定产物决定运行行为。旧草稿已保存到[修改前备份](../outputs/project_cleanup_20260909/method_docs_before.zip)。源码导航见[architecture](architecture.md)，配置差异见[配置版本索引](SEARCH_CONFIG_CANONICAL.md)。

## 1. 分类方法的范围

NKSID 分类配置输入为单通道 `224×224`，输出 8 类。分类选择指标为 macro_f1（宏平均 F1），同时记录 top1（分类准确率）。当前 ZCU104 分类配置使用 `mobilenetA_full_dmdc_cfem_v1` 搜索空间和原生 RL（强化学习）搜索器；每个候选独立训练，随后按 Pareto（多目标非支配）关系筛选。

方法结果是候选架构、训练记录和硬件代价估计。能否部署，需要该候选后续的数值一致性、完整网络 HLS / Full Route（完整布局布线）和板级验证。RL-only 范围不包含与 Random、Aging、Proxyless 或 SPOS 的优劣比较。

## 2. 八阶段分类搜索空间

来源：[20-episode 配置](../configs/search/zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml)与[200-episode 预算变体](../configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml)。前者保留旧文件名，但内容已经是八阶段 MobileNetA。下面统一用从 0 开始的代码阶段索引，避免把插入槽位遗漏后误数为七阶段。

| 位置 | 输入→输出通道 | 输出分辨率 | 实际选择 |
|---|---|---|---|
| Stem | 1→32 | 112×112 | 固定 3×3，stride=2 |
| stage[0] | 32→16 | 112×112 | 固定 1×1 Conv |
| stage[1] | 16→24 | 56×56 | MBConv，k∈{3,5}，e∈{3,6} |
| stage[2] | 24→32 | 28×28 | MBConv，k∈{3,5}，e∈{3,6} |
| stage[3] 目标槽位 | 32→32 | 28×28 | Skip、MBConv-k3/e3、MBConv-k5/e3、DMDC、CFEM |
| stage[4] | 32→64 | 14×14 | MBConv，k∈{3,5}，e∈{3,6} |
| stage[5] | 64→96 | 14×14 | MBConv，k∈{3,5}，e∈{3,6} |
| stage[6] | 96→160 | 7×7 | MBConv，k∈{3,5}，e∈{3,6} |
| stage[7] | 160→320 | 7×7 | MBConv，k∈{3,5}，e∈{3,6} |
| 分类头 | 320→1280→8 | 全局汇聚 | 1×1 Conv、GAP（全局平均池化）、FC（全连接） |

这些配置的阶段宽度、深度均固定，深度均为 1；搜索阶段内算子。Skip 仅允许 stride=1 且输入输出通道相同；stage[5]/stage[7] 的 YAML 虽列出 Skip，但该通道转换不合法，不能按配置列表直接相乘计算空间大小，也不能解释成带投影的跳连。合法性由 [SearchSpace](../src/hwnas_fpga/search_space/space.py) 和搜索器处理。

## 3. DMDC 与 CFEM 的实现

权威实现是 [models/builder.py](../src/hwnas_fpga/models/builder.py)。以下描述对应本地代码，不把模板存在当作软件与硬件已完成数值等价验证。

| 算子 | 本地计算结构 | 使用条件 |
|---|---|---|
| `DMDCConvSonarBlock` | 三条 dilation=1/3/5 分支；各分支结合空洞卷积特征与区域池化特征，乘法融合后投影；拼接三分支并用 1×1 压缩 | 输入/输出均32通道，stride=1 |
| `CFEMSonarBlock` | dilation=2/3/5 的三尺度分支；平均/最大池化描述子生成尺度权重；上下文与输入投影通过注意力融合，再经通道/空间加权与 sigmoid 输出 | 输入/输出均32通道，stride=1 |

分类目标槽位是 28×28；检测的 640 输入对应目标槽位是 80×80。Python 构造器的通道/步长约束不等于硬件覆盖所有分辨率。相关 HLS 模板在 [templates](../hls_lut_builder/templates) 中的 `dmdc_conv_sonar.cpp.tmpl` 和 `cfem_sonar.cpp.tmpl`。旧 `denoise`、`edge`、Dir-v1 与这两个算子不是同一实现，旧结论不能直接转移。

## 4. RL、奖励与筛选

[run_search.py](../run_search.py) 复用 [RLSearcher / Controller / RewardFunction](../src/hwnas_fpga/search/rl_searcher.py)、[共享搜索逻辑](../src/hwnas_fpga/search/searcher.py)及 [Pareto 实现](../src/hwnas_fpga/search/pareto.py)。当前配置使用控制器温度、熵正则、探索概率和重复架构惩罚；硬件归一化为 `raw_space_minmax`，硬件标量化为 `pareto_rank`。精确权重以保存的配置与运行记录为准。

控制器是按位置输出分类分布的 MLP（多层感知机），没有 RNN（循环神经网络）的递归状态。多个奖励实现由 `_compute_controller_reward` 互斥分派，应保留这一入口。当前主项是 `macro_f1 − hardware_rank_weight × normalized_hardware_rank`，再叠加适用的重复、不可行和探索处理；不能将图示延迟/资源加权和直接作为当前执行公式。

2026-09-10 修复后，硬件参考集采用固定 seed=0 的均匀索引抽样。参考集之外的候选根据支配它的参考点计算插入等级，除数固定为 `max(1, max_reference_rank + 1)`，不再因未命中而默认免除硬件惩罚。它是固定参考集的近似等级，并非全部 20,480 架构的精确 Pareto 等级。奖励协议保存版本、参考集 SHA256 和搜索代理配方；不同协议的断点不能直接续接。

两份分类配置用 `search.proxy_training` 显式记录轻量搜索代理：独立 AdamW、lr=0.001、weight_decay=0.0001、batch_size=8，无 AMP（自动混合精度）、梯度累积、预热、标签平滑或 logit 调整。完整重训练仍使用 `training`，其中 AMP=true、梯度累积=4 等仅属于重训练。代理暂不支持的非默认选项会报错，实际解析配方写入运行配置与候选 history。需要统一两套训练算法时应另行决定，不能仅复制配置字段。

显式 Pareto 目标列表现在仍会合入已启用的物理风险目标，并强制最小化。候选冻结直接消费原运行 `pareto_representatives.json` 的角色，包含 Top-K 之外的代表点；配置、算子表文件及载荷哈希、候选身份和指标必须与该运行一致。

`episodes` 是 RL 循环预算；实际评估数需读取产物，不能把 `num_candidates` 或 Pareto Top-k 当作额外训练预算。候选 early stopping（早停）、运行超时、约束处理与证据准入分别生效。两份分类配置的预算与约束不同，不能合并为同一冻结协议。

配置中的 Pareto 目标包含 macro_f1、latency_ms（延迟）、energy_mj（能耗）、DSP（数字信号处理单元）、BRAM（块 RAM）、LUT（查找表资源）。功率和能耗在搜索阶段的来源如下。

## 5. 硬件代价的真实口径

[runtime.py](../src/hwnas_fpga/runtime.py) 已接入 [ZCU104MobileNetAOperatorCostEstimator](../src/hwnas_fpga/hardware/mobilenetA_operator_cost.py)，检查板卡、时钟、版本、哈希与覆盖。当前配置指向 `mobilenetA_full_dmdc_cfem_v3_gap1280`。该表现已包含真实 `global_avg_pool/gap_7_1280` 的 HLS/route 记录；本轮重新读取服务器原始报告，核对 HLS SHA256、21,483 周期及 route 指标，并通过本地表/coverage/preflight 哈希检查。88 行记录齐全，其中 80 行满足记录的时序条件；这不表示每个候选或完整网络均通过。历史 v2_recovered 只有 `gap_7_320`，不符合分类头 1280 通道，当前代码会拒绝它，即使历史 coverage/preflight 标记通过。旧 RAW20 / 9 行 CFEM20 适配器已退役。

| 搜索字段 | 当前聚合来源 | 能证明的范围 |
|---|---|---|
| `latency_ms` | Σ(HLS 周期数) / (200 × 1000) | 冻结 200MHz 的算子模型估计，不是板端实测延迟 |
| `lut` / `ff` / `dsp` | 算子布局布线资源求和 | 聚合资源估计，不是完整网络布局布线结果 |
| `bram` | 算子 BRAM tile 求和后向上取整；物理 BRAM18 压力为 2×聚合 tile / 624 | 搜索预算以 tile 表示，物理容量校验转换为 BRAM18 |
| `power_w` | 算子 `power_w_vivado_vectorless` 的最大值 | Vivado 无向量功率代理，不是板卡实测功率 |
| `energy_mj` | Σ(算子周期数 / 200000 × 算子无向量功率 W) | 使用同一固定时钟的搜索能耗代理，不是实测单次推理能耗 |

200-episode 变体另启用物理风险筛选。其中 `route_lutram_pressure` 是聚合 LUT 与配置容量的比值代理，不能解释成实际 LUTRAM 布线占用率。2026-09-11 按用户要求将 BRAM18 压力≤1设为硬约束，聚合 BRAM 预算同步由700收紧到312 tile（624 BRAM18）。当前硬限制清单见[限制与验证记录](../outputs/method_conflict_fixes_20260909/bram_hard_limit_20260911/HARD_LIMITS.md)。

`feasible` 表示当前搜索配置和估计器下的可接受性；完整网络能否通过 HLS、布局布线和板测，必须查对应候选证据。板载 INA226/hwmon 的整板输入功率采集另见[分类记录协议](zcu104_full_experiment/NKSID_RL_CLASSIFICATION_RECORDING.md)，不能用上表代理补齐。

整数参考已修复模块式与函数式 ReLU6 的上界截断。DMDC/CFEM 所需的 sigmoid、双线性插值和非全局自适应池化整数支持，以及八阶段完整网络部署验证仍未闭合；ReLU6 回归通过不代表完整网络 INT8 已通过。逐项修复状态见[修复记录](../outputs/method_conflict_fixes_20260909/REPORT.md)，原审计保留为历史证据。

## 6. 分类评估与检测分支

固定分类候选通过 [run_eval_protocol.py](../run_eval_protocol.py) 进行外层五折、多种子评估；内层用于选择，外层用于最终评价。固定候选评估不等于每个外层单元独立 NAS。源码冻结、checkpoint 和协议版本要求见[评估说明](EVAL_PROTOCOL.md)。

检测配置为[640 输入、多尺度、固定 neck/head](../configs/search/zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml)：三类框标注，取 stage[3/5/7] 的步长8/16/32特征，使用固定 FPN（特征金字塔）和检测头，以 mAP（平均精度均值）、precision、recall、F1 评价。

当前 [run_detection_experiment.py](../scripts/run_detection_experiment.py) 调用 `baseline_architecture()` 后训练与评估，没有执行 RL 搜索循环。配置写有 `method: rl` 和 200 episodes 只是配置意图，不能据此报告检测 NAS 已实现。检测补充硬件表的固定 neck/head 覆盖也不能替代完整骨干、完整网络布局布线或板端证据。
