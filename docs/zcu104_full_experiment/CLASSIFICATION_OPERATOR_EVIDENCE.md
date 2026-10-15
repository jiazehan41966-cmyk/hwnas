# NKSID 分类算子证据表（MBConv / DMDC / CFEM）

整理日期：2026-09-14。本页只汇总已有产物，不启动训练、搜索或板测。分类任务与检测任务分开记录；分类 macro_f1（宏平均 F1）上升不得写成检测定位改善。

## 1. 核对范围与不可直接比较的协议

| 系列 | 数据划分 | 种子 | 训练预算 | 目标槽位 | 硬件表 | 可否与正式评估并列 |
|---|---|---|---|---|---|---|
| v3gap1280 RL200 正式重训 | `nksid_outer5fold_inner_contiguous_v1`，5 外层折 × seeds 42/43/44 | 42/43/44 | 搜索代理 10 epoch；正式评估 retrain 150 epoch | 八阶段 `stage[3]`：32→32，28×28，stride=1 | `mobilenetA_full_dmdc_cfem_v3_gap1280` | **正式评估主表**；`source_freeze_verified=true` |
| 20260908 RL200 重训 | 同上 5×3 | 42/43/44 | 同上 10 / 150 | 同上 `stage[3]` | 搜索时绑定 `v2_recovered`（GAP320，现已被估计器拒绝） | 软件数字可引用，但 **source_freeze 未过**，与 v3 表 **不可直接比较硬件或 freeze 身份** |
| v3gap1280 / 20260908 搜索代理 | 外层折 0 + 内层 15%（搜索不消费外层验证） | 搜索 seed=42 | 每候选 10 epoch，早停 patience=2 | 同上 | 分属 v3 / v2 | **短训练代理**；不得替代 150 epoch 外层评价 |
| fourstage CFEM20 v1/v2 | 历史四阶段槽位，非现行八阶段 | 多折多种子，eval10 | 20 episode 级；延迟上限 25 ms | 四阶段末槽 32→32 | 旧 9 行 CFEM20 适配器（已退役） | **不可直接比较**：空间、预算、LUT、阶段数均不同 |
| 本地 M1 smoke 20260907 | fold 0 内层 | seed 42 | 1 个候选、极短预算 | 深度为 2 的非冻结空间，DMDC 不在现行单块槽 | 解析代理，非 88 行正式表 | **不可直接比较** |
| G5v2 / sonar_v2 加噪 | NKSID 8 类，15 次（5 折×3 种子） | 协议内 | 历史四阶段 denoise/edge，**不是** DMDC/CFEM | 旧 `denoise`/`edge` | 与现行算子不等价 | 只作噪声机制旁证，**不算 DMDC/CFEM 对照** |
| 检测对照 20260912 | Roboflow Cylinder2 v6，train 330 / valid 99，禁止测试集 | 旧 ckpt 未固定模型初始化 | 检测 80 epoch 意图；旧队列已停 | 检测 640，目标槽 80×80 | 检测表，非分类表 | **另一任务**；只提供错误结构，不并入分类表 |

现行合法槽位选择（分类 YAML）：`Skip`、`MBConv-k3/e3`、`MBConv-k5/e3`、`DMDC`、`CFEM`。Skip 仅允许同通道 stride=1。

## 2. 正式评估（retrain 150，5 折 × 3 种子）

**共同事实：所有已冻结、已重训的 Pareto 代表在 `stage[3]` 均为 Skip，没有一张正式表比较过 MBConv / DMDC / CFEM 替换。** 下列数字描述的是「Skip 槽 + 其余阶段 MBConv」骨干，不是声呐扩展算子收益。

### 2.1 v3gap1280（可声明软件身份；`claimable=true`）

来源：`artifacts/zcu104_full_campaign_v1/retrain_protocol_v1_summaries/mobilenetA_v3gap1280_rl200_20260910_215706/`。`eval_units=15`。板级分类、实测延迟/功率：**未测**。

| 角色 | arch_id | stage[3] | macro_f1 | top1 | fishing_net F1 | small_propeller F1 | 搜索代理 LUT / DSP / BRAM tile | 延迟代理 ms |
|---|---|---|---:|---:|---:|---:|---|---:|
| accuracy_first / knee / balanced | rl_arch_50 | Skip | 0.8534 ± 0.0219 | 0.9324 ± 0.0258 | 0.3084 | 0.8207 | 101552 / 281 / 520 | 100.615 |
| pareto_top4 | rl_arch_142 | Skip | 0.8524 ± 0.0351 | 0.9330 ± 0.0318 | 0.3090 | 0.8212 | 101567 / 282 / 514 | 97.805 |
| pareto_top3 | rl_arch_97 | Skip | 0.8516 ± 0.0153 | 0.9294 ± 0.0196 | 0.2787 | 0.8359 | 101610 / 282 / 485 | 117.353 |
| latency_energy_first | rl_arch_162 | Skip | 0.8473 ± 0.0248 | 0.9257 ± 0.0318 | 0.2876 | 0.8033 | 100804 / 282 / 476 | 96.532 |
| pareto_top5（与 arch_50 同 SHA） | rl_arch_153 | Skip | 同 arch_50 | 同左 | 同左 | 同左 | 同 arch_50 | 同左 |

`rl_arch_50` 与 `rl_arch_153` 的 `architecture_sha256` 相同，不得当作两个独立结构。

逐类 FP32 F1（arch_50 代表，15 单元均值）：

| 类别 | F1 | 相对行为 |
|---|---:|---|
| floats | 0.9751 | 稳定高 |
| cylinder | 0.9704 | 稳定高 |
| iron_pipeline | 0.9542 | 稳定高 |
| tire | 0.9463 | 稳定高 |
| big_propeller | 0.9265 | 稳定高 |
| soft_pipeline | 0.9255 | 稳定高 |
| small_propeller | 0.8207 | 中等；尾类之一 |
| fishing_net | 0.3084 | **系统偏低**；样本仅 20 / 2617 |

其余 v3 代表的逐类形态相同：多数类 ≥0.92，`fishing_net` 停在 0.28–0.31。没有「DMDC 受益、CFEM 退化」的正式逐类差。

### 2.2 20260908 重训（软件数字，freeze 未过）

来源：`artifacts/zcu104_full_campaign_v1/software_metrics_tables/`。全部 `stage[3]=Skip`。`int8_parity_pass=false`，INT8 不得写成板端正确性。

| 角色 | arch_id | stage[3] | macro_f1 | top1 | fishing_net | small_propeller |
|---|---|---|---:|---:|---:|---:|
| accuracy_first | rl_arch_13 | Skip | 0.8929 ± 0.0308 | 0.9625 ± 0.0173 | 0.4873 | 0.8500 |
| pareto_top4 | rl_arch_185 | Skip | 0.8551 ± 0.0277 | 0.9340 ± 0.0281 | 0.2911 | 0.8334 |
| latency_energy_first | rl_arch_195 | Skip | 0.8515 ± 0.0409 | 0.9294 ± 0.0364 | 0.3488 | 0.7895 |
| pareto_top2 | rl_arch_137 | Skip | 0.8509 ± 0.0310 | 0.9325 ± 0.0259 | 0.2926 | 0.8218 |
| knee_point | rl_arch_147 | Skip | 0.8364 ± 0.0554 | 0.9271 ± 0.0308 | 0.2817 | 0.7599 |

与 §2.1 **不可直接比较**：LUT 版本、source freeze、搜索约束（当时 BRAM 预算 700 tile，现主配置硬限制 312 tile）均不同。`fishing_net` 仍是唯一系统性低分母类。

## 3. 短训练代理（搜索 eval 10 epoch）

### 3.1 八阶段 v3gap1280 RL200（2026-09-10）

`total_evaluated=200`，`feasible=200`。**200/200 目标槽均为 Skip**。代理 macro_f1 均值 0.6993，最大 0.8656，最小 0.4424；延迟中位约 100.6 ms（HLS 周期求和 / 200 MHz）。该运行没有留下「同一骨干、只换槽位算子」的配对。

### 3.2 历史四阶段 CFEM20（不可与 §2 并列）

合并折/种子仅作探索计数，不是对照：

| 槽位算子 | v2 记录数 / 可行 | 代理 macro_f1 | 延迟中位 ms |
|---|---|---|---:|
| Skip | 234 / 132 | 可行子集均值 0.5041，最大 0.7218 | 20.7 |
| MBConv-k3e3 | 60 / 0 | 无记录（不可行未训满） | 45.7 |
| MBConv-k5e3 | 92 / 0 | 无记录 | 47.3 |
| DMDC | 92 / 0 | 无记录 | 449.5 |
| CFEM | 134 / 0 | 无记录 | 417.2 |

v1 同类：Skip 可行且有 F1；MBConv/DMDC/CFEM 几乎全部不可行。原因是 25 ms 上限与 CFEM/DMDC 算子 HLS 延迟（约 11 ms / 8 ms **仅槽位**，四阶段旧表上整网约 400 ms 级）冲突。**这不能解释成「DMDC/CFEM 分类更差」，只能解释成「旧约束下未被训练到」。**

### 3.3 本地 M1 smoke（不可比较）

1 候选含 DMDC，代理 macro_f1=0.0604，top1=0.3185；工作树 dirty、空间深度与现行 YAML 不一致。只证明路径可跑，不构成算子排序。

## 4. 已有硬件证据（算子级，非整网）

表：`mobilenetA_full_dmdc_cfem_v3_gap1280`，88/88 行 HLS+route 记录；`timing_met` 80/88。板级分类 / INA226 功率：**本表未测**。目标槽 **32×28×28 → 32×28×28，stride=1，200 MHz**：

| 槽位算子 | HLS 周期 | latency_ms | 布线后 LUT | FF | DSP | BRAM tile | Vivado 无向量功率 W | 相对 Skip Δlatency / ΔLUT / ΔDSP / ΔBRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Skip | 787 | 0.002 | 330 | 1345 | 0 | 0.0 | 0.612 | 基准 |
| MBConv-k3e3 | 824620 | 3.359 | 4891 | 7175 | 12 | 59.5 | 0.805 | +3.357 ms / +4561 / +12 / +59.5 |
| MBConv-k5e3 | 1079250 | 4.494 | 5807 | 7697 | 12 | 97.5 | 0.837 | +4.492 ms / +5477 / +12 / +97.5 |
| DMDC | 1666790 | 7.969 | 3494 | 5283 | 37 | 16.0 | 0.710 | +7.967 ms / +3164 / +37 / +16.0 |
| CFEM | 2460217 | 11.324 | 14219 | 11311 | 452 | 35.0 | 1.150 | +11.322 ms / +13889 / +452 / +35.0 |

现行 200 轮配置硬限制：延迟 ≤120 ms，DSP ≤1209，BRAM ≤312 tile（=624 BRAM18），LUTRAM 压力代理 ≤1（聚合 LUT / 101760）。v3 冻结 Skip 骨干已用约 100.8k–101.6k LUT、476–520 BRAM tile。在 **当前** BRAM 硬限制下，这些 Skip 代表本身也会被拒绝；在 LUTRAM 代理下，把 Skip 换成 CFEM（+1.4 万 LUT）会越过 101760。DMDC 的 LUT 增量较小，但 BRAM 与整网聚合仍受 312 tile 硬限制约束。

结论：现行搜索几乎只产出 Skip 槽，主要是 **代价/准入**，不是「已证明 Skip 分类优于 DMDC/CFEM」。

## 5. 类别稳定受益 / 退化

| 判断 | 依据 | 边界 |
|---|---|---|
| 多数类在 Skip 骨干上稳定高分 | cylinder / floats / tire / 管道 / 大螺旋桨 F1 ≥0.92 | 只描述 Skip 骨干，无算子替换对照 |
| `fishing_net` 系统失败 | 正式与 20260908 重训均约 0.28–0.49；n=20 | 不能归因于 DMDC 或 CFEM |
| `small_propeller` 中等且随候选波动 | 0.76–0.85 | 同上 |
| 历史 edge 算子伤害分类 | G5v2：edge_v2 干净图 macro_f1 0.677 vs 普通 0.717 | 旧 edge ≠ CFEM |
| 历史自适应降噪仅描述性抗噪 | 强斑点配对提升 0.021，Holm 后 p=0.375 | 旧 denoise ≠ DMDC；不得写成显著 |

**没有**「某类因 DMDC 稳定受益、因 CFEM 稳定退化」的配对证据。

## 6. 与检测诊断的接口（不混指标）

学生二检测诊断（Roboflow Cylinder2，旧检查点，验证集选阈值）：工作点漏检为主（cylider 召回 0.214，manta 0.069），低阈值背景误检以千计，定位误检远小于漏检与背景误检。分类混淆不是主错误。focal 分母梯度缺陷已见证，旧 6 次队列停用。

因此：分类 `fishing_net` 低分与检测小目标漏检 **可共享「中分辨率局部细节被 Skip 掉」这一机制假设**，但分类准确率/F1 **不能**作为检测框定位（AP75、定位误检）改善的证据。
