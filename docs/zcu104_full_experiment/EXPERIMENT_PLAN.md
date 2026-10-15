# ZCU104 完整实验计划

> 历史版本：原四阶段ZCU104集成系列的计划/协议。下文的“当前”“本轮”“必须”仅指原文所绑定的系列，不是现工作区的执行指令。当前MobileNetA分类与检测边界见[方法说明](../method_design.md)和[配置版本索引](../SEARCH_CONFIG_CANONICAL.md)；历史产物与验收记录保持原样。

## 1. 实验目标

本实验计划分为两轮：

- 第一轮“迁移与标定”：使用四个已经冻结的软件候选打通 ZCU104 的完整正确性、性能和功率链路；
- 第二轮“搜索与复测”：利用第一轮数据校准 ZCU104 代价模型，再运行正式硬件感知搜索并复测最终 Pareto 候选。

第一轮不是新的 NAS 优越性实验，其作用是建立可信的 ZCU104 物理证据和校准数据。只有第二轮完成后，才能形成“面向 ZCU104 约束的自动设计方法”结论。

## 2. 主要研究问题和假设

### RQ1：声呐算子是否提高分类性能？

- 算法层共同主要响应变量：原始数据 `clean_macro_f1` 和冻结声呐退化协议下的 `sonar_robust_macro_f1`；
- 鲁棒性护栏：`worst_condition_macro_f1`、各退化族/严重度的 Macro-F1、相对保持率和 minority-class recall；
- 实验单元：fold/seed；
- 证据：Protocol V2 软件 5 folds × 3 seeds；
- 声呐退化至少覆盖乘性散斑、加性噪声和低对比度三类，参数只用 training split 先导确定并在读取 outer 结果前冻结；
- 因果解释必须来自匹配 factorial 对照，不从任意两个部署候选直接归因。

### RQ2：候选能否在 ZCU104 上部署？

- 响应变量：时序、资源、DRC、路由状态；
- 实验单元：candidate × implementation version × clock × route seed；
- 失败的资源或时序结果也是正式结果。

### RQ3：板上 INT8 是否与软件参考一致？

- 响应变量：sample coverage、logits mismatch、argmax mismatch、CRC/error；
- 验收目标：完整冻结样本集零缺失、零不明数值失配。

### RQ4：不同候选的延迟和能耗是否构成可部署 Pareto？

- 主要响应变量：system E2E latency、total energy/inference；
- 次要响应变量：kernel latency、dynamic energy、FPS/W、资源占用；
- 功率统计单元是测量 block，不是仪器的每个采样点。

### RQ5：ZCU104 代价模型是否足以支持搜索筛选？

- 响应变量：P90/P95 APE、Spearman、false-feasible、false-infeasible、Top-K recall；
- 未通过独立 holdout 校准时，代理只能排序或将候选送入 HLS，不能硬拒绝。

## 3. 候选和对照矩阵

### 3.1 第一轮部署候选

| 编号 | 角色 | 架构 | 软件实验规模 | 硬件实验范围 |
|---|---|---|---|---|
| C0 | 原始基线 | S2 K3-e3 + S4 K3-e3 | 5×3 | HLS、Route、板级、延迟、功率 |
| C1 | Stage2-K5 代表 | S2 K5-e6 + S4 K3-e3 | 5×3 | HLS、Route、板级、延迟、功率 |
| C2 | 精度优先 | S2 K5-e3 + S4 K5-e3 | 5×3 | HLS、Route、板级、延迟、功率、长期稳定性 |
| C3 | 低成本 | S2 K5-e3 + S4 Skip | 5×3 | HLS、Route、板级、延迟、功率、长期稳定性 |

### 3.2 匹配硬件消融

至少建立以下三组匹配对照：

1. S2 K3-e3 vs S2 K5-e3，S4 固定 K3-e3；
2. S4 K3-e3 vs S4 K5-e3，S2 固定 K5-e3；
3. S4 K3-e3 vs S4 Skip，S2 固定 K5-e3。

匹配消融的主要目的，是解释单个算子对 cycles、resource、memory traffic 和 energy 的影响。四个部署候选用于系统 Pareto，而非全部用于单因素因果解释。

### 3.3 外部算法基线

- MobileNetV2 scratch：建议作为主要标准网络基线；
- MobileNetV2 grayscale pretrained：可作为迁移学习基线；
- 历史 RL 候选：只作为历史参考，除非在 Protocol V2 下重新训练和绑定。

正式比较要求相同数据划分、预处理、epoch、checkpoint 选择、量化、runtime 和硬件测量协议。

## 4. 实验单元和冻结规则

### 4.1 软件实验单元

`architecture × outer_fold × seed × training_protocol`

正式 NKSID 结果使用 folds 0–4 和 seeds 42/43/44。outer 数据不得参与结构、超参数、epoch 或 checkpoint 选择。

### 4.2 硬件实现单元

`architecture × checkpoint × implementation_version × toolchain × part × clock × route_seed`

不同 route seed 是实现随机性重复，不得只挑 WNS 或功率最好的 seed 作为主结果。

### 4.3 板级测量单元

`candidate × checkpoint × bitstream × runtime × board × clock × measurement_block`

候选精度、延迟、资源和功率只有在上述身份完全一致时才能合并为联合工作点。

## 5. E0：环境、源码和协议冻结

### 输入

- 新的干净 ZCU104 worktree；
- Protocol V2；
- 四个部署候选和 checkpoint 索引；
- ZCU104 与服务器。

### 操作

1. 生成源冻结 manifest；
2. 记录 Git commit、dirty status；
3. 记录 Python/PyTorch/CUDA；
4. 记录 Vivado/Vitis HLS；
5. 记录服务器 OS、CPU、内存；
6. 通过 JTAG 查询 ZCU104 exact device；
7. 确认启动模式、串口映射、板端 OS 和网络；
8. 记录板卡序列号、供电方式和功率仪接线。

### 产物

- `00_freeze/source_freeze_manifest.json`
- `00_freeze/protocol_freeze.json`
- `00_freeze/environment_card.json`
- `00_freeze/board_capability_card.json`
- `00_freeze/commands.txt`

### 通过门

- source freeze PASS；
- exact part 确认；
- 认证信息不进入任何文件；
- 所有正式依赖版本固定。

## 6. E1：软件分类和对照

### 核心实验

- 四候选：已有 4 × 15 个 Protocol V2 单元，先重新审计完整性和 hash；
- MobileNetV2 若用于正式比较：在 Protocol V2 下新增 15 个单元；
- 不重新使用 outer 结果挑选新的候选。

### 必测

- Macro-F1、Top-1、weighted-F1；
- `sonar_robust_macro_f1`、`worst_condition_macro_f1`、各声呐退化单元的 Macro-F1/drop/retention；
- balanced accuracy；
- clean 和各退化条件下的 per-class precision/recall/F1；
- confusion matrix；
- parameter count、MACs、model bytes；
- train time、GPU-hours；
- fold/seed 级原始结果。

### 统计

- 相同 fold/seed 配对；
- 均值、标准差、95% CI；
- paired bootstrap 或 exact paired sign-flip；
- 多个预注册比较使用 Holm 校正；
- 报告 signed delta，不只报告 p-value。

### 通过门

- 15/15 单元齐全；
- split/source/checkpoint hash 可验证；
- 无 outer selection；
- 预声明的主要和次要比较全部报告。

### E1-S：声呐专项鲁棒性与结构分析

- 按 [SONAR_AND_THERMAL_METRICS.md](SONAR_AND_THERMAL_METRICS.md) 冻结 multiplicative speckle、additive noise、contrast attenuation 三个核心退化族及 mild/moderate/severe 参数；
- 同一 sample/corruption/severity 对所有模型复用相同退化图和随机种子；
- 每个 fold/seed 独立计算 `sonar_robust_macro_f1`、最差条件、drop、retention、per-class F1 和 minority-class recall；
- FP32 与 INT8 都执行同一协议，检查量化是否在困难声呐条件下产生额外下降；
- PSNR/SSIM/MSE 单列为算子结构保持证据；NKSID 无成对 clean target 时必须标记 `reference_policy=input_as_reference`，不得称为真实去噪质量；
- CNR/ENL/SNR 只有在存在目标/背景 ROI、均匀区或采集标定时才计算，否则标记 `NOT_AVAILABLE`。

通过门：退化实现、参数、seed、样本和 SHA256 已冻结；15/15 软件单元齐全；不得遗漏最差条件或只选择有利严重度；共同主要终点及多重比较规则已预注册。

## 7. E2：INT8 量化与软件参考

### 实验范围

- 四个候选的全部 15 个 checkpoint 完成软件 INT8 评估；
- 四个 anchor checkpoint 生成完整硬件参考张量；
- C2 建议保留全部 15 个可加载权重包，为增强板级验证做准备。

### 校准

- 校准数据只来自 training；
- 记录样本 ID、类别覆盖和 SHA256；
- 保存每层 weight/activation scale、zero point、clipping rule；
- 明确 rounding、saturation 和 accumulator width。

### 建议预注册阈值

- PTQ Macro-F1 绝对下降不超过 0.02；
- 超过阈值后进入 QAT；
- QAT 阈值必须在正式运行前冻结；
- 未通过量化精度门的候选不进入正式板级功率比较。

### 数值样本

- 每类真实样本；
- 全零、全最大、全最小；
- 边缘/棋盘/脉冲图；
- 固定种子的随机输入；
- 接近量化饱和值的边界张量。

## 8. E3：HLS C-sim、RTL co-sim 和资源筛选

### HLS 主实验

- C0–C3 在 external-scratch V0 上全部运行；
- 每个候选保存生成源码、testbench、Tcl、完整日志、报告和 hash；
- 之后对 C2/C3 执行 V1/V2/V3 实现优化。

### RTL 验证

- 所有算子正式 shape 完成 RTL co-sim；
- C0、C2、C3 至少一个真实样本完成完整网络 RTL co-sim；
- 如果完整网络超时，执行预注册的替代验证协议，不得静默跳过。

### 通过门

- C-sim/RTL mismatch 为 0；
- HLS return code 为 0；
- 无资源超限；
- 报告不把 HLS estimate 写成 post-route 或板级结果。

## 9. E4：ZCU104 synthesis、implementation 和 Full Route

### 工作点

- 主工作点：200 MHz，C0–C3；
- 频率敏感性：100/150/200 MHz，C2 和 C3；
- 每个 `candidate × clock` 生成独立 bitstream 和 hash。

### Route seed

- 所有候选使用预注册主 seed；
- C2/C3 额外运行至少两个 route seed，形成三个 seed 的实现稳定性样本；
- 主报告使用预注册 seed，不按结果选 seed。

### 必测

- LUT、LUTRAM、FF、DSP；
- BRAM18、BRAM36、URAM；
- WNS、TNS、WHS、THS；
- unconstrained paths、DRC、route errors；
- achieved clock、congestion；
- synthesis/route wall time；
- Vivado estimated power，明确标为估计。

### 通过门

- WNS >= 0；
- TNS = 0；
- hold timing 无违规；
- route errors = 0；
- critical DRC = 0；
- 无未约束关键时钟；
- 资源低于冻结阈值。

200 MHz 若失败，只能报告 FAIL 并执行预注册回退；150 MHz 结果不得覆盖 200 MHz 失败。

## 10. E5：板级数值一致性和全样本分类

### 三级验证

1. 固定向量 smoke：状态码、CRC、cycles、logits；
2. 真实/随机/边界向量：软件—板卡数值一致性；
3. 完整 outer split：逐 sample ID 板级分类。

### 最低正式范围

- C0–C3 各一个预注册 anchor checkpoint；
- 每个 anchor 覆盖对应完整 outer split；
- anchor 规则在读取 outer 指标前确定，例如固定 fold0/seed42。

### 增强范围

- C2 的 15 个 checkpoint 全部板级运行；
- 若要报告板级 5×3 平均准确率，必须完成全部对应 checkpoint；
- 如果只运行一个 checkpoint，只能报告该 checkpoint 的板级准确率和数值等价性。

### 通过门

- sample coverage = 100%；
- missing/duplicate = 0；
- CRC/status error = 0；
- INT8 logits mismatch = 0；
- argmax mismatch = 0；
- deployment binding PASS。

## 11. E6：延迟、吞吐和冷启动

### 稳态延迟

- 预热 100 次；
- 每候选 5 个独立 block；
- 每 block 1000 次 batch-1 推理；
- 固定 CPU governor、affinity、clock、runtime 和温度策略。

### 冷启动

单独测量：

- bitstream/overlay load；
- runtime init；
- weight load；
- 第一次 inference。

冷启动结果不得并入稳态单次推理均值。

### 必测

- preprocess；
- H2D DMA；
- kernel；
- D2H DMA；
- postprocess；
- runtime overhead；
- system E2E；
- p50/p90/p95/p99、mean/std、CV；
- throughput、timeout、retry。

## 12. E7：外部功率和能耗

### 设计

- C0–C3 全部测量；
- 5 轮 randomized block；
- 每轮内候选顺序由固定随机种子生成；
- 每候选每轮包含 idle 和 active；
- idle >= 60 s；
- active >= 60 s 且 inference_count >= 1000；
- 输入和权重在 active 前加载；
- 编程、上传和日志导出不进入 active 窗口。

### 温度控制

- 预热到预注册稳态；建议先导后冻结为连续 5 min 温度斜率绝对值不超过 0.2 °C/min；
- 通过 Zynq UltraScale+ AMS/System Monitor 记录 PS/PL 芯片结温，建议 1 Hz，并与功率、推理 receipt 使用统一时间基准；
- 环境温度必须来自外部探头；只有空调设定值时记录为 `room_setpoint`，不得冒充 `ambient_temperature`；
- 候选切换后按同一规则等待；
- 不按实时功率结果决定是否延长某个候选。

### 主要结果

- idle/active/dynamic power；
- total/dynamic energy per inference；
- FPS/W；
- power/energy 的 block-level 95% CI；
- 频率敏感性仅对 C2/C3 执行。

## 13. E8：长期稳定性与热漂移

### 候选

- C2 精度优先；
- C3 低成本。

### 方法

- 每个候选连续运行 30–60 min；
- 固定输入集合循环；
- 定时记录 PS/PL 结温、外部环境温度（若有）、功率、延迟、吞吐、CRC 和累计 inference count；
- 报告温度起点/均值/最大值/终点、温升斜率、达到稳态时间，以及 latency/power 对温度的漂移。

### 建议门槛

- 无 reset、hang、timeout、CRC error；
- 无温度保护或降频；
- 后半程 latency 与前半程差异不超过 5%；
- active power 漂移处于预注册范围。

## 14. E9：ZCU104 代理校准

### 样本

- 第一轮四个部署候选；
- 匹配消融候选；
- 至少补充到 8 个 semantic-safe 完整网络硬件样本；
- 功率回归建议补充到 12–20 个完整网络点；
- 按 architecture 分组划分训练和 blind holdout，禁止同一架构不同记录跨集合泄漏。

### 指标

- MAPE、median APE、P90/P95 APE；
- Spearman、Kendall；
- false-feasible、false-infeasible；
- prediction interval coverage；
- Top-K recall、shortlist coverage。

### 建议门

- P90 APE <= 25%；
- Spearman >= 0.8；
- hard reject 的 false-infeasible = 0；
- shortlist HLS coverage = 100%。

未通过时，代理只用于排序，置信区间重叠或接近约束边界的候选继续送 HLS/Route。

## 15. E10：正式 ZCU104 HW-NAS 与最终复测

### 方法比较

主比较为 RL、Random、Aging Evolution。三者使用相同：

- 搜索空间和算子准入；
- folds/seeds；
- candidate evaluations；
- 每候选训练 epoch/batch；
- evaluator；
- 硬件代理版本；
- 资源约束；
- Pareto 目标。

Proxyless 若仍使用 weight-sharing supernet，应作为不同 evaluator 范式单列，除非先建立可交换的共同预算。

### 泛化实验

若声明 NAS generalization：

- 5 outer folds × 3 search seeds；
- 每个 outer unit 独立搜索；
- outer 数据不进入搜索；
- 保存每次搜索的完整轨迹、候选数、GPU-hours 和最终架构 hash。

### 最终部署

从新搜索冻结：

- accuracy-first；
- knee-point；
- latency/resource-first。

三者重新完成 INT8、HLS、Route、板级全样本、延迟、功率和稳定性，不能直接复用第一轮旧候选的物理结果。

## 16. 统计分析计划

### 软件精度

- fold/seed 为实验单元；
- 配对 delta、标准差和 cluster-aware 95% CI；
- paired bootstrap 或 exact sign-flip；
- Holm 校正；
- 报告效应量和每 fold 方向。

### Route

- 主 seed 给出正式工作点；
- 多 seed 结果报告范围和变异，不以最佳 seed 代替预注册 seed。

### 延迟

- block 为统计单元；
- 单个 block 内的 1000 次推理用于估计该 block 的分布；
- 跨 block 报告 p50/p95 的均值、标准差和 block bootstrap CI；
- 不把 5000 次相关推理当作 5000 个独立样本。

### 功率

- idle/active 配对 block 为统计单元；
- 候选顺序随机化；
- 报告配对差值和 block bootstrap CI；
- 仪器采样点只用于时间积分，不作为独立重复。

### Pareto

主要目标：

- maximize Macro-F1；
- minimize system E2E latency；
- minimize total energy/inference；
- minimize LUT/DSP/BRAM/URAM utilization。

Pareto 基于物理实测和 route 结果，不用任意加权总分替代主结论。

## 17. 预注册停止规则

以下情况阻止候选进入下一阶段：

- 源、协议、数据、候选或 checkpoint hash 不一致；
- INT8 精度门失败且 QAT 未恢复；
- C-sim/RTL mismatch 非零；
- HLS 资源超限；
- Full Route timing/DRC/route fail；
- 板级样本缺失或数值不一致；
- 功率仪未校准、时间无法对齐或 raw CSV 不完整；
- 温度越界或板卡出现 reset/hang；
- bitstream/runtime/receipt/power 不能建立 hash chain。

技术失败可以修复后重新运行，但旧失败必须保留。不得根据不利结果修改阈值、剔除规则或主要终点。

## 18. 结果目录

```text
results/zcu104_full_campaign_v1/
  00_freeze/
  01_software_accuracy/
  02_int8/
  03_hls/
  04_rtl_cosim/
  05_vivado_route/
  06_board_correctness/
  07_latency/
  08_power/raw/
  08_power/receipts/
  08_power/manifests/
  09_thermal_stability/
  10_calibration/
  11_search/
  12_final_reports/
```

每级至少包含 manifest、commands、environment、raw、summary、hashes 和 gate status。
