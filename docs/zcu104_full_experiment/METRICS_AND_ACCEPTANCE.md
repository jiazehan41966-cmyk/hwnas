# ZCU104 实验指标与验收规范

> 历史版本：原四阶段ZCU104集成系列的计划/协议。下文的“当前”“本轮”“必须”仅指原文所绑定的系列，不是现工作区的执行指令。当前MobileNetA分类与检测边界见[方法说明](../method_design.md)和[配置版本索引](../SEARCH_CONFIG_CANONICAL.md)；历史产物与验收记录保持原样。

## 1. 公共身份字段

所有软件、HLS、Route、板级、延迟和功率记录必须包含或引用以下公共字段。

| 字段 | 含义 | 要求 |
|---|---|---|
| `campaign_id` | 实验批次 | 必须 |
| `git_commit` | 代码提交 | 必须 |
| `source_freeze_sha256` | 源冻结摘要 | 必须 |
| `protocol_id` / SHA256 | 协议身份 | 必须 |
| `dataset_manifest_sha256` | 数据身份 | 必须 |
| `split_manifest_sha256` | 数据划分身份 | 必须 |
| `arch_id` | 候选标识 | 必须 |
| `architecture_sha256` | 候选内容摘要 | 必须 |
| `checkpoint_sha256` | 权重摘要 | 必须 |
| `quantization_manifest_sha256` | INT8规范摘要 | INT8后必须 |
| `hls_source_sha256` | HLS输入源码摘要 | HLS后必须 |
| `bitstream_sha256` | bitstream摘要 | 板级后必须 |
| `xsa_sha256` / `hwh_sha256` | 平台摘要 | PS–PL后必须 |
| `runtime_sha256` | 板端runtime摘要 | 板级后必须 |
| `board_id` / serial | 板卡身份 | 板级后必须 |
| `tool_versions` | 工具版本 | 必须 |
| `clock_mhz` | 实际工作频率 | 硬件后必须 |
| `route_seed` | 实现seed | Route后必须 |
| `instrument_serial` | 功率仪身份 | 功率后必须 |
| `rail_scope` | 功率测量范围 | 功率后必须 |

任何一项关键身份不一致时，不允许把不同记录拼接成同一候选工作点。

## 2. 软件分类指标

| 指标 | 定义/来源 | 聚合 | 级别 |
|---|---|---|---|
| Macro-F1 | 各类别F1的算术平均 | 5 folds × 3 seeds | 主要 |
| Top-1 | 正确预测数/样本数 | 5×3 | 主要次级 |
| weighted-F1 | 按类别支持数加权F1 | 5×3 | 次级 |
| balanced accuracy | 各类recall平均 | 5×3 | 次级 |
| per-class precision | 每类TP/(TP+FP) | 每fold/seed及汇总 | 必测 |
| per-class recall | 每类TP/(TP+FN) | 每fold/seed及汇总 | 必测 |
| per-class F1 | 每类precision/recall调和平均 | 每fold/seed及汇总 | 必测 |
| confusion matrix | 真实类×预测类 | 每fold/seed及总和 | 必测 |
| worst-class F1 | 最小类别F1 | 5×3 | 建议 |
| NLL | 负对数似然 | 5×3 | 可选 |
| ECE | 期望校准误差 | 5×3 | 可选 |
| params | 参数数量 | 每架构 | 必测 |
| MACs | 一次推理乘加量 | 每架构 | 必测但非硬件结果 |
| model bytes | 权重文件大小 | 每checkpoint | 必测 |
| train GPU-hours | 正式训练成本 | 每方法/单元 | NAS比较必须 |

### 报告要求

- 每个指标保留 15 个原始 fold/seed 值；
- 报告 mean、std、95% CI；
- 比较时报告 paired delta 及其 CI；
- Macro-F1 是类别不均衡场景的主要指标，Top-1 不替代 Macro-F1。

## 3. 声呐结构与鲁棒性指标

| 指标 | 定义/说明 | 级别与边界 |
|---|---|---|
| `clean_macro_f1` | 原始冻结 outer 图像的 Macro-F1 | 算法主要终点 A1 |
| `sonar_robust_macro_f1` | 先对每个退化族内严重度等权平均，再对退化族等权平均 | 若声称强噪声/低对比度鲁棒性，则为共同主要终点 A2 |
| per-corruption Macro-F1 | 每个退化族×严重度的分类结果 | 必测，不得只报聚合值 |
| `absolute_drop` | clean Macro-F1 - corruption Macro-F1 | 必测 |
| `retention` | corruption Macro-F1 / clean Macro-F1 | 必测 |
| `worst_condition_macro_f1` | 每个 fold/seed 中所有冻结退化单元的最小 Macro-F1 | 主要护栏 |
| robust per-class F1 | 退化条件下逐类别 F1 | 必测 |
| minority-class recall | 预注册少数类别在退化条件下的 recall | 必测 |
| PSNR | 像素差异 | 无 clean target 时仅结构保持 |
| SSIM | 结构相似性 | 无 clean target 时仅结构保持 |
| MSE | 均方误差 | 与 reference policy 绑定 |
| `reference_policy` | `paired_clean_target` 或 `input_as_reference` | 必须 |
| CNR/ENL/measured SNR | 需要 ROI、均匀区或采集标定 | 条件不足时 `NOT_AVAILABLE`，不得估造 |

正式退化至少包含 multiplicative speckle、additive noise、contrast attenuation，每类 mild/moderate/severe。参数只能用 training split 先导确定并在 outer 评估前冻结；同一退化样本必须跨模型复用。完整公式、证据边界和记录要求见 [SONAR_AND_THERMAL_METRICS.md](SONAR_AND_THERMAL_METRICS.md)。

## 4. INT8与数值一致性指标

| 指标 | 定义 | 验收 |
|---|---|---|
| FP32 Macro-F1/Top-1 | 浮点基准 | 记录 |
| INT8 Macro-F1/Top-1 | 量化结果 | 达到冻结精度门 |
| `delta_macro_f1` | INT8-FP32 | 建议 >= -0.02，最终以冻结协议为准 |
| scale/zero point | 每层或每通道量化参数 | 完整记录 |
| clipping/saturation rate | 落在量化边界的比例 | 记录并审查异常层 |
| accumulator width | 累加位宽 | 不允许溢出 |
| layer max abs error | 层间最大误差 | 按契约 |
| layer mismatch count | 层间整数失配数 | bit-exact门要求0 |
| logits mismatch count | 最终INT8输出失配数 | 0 |
| argmax mismatch count | 最终类别失配数 | 0 |
| calibration sample count | 校准样本量 | 记录 |
| calibration class coverage | 校准类别覆盖 | 100%类别 |

## 5. HLS指标

| 指标 | 来源 | 验收/用途 |
|---|---|---|
| C-sim status | Vitis HLS | PASS |
| C-sim mismatch count | testbench | 0 |
| RTL co-sim status | Vitis HLS/XSim | PASS或预注册替代门 |
| RTL mismatch count | testbench | 0 |
| latency min/max/avg cycles | csynth | 估计，不等同板级 |
| II | csynth | 记录 |
| estimated period | csynth | 记录 |
| estimated Fmax | csynth | 记录 |
| LUT/FF/DSP | csynth | 资源预筛 |
| BRAM18/BRAM36/URAM | csynth | 资源预筛 |
| external scratch bytes | 生成器/报告 | 必测 |
| read/write bytes | HLS/runtime model | 必测 |
| burst length | AXI接口 | 必测 |
| synthesis return code | 进程 | 必须为0 |
| warnings/errors | 日志 | 关键错误为0 |
| synthesis wall time | 日志 | 记录 |

HLS资源和周期仅是 HLS 层证据，不得写成 post-route 或板级实测。

## 6. Vivado综合与Route指标

| 指标 | 说明 | 正式门 |
|---|---|---|
| LUT / utilization % | 逻辑资源 | 低于冻结阈值 |
| LUTRAM | 分布式存储 | 记录 |
| FF / utilization % | 寄存器 | 低于冻结阈值 |
| DSP / utilization % | DSP资源 | 低于冻结阈值 |
| BRAM18/BRAM36 | 块RAM | 分开记录 |
| URAM | UltraRAM | 分开记录 |
| WNS | 最差setup裕量 | >= 0 |
| TNS | setup总负裕量 | = 0 |
| WHS/THS | hold裕量 | 无违规 |
| unconstrained paths | 未约束路径 | 关键路径为0 |
| route errors | 路由错误 | 0 |
| critical DRC | 关键设计规则错误 | 0 |
| achieved clock | 实际可用频率 | 记录 |
| congestion | 拥塞 | 记录 |
| route seed | 实现随机性 | 必须记录 |
| synth/route wall time | 编译成本 | 记录 |
| bitstream bytes/SHA256 | 产物身份 | 必须 |
| Vivado estimated power | 工具估计 | 只作辅助，不是板级功率 |

建议主要资源目标低于85%，90%作为硬上限；最终阈值必须在正式Route前冻结。

## 7. 板级正确性指标

| 指标 | 定义 | 验收 |
|---|---|---|
| expected samples | manifest样本数 | 记录 |
| observed samples | 板级返回样本数 | 等于expected |
| coverage | observed/expected | 100% |
| missing samples | 未返回样本 | 0 |
| duplicate samples | 重复样本 | 0 |
| CRC errors | 传输错误 | 0 |
| status errors | accelerator/runtime错误 | 0 |
| retries | 重试次数 | 记录，正式建议0 |
| max abs logits error | 板级vs参考 | bit-exact时0 |
| logits mismatch count | 整数失配元素数 | 0 |
| argmax mismatch count | 类别失配 | 0 |
| board Macro-F1 | 板级预测计算 | 对相应checkpoint有效 |
| board Top-1 | 板级预测计算 | 对相应checkpoint有效 |
| per-class F1/confusion | 板级分类结果 | 必测 |
| deployment binding | 完整hash chain | PASS |

一个 anchor checkpoint 的板级结果不能冒充 5×3 板级均值。

## 8. 延迟和吞吐指标

| 指标 | 起止边界 | 单位 |
|---|---|---|
| preprocess latency | 原始样本到INT8输入 | ms |
| H2D DMA latency | DDR输入准备/提交到DMA完成 | ms |
| kernel cycles | accelerator start到done | cycles |
| kernel latency | cycles / actual clock | ms |
| D2H DMA latency | 输出DMA提交到完成 | ms |
| postprocess latency | logits到最终类别 | ms |
| runtime overhead | 控制、同步等非DMA/kernel | ms |
| system E2E latency | 冻结输入边界到预测返回 | ms |
| cold-start latency | bitstream/runtime/weight首次准备 | ms或s |
| p50/p90/p95/p99 | 每block延迟分布 | ms |
| mean/std/CV | 每block及跨block | ms/% |
| throughput | 持续推理数/时间 | inference/s |
| timeout/retry/error rate | 运行稳定性 | % |

### 测量方案

- 100 次预热；
- 5 个独立 block；
- 每 block 1000 次 batch-1；
- block 是统计重复，单个 block 内推理不是完全独立实验单元。

## 9. 功率和能耗指标

### 原始字段

| 字段 | 要求 |
|---|---|
| `timestamp_s` | 单调且可与runtime marker对齐 |
| `power_w` | 或由`voltage_v × current_a`得到 |
| `voltage_v` / `current_a` | 若仪器提供则保留 |
| sample rate | 必须记录 |
| calibration | 型号、序列号、日期/证书 |
| `rail_scope` | 主实验必须是`board_input_total` |
| temperature/fan | 与功率block同步 |
| inference_count | 与receipt一致 |

### 计算

`P_idle = mean(P_idle(t))`

`P_active = mean(P_active(t))`

`P_dynamic = P_active - P_idle`

`E_total_mJ = 1000 * integral(P_active(t) dt) / N`

`E_dynamic_mJ = 1000 * integral(P_active(t) - P_idle dt) / N`

`FPS_per_W = throughput / P_active`

可选：

- `EDP = E_total_mJ × system_e2e_latency_ms`
- `ED2P = E_total_mJ × system_e2e_latency_ms^2`

### 正式门

- idle和active各至少3组，建议5组；
- 每组 >= 60 s；
- active每组 >= 1000次推理；
- 至少3个候选；
- 同一仪器、接线、量程、采样协议；
- 编程和上传不在active窗口；
- receipt inference count、时间和bitstream hash一致；
- raw CSV hash完整；
- dynamic power非负，否则触发接线/同步审计。

## 10. 温度和稳定性指标

| 指标 | 说明 | 来源/要求 |
|---|---|---|
| PS LPD/FPD temperature | 处理系统芯片结温 | ZynqMP AMS/System Monitor；记录实际 IIO 节点 |
| PL temperature | 可编程逻辑芯片结温 | ZynqMP AMS/System Monitor；记录实际 IIO 节点 |
| ambient temperature | 板周围环境温度 | 只能来自外部探头；无探头则 `NOT_MEASURED` |
| room setpoint | 空调/环境设定值 | 仅上下文，不替代 ambient 实测 |
| fan mode/RPM | 散热状态 | RPM 仅在有可读 tach 时记录 |
| `temp_start/mean/max/end_c` | 测量窗口温度摘要 | 每 block 必测 |
| `temp_slope_c_per_min` | 窗口温度线性斜率 | 稳态和漂移判定 |
| `time_to_thermal_steady_s` | 达到冻结稳态规则的时间 | 每候选必测 |
| power drift | 前后窗口功率差 | 与温度时间戳对齐 |
| latency drift | 前后窗口延迟差 | 与温度时间戳对齐 |
| throughput drift | 前后窗口吞吐差 | 与温度时间戳对齐 |
| inference count | 连续运行计数 | 必测 |
| reset/hang/timeout/CRC | 稳定性事件 | 必须为 0 才通过 |
| overtemperature/throttle | 过温、降频或保护事件 | 必须为 0 才通过 |

建议 1 Hz 采集 PS/PL 结温，C2/C3 各运行 30–60 min。稳态阈值需在正式测量前冻结；可从“连续 5 min 的温度斜率绝对值不超过 0.2 °C/min”开始先导。详细能力核验和 Linux/JTAG 读取方法见 [SONAR_AND_THERMAL_METRICS.md](SONAR_AND_THERMAL_METRICS.md)。

## 11. 代理校准和搜索指标

| 指标 | 说明 | 建议门 |
|---|---|---|
| MAPE | 平均绝对百分比误差 | 报告 |
| median APE | 中位误差 | 报告 |
| P90/P95 APE | 尾部误差 | P90 <=25% |
| Spearman | 排序相关 | >=0.8 |
| Kendall tau | 排序稳健性 | 报告 |
| false-feasible | 把不可行误判为可行 | 报告并设安全裕量 |
| false-infeasible | 把可行误判为不可行 | hard reject时必须0 |
| interval coverage | 预测区间覆盖真实值 | 报告 |
| Top-K recall | 真实优候选召回 | 报告 |
| shortlist coverage | 候选HLS覆盖 | 100% |
| candidate evaluations | 搜索预算 | 方法比较必须相同 |
| GPU-hours | 实际计算预算 | 独立报告 |
| Pareto size/hypervolume | 搜索质量 | hypervolume可选 |

## 12. 统计报告规范

1. clean 与声呐鲁棒精度：fold/seed为单元，使用相同退化样本做配对分析；
2. 延迟：block为单元，报告block-level CI；
3. 功率：idle/active配对block为单元；
4. 仪器采样点不作为独立重复，防止伪重复；
5. 多候选/多算子比较使用Holm或FDR；
6. 同时报告effect size、CI和原始值；
7. 不以“无显著差异”解释为“等价”，除非预注册等价界值并执行等价检验；
8. 失败、缺失和排除必须按预注册规则报告；
9. Pareto以原始物理目标为主，不用单一加权分数替代；
10. cross-board比较必须统一候选、checkpoint、clock、toolchain、runtime和功率范围，否则只做描述。
