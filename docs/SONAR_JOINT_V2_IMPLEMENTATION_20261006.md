# 声呐架构、量化与 FPGA 映射联合选择：核查与 V2 补齐

核查日期：2026-10-06（北京时间）。范围：当前 640 输入声呐检测、V0.1 的 144 架构空间及其联合选择。依据本地源码和已归档产物；本次未连接远端、未启动真实训练/量化实验、HLS、布局布线或板测。带历史日期的运行快照不用于推断当前进程状态。

版本说明：这里的“V2”是新增联合筛选流程，实际仍调用 `src/hwnas_fpga/models/sonar_candidate` 的旧 V0.1 模型。没有接入 `tools/detector_b1_light/model.py` 的 L1/L0 轻量 neck/head，也没有接入 V0.2 或 elastic pilot。下文 2592 个成本点和 0 个通过筛查的结论只适用于旧 V0.1 成本空间，不能外推到新版轻量网络。新模型改变 neck/head 通道和算子，必须先验证其专属图、PTQ 兼容性及精确成本覆盖。

## 核查结论

附图四步并非全部未做。已有联合候选表示、精确 HLS（高层综合）查表、最终量化精度绑定与 Pareto（非支配前沿）筛选，也已有真实权重的完整串行检测图执行器、三网络重复性能测量和 D 的完整 valid99 板端评估。主要缺口是：**旧 40→60→80 晋升仍按 FP32（浮点）mAP 排序，没有在每轮晋升前执行已有硬件预算和量化精度约束。** 问题是约束没有接入该流程，而非项目没有预设预算。

| 附图步骤 | 已有实现或证据 | 尚缺内容 / 本轮补齐位置 |
|---|---|---|
| 01 联合候选 | [candidate_graphs.py](../tools/fpga_h4/candidate_graphs.py) 给出 144 种输出层组合与卷积核架构；[cost_api.py](../tools/fpga_h4/cost_api.py) 接受架构×量化合同×映射。量化有 W8A8、HEAD_W8A16、HEAD_W16A16；映射有六种已测策略。 | 联合候选需要贯穿逐轮选择；不能等到 FP32 shortlist 确定后才引入 q、m。新入口独立放在 `tools/sonar_joint_v2/`，旧冻结源不改。 |
| 02 预算前置 | [baseline_sonar_frozen.yaml](../configs/baseline_sonar_frozen.yaml) 已保存检测筛查延迟等约束；ZCU104 另有正式预注册资源、时序、板测与测量质量门槛。 | 旧 H4 返回 `constraint_status=REPORTED_NOT_BUDGETED`；算子峰值和串行 HLS 延迟未按应用预算拒选。V2 把已有预算接入每轮候选筛查，同时保留整网准入的独立合同。 |
| 03 感知晋级 | [accuracy_api.py](../tools/fpga_h4/accuracy_api.py) 只绑定同架构、同量化合同、同 checkpoint 的实测 valid 精度；[joint_search.py](../tools/fpga_h4/joint_search.py) 对旧 B1/C/D 的冻结 Q(x) 构建最终前沿；[独立收尾器](../tools/sonar_search_finalize/README.md) 修复新架构被旧白名单排除的问题。 | 旧阶段排序只使用 FP32 mAP，后续三种子量化不足以挽回此前弃选。V2 在 40/60/80 每轮纳入 PTQ（训练后量化）valid mAP 和硬件成本，执行前沿选择、临界候选保留、量化弃选抽查及最终三种子门槛。 |
| 04 部署验证 | 三网络真实 INT8 参数的软件整数评估、完整图执行器及路由；D/C/B1 固定 seed42 图的逐节点/输出头核对；三网络各 native/E2E 预热 1 次＋测量 10 次；D train8＋valid99 零失配及 mAP 已归档。 | C/B1 的 valid99 本地未见完成报告；现有串行执行器不实现 H3 六种优化映射。完整严格部署准入未通过，不能将功能可执行或性能测量合同 PASS 写成满足时延预算。 |

当前架构尺度指 P2/P3/P4/P5 输出层组合；输入固定为 640，不把它描述为已执行任意输入分辨率、宽度或深度搜索。混合精度仅改变 head，不能描述为全网络任意逐层位宽搜索。

## 预算的来源、单位和执行层级

已有权威预算见 [ZCU104 实验配置](../configs/experiments/zcu104_full_campaign_v1.yaml) 和 [2026-08-10 硬件门槛预注册](../artifacts/zcu104_full_campaign_v1/control/hardware_gate_preregistration_20260810.json)。复用其硬件工程约束，不引入该分类实验的准确率、数据或候选结论。

| 层级 | 约束 | 应用方法与边界 |
|---|---|---|
| 检测算子筛查 | 串行 HLS 延迟 ≤120000 μs（120 ms） | 来自旧检测配置的 `max_latency_ms=120.0`。用于查表筛查，不是实测板端时延。 |
| 算子资源筛查 | LUT（查找表）≤161280；DSP（数字信号处理单元）≤1209；BRAM18（18 Kb 块 RAM 等价量）≤436 | 在 H4 算子峰值上执行筛查；通过仅说明未违反这一必要约束。H4 未测 shell、互连、DDR、DMA 和完整并行部署的资源，不能据此声明整网可容纳。 |
| 整网正式资源门槛 | 完整 post-route（布局布线后）设计，含 AXI/DMA/控制 shell；LUT≤161280、DSP≤1209、BRAM18≤436（BRAM36≤218）、URAM≤67 | 必须使用实际设计资源。FF≤322560 为记录/建议值。设备总容量不作为应用预算。 |
| 整网正式时序门槛 | 5 ns 目标周期；WNS（最差建立时间裕量）≥0.2 ns；TNS（总负裕量）=0 | 正时序不自动等于达到已有 0.2 ns 门槛。 |
| 板端正式性能门槛 | 时延≤25 ms；动态功耗≤15 W；动态能量≤375 mJ/操作 | 动态量须使用配对空闲基线和实际积分。总功耗与总能量不能替代动态量。 |
| 正式测量质量门槛 | 空闲/运行配对≥5 次；捕获窗口≥60 s；空闲/运行样本各≥100；完成操作和计时操作各≥1000；变异系数≤0.05 | 现有声呐检测的每组 10 次重复不等于该预注册正式测量质量门槛已通过。 |

旧检测配置里的 `max_bram=700` 大于 ZU7EV 的 624 BRAM18 设备总容量，且缺少明确单位，不能继续当作有效 ZCU104 预算。V2 显式使用已有预注册 `bram18_max=436`，保留旧文件作为历史证据。BRAM36 与 BRAM18 按 2 倍等价量换算，避免把 51 个 BRAM36 当成 51 个 BRAM18。

无法精确查得某个架构、位宽或映射时，原 API 返回 `NEEDS_MEASUREMENT`，主要成本留空。新筛查保留该状态并列出缺项，不用临近形状、INT8 替代混合精度或回归外推填数。未知成本不进入可晋升前沿，也不被表述为实证不可部署。

## V2 实际本地预检结果：已有预算下没有可晋升候选

本轮预检已经落盘：[cost_snapshot.json](../results/sonar_joint_v2_preflight_20261006/cost_snapshot.json) 和 [plan.json](../results/sonar_joint_v2_preflight_20261006/plan.json)。本页读取这两个实际文件核对统计，没有以合成夹具数值代替结果。

| 项目 | 实际结果 |
|---|---:|
| 架构数 | 144 |
| 架构×3 量化合同×6 映射候选 | 2592 |
| 精确成本查表 `LOOKED_UP` | 2592/2592 |
| 通过延迟和资源筛查的候选 | 0/2592 |
| 通过筛查的架构 | 0/144 |
| 拒绝原因 `OVER_BUDGET_Latency` | 2592/2592 |
| 最短算子串行 HLS 估计 | 4509321.045 μs = 4.509321045 s |
| 最长算子串行 HLS 估计 | 18611997.455 μs = 18.611997455 s |
| 状态 | `NO_BUDGET_FEASIBLE_TRIPLES` |

本轮所有算子资源峰值通过对应筛查，全部候选因延迟超出已有 120 ms 预算被拒绝。最短估计也约为预算的 37.58 倍。不能用算子资源的 sum（求和）列冒充 peak（峰值）解释拒选；该代理也不等于整网 routed 资源。

这是应保留的负结果，未放宽预算。新 `freeze` 入口在零可行架构时拒绝形成训练冻结，因此没有据此启动新训练。下一项可检验的工作是完善能达到既有时延目标的映射/执行器及其真实成本证据，而非继续按浮点精度选出已经超预算的候选。当前新量化晋升与三种子确认逻辑仅处于代码验收层，尚无实际 V2 train/PTQ 结果。

已有部署实测也已通过独立脚本 [audit_existing.py](../tools/sonar_joint_v2/audit_existing.py) 归约检查，最终实际产物为 [existing_deployment_budget_audit_final.json](../results/sonar_joint_v2_preflight_20261006/existing_deployment_budget_audit_final.json)：D/C/B1 均为 `BUDGET_FAIL`，总状态 `EXISTING_DEPLOYMENT_BUDGETS_NOT_MET`。该工具记录原始报告、VCC12 传感器 CSV 和 D 的 route 文件 SHA，复用 [deployment_budget.py](../tools/sonar_joint_v2/deployment_budget.py) 检查原预注册的资源、周期、时序、时延、动态能量、温度及测量质量。捕获时长取空闲和运行窗口的最小值，已有约5秒空闲窗口未通过60秒门槛。优化映射与配置比特流匹配、功耗是否高于噪声底线等未证项保持未知，`full_deployment_acceptance=false`、`hardware_launched=false`。C/B1 不从 D 的原始 route 文件外推资源和时序，其 valid99 缺项记为未知而非测得不一致。

可复算命令只写到新目录，不启动训练或硬件；同一输出目录不得重复覆盖：

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_joint_v2/run.py plan --output results/NEW_SONAR_JOINT_V2_PREFLIGHT
```

## 当前可核验的量化与部署证据

### 三种子软件量化与整数精度

[量化有效性核验](../reports/sonar_quantization_effectiveness/QUANTIZATION_EFFECTIVENESS.md)、[配对数据](../reports/sonar_quantization_effectiveness/PAIRED_RESULTS.csv) 和 [来源哈希清单](../reports/sonar_quantization_effectiveness/AUDIT_MANIFEST.json) 绑定同一网络、seed 和 `ALL_BEST_PT` checkpoint。D/C/B1 分别有 seeds42/43/44 的纯整数评估，不能将旧 PTQ 指标直接复制成纯整数指标。

| 网络 | FP32 三种子均值 mAP50:95 | PTQ 三种子均值 | 纯整数三种子均值 | 纯整数 Recall（召回率）均值 |
|---|---:|---:|---:|---:|
| D | 0.026014 | 0.022216 | 0.022223 | 0.350168 |
| C | 0.032762 | 0.023835 | 0.023748 | 0.377104 |
| B1 | 0.032770 | 0.027258 | 0.027275 | 0.430976 |

依据 [INTEGER_ACCURACY_GATE.json](../reports/sonar_real_int8_deployment_v1/INTEGER_ACCURACY_GATE.json)，B1/C 三种子均值 mAP50:95 和 Recall 都高于纯整数 D，属于相对底线 PASS，不证明应用准确率充足。依据原 Phase5A 绝对下降 0.003/相对下降 20% 门槛，三网络 PTQ 都未达到 `PTQ_SUFFICIENT`；QAT（量化感知训练）恢复尚未被证实。不得把“整数实现一致”写成“量化没有精度代价”。

### 三网络重复板端性能与功耗

实际完整性能产物位于 [formal_performance_20260928T231114](../results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114/PERFORMANCE_SUMMARY.csv)，该目录的 [STATUS.json](../results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114/STATUS.json) 为 `FORMAL_PERFORMANCE_COMPLETE`。逐候选记录为 [D](../results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114/D/PERFORMANCE_REPORT.json)、[C](../results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114/C/PERFORMANCE_REPORT.json)、[B1](../results/sonar_real_int8_deployment_v1/formal_performance_20260928T231114/B1/PERFORMANCE_REPORT.json)。

协议：seed42、INT8/W8A8、batch1、输入640、权重驻留；native（板端图执行）与 E2E（端到端主机事务）分别预热 1 次＋测量 10 次，无中间层核对。实际板端时钟 187.498125 MHz。E2E 的主机角色为 server CPU，不能表述为 ARM 独立部署。FPS 是测量窗口的完成图数/墙钟时长。

| 网络 | native 中位数（s/图） | E2E 中位数（s/图） | E2E FPS | 板输入平均功耗（W） | 总能量（J/图） | 扣空闲的诊断动态能量（J/图） |
|---|---:|---:|---:|---:|---:|---:|
| D | 131.863618 | 131.911043 | 0.007581 | 10.438624 | 1376.442240 | 27.197767 |
| C | 137.292034 | 137.341008 | 0.007281 | 10.435631 | 1432.731413 | 25.951956 |
| B1 | 366.688445 | 366.753376 | 0.002727 | 10.462555 | 3836.492541 | 95.156144 |

原表能量单位为 mJ，本表除以 1000 显示为 J。功耗来自 ZCU104 VCC12 INA226 整板输入测量，积分覆盖板端推理调用窗口，排除主机能量；原始样本在各候选 `native_results/power/power_samples.csv`。上述秒级时延远高于既有 25 ms 准入门槛；动态能量也高于 375 mJ。其主要意义是可执行性和串行瓶颈证据，不是硬件效率收益或预算达标。

归档路由资源为 LUT=25531、FF=28926、DSP=83、BRAM36=51（BRAM18=102），WNS=0.161 ns、TNS=0、实际路由报告时钟187.512 MHz。直接原始文件见 D valid99 归档的 [timing.rpt](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build/board_evidence/timing.rpt)、[utilization_hier.rpt](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build/board_evidence/utilization_hier.rpt) 及 [route_status.rpt](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build/board_evidence/route_status.rpt)。本次已核对三份文件及 bitstream 的 SHA 与该目录 [board_manifest.json](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build/board_evidence/board_manifest.json) 一致。资源来自三候选复用的同一通用执行器，不能从资源相同推出时延或功耗相同。WNS 为正但低于正式 0.2 ns；当前执行器也不是 H4 算子映射前沿的实现，不能用它校准全部六种 H3 映射。

### D 的完整 valid99 板端精度

10/02 回收的原始返回见 [D_ORIGINAL_RETURN_RECOVERY_20261002.json](../reports/sonar_real_int8_deployment_v1/D_ORIGINAL_RETURN_RECOVERY_20261002.json)。它记录归档 SHA256=`fbb0e5ef3a208afb975289136da3d7fc83f62e957b02919a615ce5ee12288c11`、107 帧（train8＋valid99）、48 节点、真实权重绑定、零失配，`hardware_rerun=false`。精确原包是 [board_evidence/board_return.tar.gz](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/graph_build/board_evidence/board_return.tar.gz)，本次已重核原包 SHA，并只读调用 `board_result.validate_actual_frames` 校验 107 帧 head 字节、零填充及 train8 逐层 trace，结果 PASS。`graph_build/returned_evidence.tar.gz` 是另一份二次包，不能用其 SHA 代替该原始返回包。

对应 [D/VALID99_REPORT.json](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/D/VALID99_REPORT.json) 为 `VALID99_PASS`：

| 项目 | 值 |
|---|---:|
| seed | 42 |
| valid 图数 | 99 |
| AP50 | 0.06711092928890139 |
| AP75 | 0.010836590881808661 |
| mAP50:95 | 0.02123051038943231 |
| Precision（精确率） | 0.028189910979228485 |
| Recall | 0.3838383838383838 |
| 输出失配 | 0 |

报告的 `integer_metric_differences={}`，与同 seed 软件整数指标一致。数据身份：`deployment_binding_sha256=2bfd854bc4f94156cb8aae85d46a773cf10d96fa5483c6938dc3959d5e4899f8`；`graph_sha256=de6516bbb7e4b4f66da8d5794f12974513a5fc22fd909c1e2b81345711ad8f36`。报告仍明确 `full_deployment_acceptance=SEPARATE_FINAL_REVIEW_REQUIRED`；不能将 D 的一条 seed42 valid99 PASS 扩大为所有候选或三种子板端精度完成。

C/B1 的 valid99 在该归档目录未见完成报告。[RESUME_FAILURE.json](../results/sonar_real_int8_deployment_v1/valid99_restart_20260930T213829/RESUME_FAILURE.json) 保存 `FAILED_NO_AUTORETRY`：C 板端准入读取 `/home/.../cpu_decode_v1_route02/D/seed_42/board.json` 时 `FileNotFoundError`。这是一条带日期的失败记录，不据此推断实时远端状态；恢复时应修复/验证证据路径和身份，不能伪造缺失 D 收据或重新运行已完成 D 事务。

以上均为检测数据的开发 valid 协议，`test_evaluated=false`。分类 macro_f1、Top-1 与旧 AV7K325 指标没有用于本次量化选择或检测结果比较。

## V2 的独立实现合同

本轮配置路径为 `configs/search/sonar_joint_v2.yaml`，入口放在 `tools/sonar_joint_v2/`。代码验收和真实科研执行分别记账：前者可以通过静态检查、合成夹具与本地只读预检验证；后者必须有实际 train/PTQ 收据和冻结身份，代码测试不能制造实验精度。

1. 联合候选：为每个架构保留三个既有量化合同和六个既有映射策略，调用原 `CostDatabase.estimate_cost(architecture_id, profile, mapping)`，记录合同、图、映射和测量身份。
2. 预算前置：按原检测延迟筛查值和明确 BRAM18 单位的既有资源门槛检查算子成本；未知/冲突成本不进入晋升；整网准入状态保持独立。
3. 逐轮量化与晋升：40、60、80 每轮使用各自预算 checkpoint；校准仅用冻结 train 前缀，评估仅用 valid。量化分数与延迟、LUT、DSP、BRAM 合同共同参与前沿排序，保留临界候选，记录每个弃选原因；不能把最终 checkpoint 的分数回填到较早阶段。
4. 弃选抽查与最终确认：保留原宏结构分层抽查思想，抽查使用量化协议复核误淘汰；最终 seeds42/43/44 绑定实际 checkpoint/profile，执行原绝对/相对精度下降和改进 seed 数门槛。失败和缺项均保留，不发布部分数据组成的最终联合前沿。
5. 部署交接：候选、权重、量化合同、参数包、帧宽、loader 和 bitstream（配置比特流）一致后，才可讨论该候选的整网准入。软件选择成功不触发 HLS、板测或功耗作业，也不等于正式部署成功。

可复用 API：

| API | 用途 | 禁止扩大解释的边界 |
|---|---|---|
| `tools/fpga_h4/cost_api.py: CostDatabase.estimate_cost` | 精确已测算子查表；支持已冻结架构、三个量化合同和已测映射 | 不预测未知形状/映射；成本仍是算子级 HLS，shell=`UNKNOWN`。 |
| `tools/fpga_h4/accuracy_api.py: lookup_accuracy` | 已有 B1/C/D 三种子量化合同与精度绑定 | 新运行各轮的实际量化精度需用其独立收据；不能被旧网络精度代替。 |
| `tools/sonar_search_finalize/run.py: Evidence / finalize` | 独立验收完整 v1 归档；缺项返回 `PENDING_CAMPAIGN_EVIDENCE` | 当前校验器严格检查 v1 的 FP32 晋升顺序，不是 V2 新策略的结果校验器；不要改写它以重新解释 v1。 |
| `tools/sonar_deploy/admission.py: accuracy_admission / stage_evidence / action_admission` | 现有 B1/C/D 部署证据与动作准入 | 当前部署白名单是冻结 B1/C/D，新 V2 候选尚不能无合同直接接入。 |
| `tools/sonar_deploy/board_result.py: validate_actual_frames / verify_board_result` | 对真实 train8/valid99 返回字节及 bitstream/包/图身份做零失配校验 | 不运行板子，不用软件输出冒充返回字节。 |
| `tools/sonar_deploy/power.py: paired_energy / verify_iio_evidence` | 对配对原始传感器样本积分及身份校验 | 总功耗、动态功耗、整板范围、主机范围和正式测量质量门槛需分别记录。 |

## 冻结、历史文件与完成边界

[results/sonar_joint_v1/freeze.json](../results/sonar_joint_v1/freeze.json) 绑定 190 个文件，包括 `tools/sonar_search/run.py`、`worker.py`、旧 search config、`tools/fpga_h4/` 的成本/精度/准入实现、合同和 LUT 产物。旧 `freeze.json`、`snapshot.json`、`source_snapshot.zip`、`cost_snapshot.json`、`Q_OF_X.json` 及既有结果不覆盖。本轮新增代码、配置和产物必须独立命名，不同步覆盖远端冻结树。

[SONAR_WORKFLOW_CURRENT.md](SONAR_WORKFLOW_CURRENT.md) 是 9/22 历史观测，里面的“正在训练”“整网执行器未完成”不能继续作为当前状态。同理，9/28 的 [CURRENT_BOARD_MEASUREMENTS.md](../reports/sonar_real_int8_deployment_v1/CURRENT_BOARD_MEASUREMENTS.md) 和 [DEPLOYMENT_PROGRESS.json](../reports/sonar_real_int8_deployment_v1/DEPLOYMENT_PROGRESS.json) 未纳入上述更晚性能/valid99 证据。原文保留其时间和范围，本页用具体新产物补充。

本地 `results/sonar_joint_v1/` 本轮未发现完整三阶段 train/PTQ receipts；其“training_started=false”是冻结时字段，既不证明未曾启动，也不证明已经完成。本次不作实时训练结论。独立收尾器测试使用临时合成分数/checkpoint 字节，属于工具测试，不是训练实证。

本轮完成门槛是：新入口能在逐轮选择中执行联合候选、明确预算、量化/硬件前沿和临界/弃选复核，独立记录来源及缺项，且不修改旧冻结。真实 V2 精度收益、全空间优越性、C/B1 valid99、优化映射部署和正式板端性能准入仍需各自实际证据。

最终代码回归验收为 **148 passed，76.31 s**：新政策、集成、worker 和部署预算四个测试文件共91例；旧独立收尾器与 H4 两个回归文件共57例。实际命令如下，测试分数和 checkpoint 使用隔离合成夹具，不构成真实 V2 精度或训练证据：

```powershell
.venv_cuda/Scripts/python.exe -B -m pytest tests/test_sonar_joint_v2.py tests/test_sonar_joint_v2_integration.py tests/test_sonar_joint_v2_worker.py tests/test_sonar_joint_v2_deployment_budget.py tests/test_sonar_search_finalize.py tests/test_fpga_h4_hardware_agent.py -q -p no:cacheprovider
```

本轮只读复核发现并已由实现修复：源归档 SHA 直接绑定 freeze，ZIP 成员逐项核对；H4 动态加载的 H3 源码进入新冻结；PTQ profile 对 checkpoint、seed、预算、校准、split 和 freeze 的 binding（身份绑定）及位宽语义逐项核对；相对配置路径在冻结前解析。旧冻结文件未被这些修复改写。

本次复核属于同一协作团队的实现审查与本地回归测试，没有外部独立模型验收，不将该门槛标为通过。真实成本预检的结论仍为0/2592可行、未启动新训练；D真实返回字节与原包哈希已重核；三网络实际测量的预算归约为失败，科研收益和最终部署准入保持未建立。
