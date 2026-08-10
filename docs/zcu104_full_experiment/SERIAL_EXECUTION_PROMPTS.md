# ZCU104 串行执行 Prompts

## 使用说明

本文件中的 prompt 按 P0→P11 顺序使用。每一个 prompt 都要求实际推进到通过门、明确失败或真实外部阻塞，不允许停留在“代码已准备”“smoke通过”之类中间状态。

P0 的唯一集成工作树已经固定为 `E:\1\hwnas\hwnas-zcu104-full-campaign-v1`；后续 prompt 只能在该集成工作树或总控登记的三个执行工作树中运行。服务器连接使用：

```text
ssh -p 7889 jiajia@10.70.23.121
```
认证信息必须交互输入或通过安全凭据机制提供，不得写入仓库、命令记录、日志、环境卡或 prompt 产物。

---

## 总控 Prompt：完整修改与实验闭环

如果希望由一个长任务负责从修改一直推进到实验闭环，可以直接使用下面的总控 prompt。它仍必须按照 P0–P11 的门顺序执行。

```text
你负责E:\1\hwnas\hwnas项目的ZCU104完整迁移、修复和正式实验闭环。先完整阅读docs/PROJECT_MEMORY.md、docs/FORMAL_EXPERIMENT_CONTRACT.md以及docs/zcu104_full_experiment/目录下全部文件，并把SERIAL_EXECUTION_PROMPTS.md中的P0-P11作为强制执行顺序。

保护当前包含用户修改的主工作树。所有实质修改和实验必须在从codex/sonar-fourstage-operator-closure已推送提交创建的独立worktree/分支中完成。不得reset、覆盖或整理用户无关修改。服务器使用ssh -p 7889 jiajia@10.70.23.121；认证信息只能安全交互输入，禁止写入文件和日志。

核心技术主线固定为：Protocol V2四候选C0-C3 -> external-scratch/shared-compute完整网络 -> ZCU104 PS DDR/AXI DMA/PL accelerator -> 完整INT8/Route/板级全样本验证 -> block-level延迟 -> 外部board_input_total功率/能耗 -> ZCU104代理校准 -> 正式RL/Random/Aging搜索 -> 三个新Pareto候选完整复测。

必须保持软件精度、INT8、HLS、Route、板级分类、延迟、Vivado估计、外部实测功率和NAS代理分层。任何阶段只在对应gate真实通过后进入下一阶段。失败、TIMEOUT、RESOURCE_INFEASIBLE、NOT_MEASURED必须保存，不能用历史AV7K325或其他候选结果补齐。

先完成代码/协议冻结、ZCU104 profile、external-scratch资源可行实现、RTL验证门、PS-PL runtime、通用transport和功率backend。然后按EXPERIMENT_PLAN.md执行E0-E10，按METRICS_AND_ACCEPTANCE.md采集全部指标和统计重复。长任务无需高频轮询，只在gate变化、失败、阻塞或最终完成时更新。

每一步记录命令、config、seed、环境、原始日志、输出路径和SHA256；运行针对性测试、git diff --check和完整pytest。代码测试通过不自动代表实验完成。遇到需要新增硬件、安装工具、修改网络/板卡非易失状态或其他新授权时，停止相应lane并明确报告，不扩大权限。

最终交付：clean commit链、campaign ledger、机器可读JSON/CSV、原始测量、统计报告、Pareto表、失败清单、复现命令和结论边界。只有P0-P11要求和最终ledger全部通过，才能声明ZCU104完整实验完成。
```

---

## P0：创建干净实验基线并冻结协议

```text
你正在处理 E:\1\hwnas\hwnas 的 HW-NAS/FPGA/声呐项目。目标是为 ZCU104 创建正式完整实验基线。

先只读检查：
1. docs/PROJECT_MEMORY.md
2. docs/FORMAL_EXPERIMENT_CONTRACT.md
3. docs/zcu104_full_experiment/README.md
4. docs/zcu104_full_experiment/MODIFICATION_PLAN.md
5. docs/zcu104_full_experiment/EXPERIMENT_PLAN.md
6. docs/zcu104_full_experiment/SONAR_AND_THERMAL_METRICS.md
7. artifacts/measurement_first_rebuild/status.json
8. codex/sonar-fourstage-operator-closure 分支及 E:\1\hwnas\hwnas-sonar-fourstage-operator-closure 工作树

当前主工作树有大量用户修改，不允许清理、reset、checkout覆盖或直接合并。复验现有 `E:\1\hwnas\hwnas-zcu104-full-campaign-v1` / `codex/zcu104-full-campaign-v1`；若不干净、分支不符或有来源不明修改，立即停止，不得覆盖。通过后只能按并行登记文件创建硬件、runtime 和 evidence 三个执行工作树。

在新worktree中建立：
- configs/experiments/zcu104_full_campaign_v1.yaml
- artifacts/zcu104_full_campaign_v1/protocol_freeze.json
- artifacts/zcu104_full_campaign_v1/source_freeze_manifest.json
- artifacts/zcu104_full_campaign_v1/environment_card.json
- artifacts/zcu104_full_campaign_v1/status.json

冻结四个候选C0-C3的arch_id、architecture_sha256、15个checkpoint索引、数据split、Protocol V2、代码commit和工具版本。当前仅四个代表 checkpoint 已绑定；60 个 checkpoint 完成正式 SHA256 登记前，相应门必须保持 `PENDING`。不得重新访问outer结果挑选候选。服务器密码不得写入文件。

为新增配置和冻结审计增加测试，运行针对性测试、git diff --check和完整pytest。若完整测试存在历史失败，区分本次引入失败和既有失败。

最终报告：新worktree、branch、commit、dirty状态、冻结文件、候选数量、checkpoint数量、测试结果、未通过项和下一门。只有source freeze可重验且worktree clean才标记P0 PASS。
```

---

## P1：确认 ZCU104 与服务器能力

```text
在已冻结的 ZCU104 独立worktree中，只读确认服务器和板卡能力，不修改板卡非易失状态，不中断已有Vivado/XSim任务。

服务器：ssh -p 7889 jiajia@10.70.23.121。认证信息不得落盘。

需要确认并生成 machine-readable capability card：
- Ubuntu/kernel/CPU/RAM/磁盘空间
- Vivado与Vitis HLS版本、license和exact part支持
- ZCU104 JTAG target和device ID
- 三路UART与/dev/serial/by-id映射
- 启动模式、板端OS/PYNQ/PetaLinux、网络接口和IP
- 板卡序列号、时钟来源、供电方式
- 是否存在USB005/PMBus或外部功率仪
- ZynqMP AMS/System Monitor 是否可用：检查 CONFIG_XILINX_AMS、device tree、IIO `in_temp*_input` 节点、label、单位和只读采样；如 Linux 不可用，再用 Vivado/JTAG 交叉核验
- 外部环境温度探头与风扇 tach 是否可读；不可读时分别标记 `ambient_temperature=NOT_MEASURED` 和 `fan_rpm=NOT_AVAILABLE`
- 服务器ACPI功率读数的measurement scope，明确它不能作为板卡功率
- 当前运行的Vivado/XSim进程和资源冲突

不得把“板卡具备PMBus”推断为“当前服务器已经能采集功率”。不得把UART当成功率通路。

输出到 artifacts/zcu104_full_campaign_v1/environment_card.json 和 board_capability_card.json，并保存只读探测命令和脱敏日志。对exact part、board-part、串口、板端OS、网络、功率仪分别给出PASS/PENDING/BLOCKED。

只有JTAG exact device、至少一种控制通路和工具链确认后，P1硬件bring-up才可PASS；功率仪可以单独保持PENDING，但必须明确其阻塞正式功率实验。
```

---

## P2：实现 ZCU104 board profile 和 Vivado 工程骨架

```text
基于已冻结的ZCU104实验分支，实现板卡profile和可复现Vivado工程骨架。先读现有boards.py、board_av7k325.yaml、generate_harness.py、XDC/Tcl模板和相关测试。

完成：
1. 在src/hwnas_fpga/hardware/boards.py增加ZCU104/XCZU7EV profile和别名；资源值必须由官方资料与Vivado查询交叉验证。
2. 新增configs/hardware/zcu104.yaml。
3. 新增hls_lut_builder/board_harness/configs/board_zcu104.yaml。
4. 将generate_harness.py的7-series/MMCME2耦合抽象为board/clock backend。
5. 正式路径支持PS FCLK；若保留PL时钟，必须使用UltraScale+合法原语。
6. 新增Zynq MPSoC + AXI4-Lite + AXI DMA + HP/HPC + interrupt + reset/clock的BD Tcl模板。
7. 自动导出bitstream、XSA、HWH、utilization、timing、DRC和route status。

不得运行完整候选长Route；先做工程生成、Tcl语法、BD validate和最小接口smoke。增加ZCU104 profile/harness/BD契约测试，保持AV7K325测试兼容。

输出修改文件、exact part、时钟方案、地址映射、接口宽度、测试命令和结果。只有工程生成可复现、BD validate通过且旧板回归未破坏，P2才PASS。
```

---

## P3：修复完整网络 external-scratch HLS 主线

```text
目标是把四阶段完整网络从资源不可行的HWC全并行实现迁移到可部署的external-scratch/shared-compute实现。禁止改变Protocol V2冻结的网络语义和候选身份。

先审计现有：
- fourstage_external_scratch_hls_summary.json
- fourstage_hwc_parallel_hls_summary.json
- fourstage_hwc_parallel_hls_v2_summary.json
- 最新sample1路径过长失败日志
- Python INT8 reference和C-sim zero-mismatch证据

在服务器Linux短路径中实现并验证V0：
- PS DDR external scratch语义
- ping-pong feature buffers
- AXI burst和对齐
- 共享K3/K5、depthwise、pointwise计算单元
- 可配置P_in/P_out/P_pix
- 运行时权重加载接口
- kernel/DMA/stall/invocation计数器

先对C0做最小可复现闭环，再按C0-C3顺序扩展。每个候选都要生成源码、testbench、Tcl、命令、return code、完整日志、csynth XML、资源和hash。不得把生成XML但进程返回1算作PASS。

验收：四候选C-sim零失配、HLS返回0、资源低于器件冻结阈值、无地址越界、结果可从manifest复现。若资源超限，继续修改数据复用和共享结构，不要通过减小冻结网络宽度/深度规避。报告每次实现版本的资源、cycles和失败原因。
```

---

## P4：建立可执行的 RTL 验证门

```text
修复当前完整网络RTL co-sim TIMEOUT门，但不得简单删除验证要求。

建立三级验证：
1. 所有正式算子shape的RTL co-sim，包括真实、随机和边界输入；
2. C0、C2、C3的完整网络单样本RTL co-sim；
3. 完整网络C-sim与Python INT8的全测试集零失配。

缩短工程和文件路径；测试向量使用外部二进制文件，不把大数组展开进C++；记录仿真吞吐、预计完成时间和timeout。先运行一个最小样本可行性试验，再决定正式timeout。

如果完整网络单样本仍被证明需要不可接受的多小时，生成正式协议修订草案：用算子RTL co-sim + 完整网络C-sim + 后续板级全样本零失配构成验证金字塔，说明剩余风险、适用范围和停止条件。修订必须在板级正式数据采集前冻结。

输出RTL验证manifest、全部命令、日志、mismatch、timeout、证据边界和下游Route是否获准。任何非零mismatch都阻止Route。
```

---

## P5：实现 ZCU104 正式 PS–PL runtime 和通用 transport

```text
实现ZCU104 A53到PL accelerator的正式运行时。Python/PYNQ仅用于bring-up，正式测量使用固定版本C/C++ runtime。

实现：
- bitstream/XSA/HWH识别
- INT8权重和量化参数加载
- 连续DDR buffer、对齐、cache flush/invalidate
- AXI DMA提交/完成/timeout
- AXI4-Lite控制和interrupt
- 单样本分类
- 缓存输入RUN_REPEAT
- kernel、H2D、D2H、runtime、system E2E计时
- error/reset/health状态
- JSON/JSONL receipt

将run_dynamic_validation.py、run_power_repeat.py、validation_report.py、acceptance_audit.py和board_protocol.py通用化为transport_type/transport_id，移除必须等于COM5或字符串包含COM5的判据，同时保持AV7K325串口兼容。

deployment binding新增board_id、device_part、bitstream、XSA、HWH、runtime、checkpoint和validation manifest的SHA256。

增加transport mock、timeout、CRC、hash mismatch、cache/length/address边界测试。先完成host-side和mock测试，再做板上固定向量smoke。不得把smoke称为全样本分类。
```

---

## P6：运行 ZCU104 HLS、Full Route 和 bitstream 门

```text
只在P0-P5通过且候选/协议/工具链冻结后执行ZCU104正式构建。

主实验：C0-C3，200 MHz，预注册route seed。为每个candidate×clock使用独立输出目录，绑定candidate/checkpoint/HLS/source/toolchain hash。

按顺序执行HLS、IP export、Vivado synthesis、implementation、report_timing_summary、report_utilization、report_route_status、report_drc、bitstream、XSA/HWH导出。自动解析LUT/LUTRAM/FF/DSP/BRAM18/BRAM36/URAM、WNS/TNS/WHS/THS、unconstrained paths、route errors、critical DRC、congestion和wall time。

C2/C3额外运行两个route seed评估实现波动，但主结果必须使用预注册seed，不允许挑最优seed。200 MHz失败时先执行一次预注册实现优化；仍失败则保留FAIL并将150 MHz作为独立回退工作点，不覆盖失败。

不要运行板级和功率实验，直到exact bitstream达到WNS>=0、TNS=0、hold clean、route errors=0、critical DRC=0、资源门通过。Vivado power只标记为estimated_on_chip_power。
```

---

## P7：执行板级全样本数值一致性和分类

```text
前提：P6 exact bitstream通过Full Route，P5 runtime和deployment binding通过。

按三级执行：
1. 固定向量smoke，验证状态码、CRC、cycles、logits；
2. 真实/随机/边界向量，验证Python INT8与板级数值；
3. C0-C3各一个预注册anchor checkpoint的完整outer split逐样本验证。

anchor必须按预注册规则固定，不能根据outer Macro-F1挑选。每条记录保存sample_id、label、prediction、Python/HLS/board logits、CRC、status、retry、kernel/DMA cycles和完整hash chain。

使用run_dynamic_validation.py的ZCU104 transport，输出JSONL、summary、confusion matrix和acceptance report。已有记录若bitstream、manifest或deployment binding hash不一致必须拒绝复用。

通过门：coverage=100%、missing=0、duplicate=0、CRC/status error=0、INT8 logits mismatch=0、argmax mismatch=0、deployment binding PASS。任何不满足项保持FAIL，不进入正式延迟/功率结论。

增强实验：若需要报告板级5×3平均准确率，则为C2运行全部15个checkpoint；未完成时只报告anchor checkpoint板级结果。
```

---

## P8：执行正式延迟、吞吐和稳定性实验

```text
在板级全样本正确性通过后，对C0-C3执行正式性能测量。

固定：batch=1、200 MHz、CPU governor、CPU affinity、runtime、board、fan、温度策略和输入集合。每候选预热100次，并按预注册温度稳态规则等待；运行5个独立block，每block 1000次推理。

分开记录preprocess、H2D DMA、kernel cycles/latency、D2H DMA、postprocess、runtime overhead和system E2E。输出每block和跨block的mean/std/p50/p90/p95/p99/CV、throughput、timeout/retry/error。

冷启动单独测bitstream load、runtime init、weight load和first inference，不并入稳态延迟。

C2/C3额外在100/150/200 MHz运行频率敏感性；每个频率使用独立route-clean bitstream。不得把不同频率或不同latency scope混入同一候选值。

用ZynqMP AMS/System Monitor建议按1 Hz采集PS/PL结温，使用monotonic timestamp与latency receipt对齐。报告temp_start/mean/max/end、temp_slope、time_to_steady及overtemperature/throttle事件。环境温度只能来自外部探头；没有探头时必须为NOT_MEASURED。

统计单元是block，不把5000次相关推理当作5000个独立实验重复。输出原始记录、summary、CI、环境、温度和hash chain。
```

---

## P9：接入功率仪并执行功率/能耗实验

```text
先确认外部仪器真实测量ZCU104板卡直流输入总功率，服务器ACPI功率不能使用。PMBus轨级数据只能作为辅助，必须标注rail_scope。

实现或配置仪器backend，执行已知负载/自检，记录manufacturer/model/serial/sample_rate/range/calibration/wiring。原始数据必须包含timestamp_s,power_w或voltage_v/current_a，并保存原始CSV。

对C0-C3执行5轮randomized block。每轮候选顺序由固定随机种子预先生成。每候选每轮：固定预热/温度稳态 -> idle>=60s -> active>=60s且>=1000次推理。输入和权重在active前缓存，bitstream编程、上传和日志导出不进入active窗口。

每个active block生成run_repeat receipt，绑定inference_count、UTC起止、bitstream/runtime/candidate/checkpoint hash。计算idle/active/dynamic power、total/dynamic energy per inference、FPS/W和block-level 95% CI。

运行power manifest和multi-candidate campaign audit。至少三个候选同仪器、同接线、同量程、同协议全部PASS后，power/energy才可进入Pareto。Vivado estimate、NAS power_w和配置max_power_w不能作为实测替代。
```

---

## P10：建立 ZCU104 硬件/功率代理校准

```text
使用已通过HLS、Route、板级延迟和外部功率审计的数据建立ZCU104校准，不导入AV7K325数值充当ZCU104真值。

分别校准analytic->HLS、HLS->route、route->board latency、resource/activity/runtime->board power。先使用四个部署候选和匹配消融，补充到至少8个semantic-safe完整网络样本；功率模型建议补充到12-20个完整网络点。

按architecture分组划分train和blind holdout，禁止同一架构的不同seed/clock记录泄漏到两侧。报告MAPE、median/P90/P95 APE、Spearman、Kendall、false-feasible、false-infeasible、prediction interval coverage、Top-K recall和shortlist coverage。

建议门：P90 APE<=25%、Spearman>=0.8、hard reject false-infeasible=0、shortlist HLS coverage=100%。不通过时只允许代理排序或defer-to-HLS，不允许硬拒绝，也不允许把power加入NAS主奖励。

更新独立的ZCU104 calibration artifact和campaign ledger，不覆盖历史AV7K325校准。
```

---

## P11：运行正式 ZCU104 HW-NAS 并完成最终报告

```text
只有P0-P10所需的G2/G4/board/power门全部通过后，才启动正式ZCU104 HW-NAS。

主方法比较为RL、Random、Aging Evolution，使用完全相同的搜索空间、5 outer folds、seeds42/43/44、candidate evaluations、每候选训练epoch/batch、独立候选evaluator、ZCU104硬件模型、约束和Pareto目标。Proxyless若仍是weight-sharing supernet则单列，除非先建立共同可交换预算。

每个outer unit独立搜索，outer数据不参与搜索、epoch或候选选择。保存完整轨迹、候选数、feasible/rejected/deferred、GPU-hours、最终architecture_sha256和source freeze。

从新搜索预注册冻结accuracy-first、knee-point、latency/resource-first。三者重新执行INT8、HLS、Full Route、板级全样本、延迟、功率和稳定性，禁止直接拼接第一轮候选物理结果。

最终报告分层呈现：clean Macro-F1、冻结退化协议下的 sonar robust Macro-F1/最差条件/保持率、声呐结构分析、INT8、HLS、Route、板级分类、kernel/system latency、PS/PL结温与热漂移、外部功率、total/dynamic energy、代理可靠性、搜索预算和Pareto。失败和负结果必须纳入附录/ledger。

只有最终evidence ledger全部通过才声明完整实验完成，并输出machine-readable CSV/JSON、Markdown报告、命令清单和SHA256索引。
```
