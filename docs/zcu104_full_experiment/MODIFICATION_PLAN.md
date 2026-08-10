# ZCU104 完整修改方案

## 1. 修改范围与原则

本方案覆盖代码基线、完整网络 HLS 数据流、ZCU104 板卡工程、PS–PL runtime、板级验证、功率采样、证据绑定、代价模型和测试。修改过程必须遵守以下原则：

- 保护当前脏工作树，不清理、不覆盖、不回退用户已有修改；
- 每条并行开发线使用独立分支、独立 worktree 和独立输出目录；
- 搜索代理、软件精度、HLS、Route、板级分类和外部功率分层保存；
- 修改协议或阈值必须发生在观察正式结果之前；
- 任何失败均保留原始日志和 SHA256，不以重新运行成功覆盖失败事实；
- 所有生成数据放入 `results/`、`artifacts/` 或 `logs/`，不散落到源码目录。

## 2. M0：建立唯一 ZCU104 实验基线

### 2.1 操作

1. 使用已经建立并复验的集成分支 `codex/zcu104-full-campaign-v1` 和工作树 `E:\1\hwnas\hwnas-zcu104-full-campaign-v1`。
2. `E:\1\hwnas\hwnas` 仅作为只读输入；三个执行分支必须从总控登记的 lane base commit 创建独立 worktree。
3. 审计该分支是否存在未跟踪实验产物；需要保留的产物写入 manifest 或归档，不需要的内容也不能未经检查直接删除。
4. 冻结：
   - Git commit；
   - Protocol V2；
   - 数据集与 split manifest；
   - 四个候选 `arch_id + architecture_sha256`；
   - checkpoint 路径、大小和 SHA256；
   - Python/PyTorch/Numpy/Vivado/Vitis HLS 版本；
   - 服务器和板卡环境卡。

### 2.2 建议新增

- `configs/experiments/zcu104_full_campaign_v1.yaml`
- `artifacts/zcu104_full_campaign_v1/protocol_freeze.json`
- `artifacts/zcu104_full_campaign_v1/source_freeze_manifest.json`
- `artifacts/zcu104_full_campaign_v1/environment_card.json`
- `artifacts/zcu104_full_campaign_v1/status.json`

### 2.3 验收

- worktree clean；
- source freeze 校验无 missing、unexpected、changed；
- 四个候选各有 15 个正式 fold/seed checkpoint 和预测索引；
- 所有正式入口均从 campaign config 读取参数，禁止散落的临时命令成为唯一协议来源。

## 3. M1：修复完整网络硬件数据流

### 3.1 当前问题

当前完整网络 HWC 并行实现通过了部分 HLS，但资源严重超限。现阶段不能依赖 ZCU104 资源较大而直接迁移；必须先把实现从“层级复制大量计算单元和片上特征缓存”改为“外部 scratch + 共享计算单元 + 可控并行度”。

### 3.2 正式主线

采用 external-scratch pixel-tiled 作为资源可行基线，并逐步优化：

1. 大特征图存放于 PS DDR；
2. 使用两个 feature buffer 做 ping-pong；
3. 通过 AXI burst 连续搬运 tile；
4. 共享 K3/K5 卷积计算引擎；
5. 深度卷积和逐点卷积使用可配置共享单元；
6. 参数化 `P_in`、`P_out`、`P_pix`，避免将并行因子写死；
7. 为可复用邻域增加 line buffer，减少 K3/K5 的 DDR 重复读取；
8. 支持层间 tile 调度，尽量减少中间结果回写；
9. 支持权重运行时加载，避免同一架构的 15 个 checkpoint 产生 15 个 bitstream；
10. 添加 kernel、DMA、stall、memory transaction 和 invocation 计数器。

### 3.3 版本演进

| 版本 | 目标 | 允许的结论 |
|---|---|---|
| V0 external scratch | 四候选资源可行、C-sim 正确 | HLS 基线，不宣称高效 |
| V1 burst + ping-pong | 降低 DMA 空洞和等待 | HLS/Route 性能改进 |
| V2 line buffer | 降低外存访问 | 可报告带宽和能耗变化 |
| V3 parallel-factor sweep | 形成多个可部署工作点 | 可用于 ZCU104 Pareto 校准 |

每次版本变化都必须重新生成 candidate/HLS/source hash，并保留前一版本失败或低效证据。

### 3.4 验收

- 四个候选 C-sim 零失配；
- HLS 正常返回，不以 XML 已生成代替成功返回码；
- 所有资源低于器件容量；
- 推荐 LUT/DSP/BRAM/URAM 各自低于 85%，不得高于 90%；
- external scratch 地址、长度、对齐和边界检查进入单元测试；
- 性能计数器在仿真和板级具有一致语义。

## 4. M2：修复 HLS 与 RTL 验证链

### 4.1 路径和测试向量

- Linux HLS 工程使用短路径，例如 `/data/jiajia/zcu104_hls/<campaign>/<candidate>/`；
- 不把数 MB 样本直接展开到生成的 C++ 源码；
- 输入、权重、量化参数和期望输出使用二进制或紧凑文本文件；
- testbench 保存输入 hash、参考输出 hash 和随机种子。

### 4.2 验证金字塔

1. Python INT8 reference：所有正式 checkpoint；
2. 完整网络 C-sim：四个 anchor checkpoint；
3. 算子 RTL co-sim：所有正式 shape 和边界输入；
4. 完整网络 RTL co-sim：至少 C0、C2、C3 的一个真实样本；
5. 板级全样本零失配：最终系统级数值证据。

如果完整网络 RTL co-sim 在预注册时间内仍不可完成，必须在板级数据采集前冻结协议修订，说明为何采用“算子 RTL co-sim + 完整网络 C-sim + 板级全样本一致性”的组合证据。不得在看到板级结果后临时删除 co-sim 门。

### 4.3 验收

- 任意 mismatch 非零立即阻止 Route；
- `TIMEOUT`、`FAIL` 和 `RESOURCE_INFEASIBLE` 独立保存；
- 所有 testbench 都能从 manifest 复现；
- 日志中记录命令、timeout、退出码和工具版本。

## 5. M3：增加 ZCU104 板卡与 Vivado 工程支持

### 5.1 修改文件

- `src/hwnas_fpga/hardware/boards.py`
- 新增 `configs/hardware/zcu104.yaml`
- 新增 `hls_lut_builder/board_harness/configs/board_zcu104.yaml`
- `hls_lut_builder/board_harness/scripts/generate_harness.py`
- `hls_lut_builder/board_harness/templates/build_bitstream.tcl.tmpl`
- 建议新增 `hls_lut_builder/board_harness/templates/zcu104_bd.tcl.tmpl`

### 5.2 板卡 profile

profile 至少包含：

- `board_name`、`device`、`target_part`、speed grade；
- LUT、FF、DSP、BRAM18、BRAM36、URAM；
- PS DDR 容量与数据端口；
- 默认/允许时钟；
- transport 和 power capability；
- Vivado board-part 是否可用；不可用时使用 exact part。

器件资源值必须从 Vivado `get_parts` / `report_utilization` 和官方资料交叉验证，不得复制 Kintex-7 profile。

### 5.3 时钟和复位

当前生成器中的 `MMCME2_BASE` 是 7-series 耦合。ZCU104 正式路径优先使用 PS FCLK；若使用 PL 时钟，需采用 UltraScale+ 合法时钟资源。时钟、reset polarity 和 clock domain crossing 必须进入自动检查。

### 5.4 PS–PL block design

建议包含：

- Zynq UltraScale+ MPSoC；
- AXI4-Lite 控制；
- AXI DMA 或等效数据搬运；
- HP/HPC DDR 端口；
- interrupt；
- clock/reset；
- accelerator IP；
- 可选温度/状态寄存器；
- bitstream、XSA、HWH 导出。

### 5.5 工具链

本批次统一使用服务器实际探测到的 Vivado `2022.2` 和 Vitis HLS `2022.2`。所有 ZCU104 HLS、Route、bitstream 和 runtime 必须绑定该版本；跨版本结果只能描述性单列，不得混入同一正式工作点。后续若安装版本变化，必须先更新环境卡、重新冻结并由总控批准，不能把建议版本当成已安装版本。

### 5.6 验收

- JTAG 识别 exact device；
- block design `validate_bd_design` 通过；
- synthesis、implementation、bitstream 返回码均为 0；
- WNS、TNS、DRC、route status 和 utilization 自动解析；
- bitstream/XSA/HWH 具有 SHA256 绑定。

## 6. M4：建立正式板端 runtime

### 6.1 分层

- bring-up：Python/PYNQ 或串口工具；
- 正式测量：固定版本 C/C++ runtime，运行于 ZCU104 A53。

### 6.2 runtime 功能

- bitstream/overlay 识别和加载；
- INT8 权重和量化参数加载；
- 连续 DDR buffer 分配和对齐；
- cache flush/invalidate；
- AXI DMA 提交和完成等待；
- accelerator 寄存器控制；
- kernel/DMA/runtime 周期计数；
- 单样本模式；
- 缓存输入 `RUN_REPEAT` 模式；
- 错误、timeout、reset 和健康状态；
- JSON/JSONL receipt 输出。

### 6.3 延迟语义

必须分别记录：

- `preprocess_latency_ms`；
- `h2d_dma_latency_ms`；
- `kernel_latency_ms`；
- `d2h_dma_latency_ms`；
- `postprocess_latency_ms`；
- `runtime_overhead_ms`；
- `system_e2e_latency_ms`。

bitstream 编程、首次模型加载、数据集从磁盘读取不计入单次推理，但必须单列 cold-start 指标。是否包含预处理必须在 campaign config 中固定。

### 6.4 运行控制

- 固定 CPU governor；
- 固定 CPU affinity；
- 记录 Linux image、kernel 和 runtime 依赖；
- 正式 active 窗口内禁止远程上传输入或编程；
- UART 仅用于调试或低速控制，不能把 UART 上传时间与 kernel latency 混写。

## 7. M5：通用化板级验证和 acceptance audit

### 7.1 修改文件

- `hls_lut_builder/board_harness/scripts/run_dynamic_validation.py`
- `hls_lut_builder/board_harness/scripts/run_power_repeat.py`
- `src/hwnas_fpga/hardware/validation_report.py`
- `src/hwnas_fpga/hardware/acceptance_audit.py`
- `src/hwnas_fpga/deploy/board_protocol.py`

### 7.2 修改要求

将 `COM5` 语义替换为：

- `transport_type`：`serial`、`axi_dma`、`xrt`；
- `transport_id`：稳定串口 by-id 或 runtime ID；
- `board_id`；
- `device_part`；
- `runtime_sha256`；
- `bitstream_sha256`；
- `xsa_sha256`；
- `hwh_sha256`。

报告不得依赖字符串中是否包含 `COM5` 来判断证据真实性。判据应依据状态码、完整 hash chain、route gate、样本覆盖和测量 receipt。

### 7.3 板级记录字段

- sample ID、label、prediction；
- Python/HLS/board INT8 logits；
- CRC、status、retry；
- kernel/DMA/system cycles；
- candidate/checkpoint/dataset/bitstream/runtime hash；
- board、clock、temperature；
- timestamp。

### 7.4 验收

- 当前 AV7K325 串口测试保持通过；
- 新增 ZCU104 transport mock 测试；
- 不同 transport 使用同一 evidence schema；
- hash 不匹配、样本缺失、重复、CRC 错误均 fail closed。

## 8. M6：功率测量系统

### 8.1 正式测量范围

主证据为 ZCU104 直流输入端 `board_input_total`。PMBus 单轨或多轨数据可并行采集，但必须标注 `rail_scope`，不能冒充整板输入功率。

### 8.2 修改文件

- `src/hwnas_fpga/hardware/power_measurement.py`
- `scripts/import_power_measurement.py`
- `scripts/audit_power_campaign.py`
- `hls_lut_builder/board_harness/scripts/run_power_repeat.py`
- 建议新增 `src/hwnas_fpga/hardware/power_backends/`
- 建议新增 `scripts/capture_zcu104_power.py`

### 8.3 功能

- 外部仪器驱动接口；
- UTC 时间同步和 monotonic clock；
- 原始 `timestamp_s,power_w` 或 `timestamp_s,voltage_v,current_a`；
- start/stop marker；
- idle/active block 编排；
- 仪器量程、采样率、序列号、校准信息；
- 板卡温度、风扇、频率；
- bitstream/runtime/receipt hash；
- 缺失值、重复时间戳、负功率和采样间隔异常审计。

### 8.4 验收

- 已知负载或仪器自检通过；
- 每个候选至少 3 组，正式建议 5 组 idle/active；
- 每组时长不少于 60 s；
- 每个 active block 不少于 1000 次推理；
- 编程和输入上传不在 active 窗口；
- 至少三个候选使用同一仪器、接线、量程和协议；
- raw CSV 和 receipt SHA256 可复核。

## 9. M7：ZCU104 LUT、代理和搜索反馈

### 9.1 重新建立硬件数据

AV7K325 LUT 不直接迁移。分别采集：

- 算子 HLS；
- 算子 post-route；
- 完整网络 HLS；
- 完整网络 post-route；
- 板级 kernel/system latency；
- 板级 total/dynamic power 和 energy。

### 9.2 校准层次

1. analytic -> HLS；
2. HLS -> route；
3. route -> board latency；
4. resource/activity/runtime -> board power；
5. latency + power -> energy。

### 9.3 最低要求

- 当前 G2 所需的 4 个 independent full-network probes；
- 至少 8 个 semantic-safe 完整网络样本；
- shortlist HLS coverage 100%；
- 功率代理建议至少 12–20 个完整网络测量点；
- 保留至少 20% architecture-level blind holdout；
- hard reject 使用时 false-infeasible 必须为 0。

## 10. 测试修改清单

建议新增或扩展：

- `tests/test_board_profiles_zcu104.py`
- `tests/test_zcu104_harness_generation.py`
- `tests/test_zcu104_bd_contract.py`
- `tests/test_zcu104_transport.py`
- `tests/test_zcu104_runtime_receipt.py`
- `tests/test_zcu104_deployment_binding.py`
- `tests/test_zcu104_power_manifest.py`
- `tests/test_zcu104_power_campaign.py`
- `tests/test_zcu104_metric_schema.py`
- 现有 `test_dynamic_validation.py`
- 现有 `test_board_protocol.py`
- 现有 `test_power_measurement_v2.py`

测试必须覆盖正常路径和以下 fail-closed 情况：

- 错误 part、clock、board ID；
- bitstream/runtime/checkpoint hash 不匹配；
- serial/AXI timeout；
- CRC 错误；
- 样本缺失或重复；
- route timing fail；
- 功率 CSV 缺列或时间倒退；
- active 时间不足或推理数不足；
- 不同仪器/协议混用；
- PMBus rail scope 被错误写成 `board_input_total`。

## 11. 建议提交边界

为降低并行冲突，建议按以下顺序形成独立提交：

1. campaign freeze 与文档；
2. ZCU104 board profile；
3. external-scratch 数据流；
4. HLS/RTL testbench；
5. ZCU104 block design；
6. runtime 和 transport；
7. board evidence schema；
8. power backend；
9. calibration/search integration；
10. 最终报告生成器。

每个提交先运行针对性测试，再运行完整测试；测试通过不自动等于实验门通过。
