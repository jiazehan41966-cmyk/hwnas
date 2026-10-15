# ZCU104 三执行线并行工作规范

> 历史版本：原四阶段ZCU104集成系列的计划/协议。下文的“当前”“本轮”“必须”仅指原文所绑定的系列，不是现工作区的执行指令。当前MobileNetA分类与检测边界见[方法说明](../method_design.md)和[配置版本索引](../SEARCH_CONFIG_CANONICAL.md)；历史产物与验收记录保持原样。

版本：v1.1

日期：2026-08-10

## 1. 唯一并行拓扑

并行准备只使用三条执行线，均从总控复验并登记的同一 lane base commit 建立：

| 执行线 | 分支 | 工作树 |
|---|---|---|
| hardware | `codex/zcu104-hardware-lane` | `E:\1\hwnas\hwnas-zcu104-hardware` |
| runtime | `codex/zcu104-runtime-lane` | `E:\1\hwnas\hwnas-zcu104-runtime` |
| evidence | `codex/zcu104-evidence-lane` | `E:\1\hwnas\hwnas-zcu104-evidence` |

机器可读的分支、工作树、输入提交、允许/禁止文件和交接文件以
`artifacts/zcu104_full_campaign_v1/control/parallel_work_registry.json`
为唯一登记源。本文是人类可读协议；若两者不一致，执行线必须停止并向总控报告。

所有执行线先读取公共契约和登记文件。服务器连接为
`ssh -p 7889 jiajia@10.70.23.121`；认证信息只允许交互输入，不得进入命令文件、日志、配置或产物。

## 2. 总控独占文件

只有集成总控可以修改：

- `configs/experiments/zcu104_full_campaign_v1.yaml`；
- `artifacts/zcu104_full_campaign_v1/status.json`；
- `artifacts/zcu104_full_campaign_v1/campaign_ledger.json`；
- `artifacts/zcu104_full_campaign_v1/control/parallel_work_registry.json`；
- 公共文档、正式实验总账、最终状态、最终报告和结论；
- 公共接口合并与正式 gate 提升。

执行线只提交局部代码、测试和交接 evidence bundle（证据包），不得把任何正式 gate 直接改成 `PASS`。

## 3. Hardware lane：硬件实现

允许修改：

- `src/hwnas_fpga/hardware/boards.py`；
- `configs/hardware/zcu104.yaml`；
- `hls_lut_builder/board_harness/configs/board_zcu104.yaml`；
- `hls_lut_builder/board_harness/scripts/generate_harness.py`；
- ZCU104 Vivado/BD/clock 模板；
- `scripts/generate_zcu104_platform.py`；
- `scripts/run_fourstage_external_scratch_hls_gate.py`；
- `scripts/run_fourstage_external_scratch_rtl_cosim_gate.py`；
- hardware lane 专属测试；
- `artifacts/zcu104_full_campaign_v1/handoffs/hardware_lane/` 和该 lane 的 `results/`、`logs/`。

禁止修改总控独占文件、板端 runtime、通用 transport、正式功率账本和其他 lane 的交接目录。

立即允许：ZCU104 profile、external-scratch/shared-compute 生成器、HLS testbench、BD/接口契约、mock/源码生成测试和短 smoke。

必须等待：完整四候选 HLS、正式 RTL co-sim、Full Route、bitstream、板卡编程。等待 source freeze/60 权重审计及相应前置门；完整实现还受服务器可用性和板卡 PL 供电影响。

交接文件：

- `artifacts/zcu104_full_campaign_v1/handoffs/hardware_lane/handoff.json`；
- `artifacts/zcu104_full_campaign_v1/handoffs/hardware_lane/board_interface_contract.json`；
- 测试清单、失败/超时清单、命令与输出 SHA256 索引。

## 4. Runtime lane：板端运行程序与模拟 transport

允许修改：

- ZCU104 A53 runtime 专属源码和构建文件；
- `src/hwnas_fpga/deploy/board_protocol.py`；
- `src/hwnas_fpga/hardware/deployment_binding.py`；
- `src/hwnas_fpga/hardware/validation_report.py`；
- `src/hwnas_fpga/hardware/acceptance_audit.py`；
- `hls_lut_builder/board_harness/scripts/run_dynamic_validation.py`；
- `hls_lut_builder/board_harness/scripts/run_power_repeat.py`；
- runtime lane 专属测试；
- `artifacts/zcu104_full_campaign_v1/handoffs/runtime_lane/` 和该 lane 的 `results/`、`logs/`。

禁止修改总控独占文件、板卡 profile、HLS 核心、Vivado/BD 模板、功率统计公式和其他 lane 的交接目录。

立即允许：host-side mock、AXI DMA/寄存器抽象、timeout/CRC/cache/地址边界、receipt schema 和 transport 回归。真实地址、寄存器和中断只可在读取 hardware lane 的
`board_interface_contract.json` 后接入。

必须等待：真实板端运行、正式全样本验证、延迟与功率测量；这些工作需要 route-clean bitstream、deployment binding、板卡 PL 可访问和对应 gate。

交接文件：

- `artifacts/zcu104_full_campaign_v1/handoffs/runtime_lane/handoff.json`；
- `artifacts/zcu104_full_campaign_v1/handoffs/runtime_lane/runtime_interface.json`；
- receipt schema、mock 测试、失败清单与 SHA256 索引。

## 5. Evidence lane：60 权重追溯与实验审计

允许修改：

- 新增的 ZCU104 checkpoint/预测/交接审计脚本；
- 新增的 ZCU104 非功率通用 schema；
- evidence lane 专属测试；
- `artifacts/zcu104_full_campaign_v1/handoffs/evidence_lane/` 和该 lane 的 `results/`、`logs/`。

禁止修改总控独占文件、公共文档、HLS/Vivado、runtime/transport、功率 backend 或其他 lane 的交接目录。

立即允许：只读审计四候选各 15 个 checkpoint 和 15 个 outer prediction；登记
`arch_id + architecture_sha256 + fold + seed + checkpoint_sha256 + prediction_sha256 + source path`；
核验 5 folds × seeds 42/43/44 的完整笛卡尔积、重复、缺失、文件大小和候选绑定。

该 lane 的本地审计 PASS 只形成交接 evidence bundle，不得直接把正式 60 权重门改为 PASS。总控合并后必须复验，且远端复制品仍需独立 SHA256 复核。

交接文件：

- `artifacts/zcu104_full_campaign_v1/handoffs/evidence_lane/handoff.json`；
- `artifacts/zcu104_full_campaign_v1/handoffs/evidence_lane/checkpoint_registry_60.json`；
- `artifacts/zcu104_full_campaign_v1/handoffs/evidence_lane/prediction_registry_60.json`；
- 审计报告、失败清单和复验命令。

## 6. 并行与等待边界

可以立即并行：

1. hardware lane 的代码/接口开发与短测试；
2. runtime lane 的 mock 和模拟 transport；
3. evidence lane 的 60 checkpoint/预测指纹审计。

必须串行或等待：

1. 公共协议、source freeze、lane base commit 和正式 ledger；
2. hardware 接口契约进入真实 runtime；
3. 完整 HLS、正式 RTL co-sim 和 Full Route；
4. 单块 ZCU104 编程、全样本板级分类和正式延迟；
5. 真实功率/温度测量；
6. 最终候选选择、gate 提升和结论撰写。

## 7. 合并与统一验收

总控一次只合并一个分支。每次先读交接文件、提交和测试证据，再检查：

1. 代码差异和文件所有权；
2. 目标测试；
3. 完整测试；
4. `git diff --check`；
5. source freeze 复验；
6. 失败、超时和资源超限是否被保留；
7. 正式状态是否仍保持证据分层。

推荐先合并 hardware 接口，再合并 runtime，evidence 可在不冲突时作为独立一次合并。局部测试 PASS 不等于 HLS、Route、板级分类、延迟或真实功率 gate PASS。
