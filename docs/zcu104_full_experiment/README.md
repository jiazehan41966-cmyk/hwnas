# ZCU104 完整实验文档包

版本：v1.1（集成总控基线）

日期：2026-08-10

适用项目：HW-NAS / FPGA / 声呐图像分类

目标板卡：AMD ZCU104（XCZU7EV）

## 1. 文档目的

本目录给出从当前四阶段声呐算子闭环迁移到 ZCU104，并完成分类、INT8、HLS、Full Route、板级全样本验证、延迟、功率、能耗、代价模型校准和硬件感知搜索反馈的完整执行方案。

这些文件是实验协议和执行 prompt，不是已经完成的实验结果。任何未通过的阶段必须保持 `PENDING`、`FAIL`、`TIMEOUT` 或 `NOT_MEASURED`，不得用历史 AV7K325 结果、Vivado 功率估计或 NAS 代理值补齐。

## 2. 当前基线

- 当前主工作树：`E:\1\hwnas\hwnas`
- 当前主分支：`codex/g5-sonar-complete-experiment`
- 主工作树存在大量用户未提交修改，只作为文档和历史证据输入，禁止在其中实验、提交、清理或回退。
- 四阶段声呐算子正式闭环位于：`codex/sonar-fourstage-operator-closure`
- 唯一 ZCU104 集成工作树：`E:\1\hwnas\hwnas-zcu104-full-campaign-v1`
- 唯一 ZCU104 集成分支：`codex/zcu104-full-campaign-v1`
- 该分支最初基于已推送提交 `cae9be4f01d1a55576bc4449b6b41680b4a889d9` 建立；后续并行线只能从总控复验通过并明确登记的 lane base commit 创建。
- Protocol V2 已支持成熟 MBConv-k5；Dir-v1 已按预注册精度门判定为不准入。
- 当前完整网络 external-scratch HLS 已有基础证据，但 RTL co-sim 超时，Full Route、ZCU104 bitstream、板级全样本分类和外部功率仍未闭环。
- 当前集成配置只绑定四个 `fold0/seed42` 代表 checkpoint；源结果目录虽已发现四候选各 15 个 checkpoint 和预测文件，但 60 个 checkpoint 的正式 SHA256 登记与可复验清单仍为 `PENDING`。
- 服务器最近一次留存的实际探测为 Vivado `2022.2`、Vitis HLS `2022.2`、PyTorch `2.7.1+cu128`；本批次不得把文档建议的 `2023.2` 写成已安装版本。2026-08-10 实时 SSH 复核因连接超时未完成，需在执行线启动前重验。
- 服务器连接命令为 `ssh -p 7889 jiajia@10.70.23.121`。认证口令不得写入配置、脚本、日志或本文档。

并行开发固定为三条执行线：硬件实现、板端 runtime（运行程序）和证据审计。分支、工作树、文件所有权、输入提交和交接文件以
`artifacts/zcu104_full_campaign_v1/control/parallel_work_registry.json` 为唯一登记源。执行线不得直接提升正式 gate 或修改总控拥有的正式配置、状态、总账和最终报告。

## 3. 文件索引

| 文件 | 内容 |
|---|---|
| [MODIFICATION_PLAN.md](MODIFICATION_PLAN.md) | 完整代码、硬件、runtime、功率与证据链修改方案 |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | 候选、对照、阶段、重复、统计、停止门和最终闭环方案 |
| [METRICS_AND_ACCEPTANCE.md](METRICS_AND_ACCEPTANCE.md) | 所有必测指标、公式、来源、聚合方式与验收门 |
| [SONAR_AND_THERMAL_METRICS.md](SONAR_AND_THERMAL_METRICS.md) | 声呐鲁棒双终点、退化协议、结构指标边界与 ZCU104 测温方法 |
| [SERIAL_EXECUTION_PROMPTS.md](SERIAL_EXECUTION_PROMPTS.md) | 按依赖顺序逐步执行的完整 prompts |
| [PARALLEL_WORK_PROMPTS.md](PARALLEL_WORK_PROMPTS.md) | 可并行工作包及可直接派发的 prompts |

## 4. 已冻结的第一阶段部署候选

| 编号 | 角色 | `arch_id` | 已有软件 Macro-F1 均值 | 当前硬件状态 |
|---|---|---|---:|---|
| C0 | 原始成熟基线 | `fourstage_s2_k3_e3_s4_mbconv_k3_e3` | 0.815649 | 待 ZCU104 闭环 |
| C1 | Stage2-K5 代表 | `fourstage_s2_k5_e6_s4_mbconv_k3_e3` | 0.836594 | 待 ZCU104 闭环 |
| C2 | 精度优先 | `fourstage_s2_k5_e3_s4_mbconv_k5_e3` | 0.863563 | 待 ZCU104 闭环 |
| C3 | 低成本 Skip | `fourstage_s2_k5_e3_s4_skip` | 0.798978 | 待 ZCU104 闭环 |

上述精度来自已完成的 Protocol V2 软件实验，只用于冻结候选角色，不得写成 ZCU104 板级准确率。

## 5. 核心技术决策

1. ZCU104 正式部署采用 PS–PL 架构：PS DDR、AXI DMA、PL accelerator、板端固定版本 runtime。
2. PL-UART 只用于 bring-up、CRC、状态码和固定向量调试，不作为正式系统端到端延迟主线。
3. 完整网络优先沿 external-scratch / tiled / shared compute 路线推进；当前资源严重超限的 HWC 全并行实现不作为正式基线。
4. 主工作频率预注册为 200 MHz；100/150 MHz 是独立频率敏感性工作点，不得与 200 MHz 结果混写。
5. 正式功率主证据为板卡直流输入端 `board_input_total`；PMBus 轨级数据只作为辅助诊断。
6. 每个联合工作点必须绑定同一候选、checkpoint、INT8、HLS、bitstream、runtime、板卡和功率记录。
7. ZCU104 代价模型校准通过前，功率和未校准硬件代理不得进入正式 NAS 硬剪枝。

## 6. 执行顺序

```text
冻结代码和协议
  -> 修复资源可行的完整网络数据流
  -> INT8 / C-sim / RTL 验证
  -> ZCU104 HLS / Full Route
  -> 板级全样本数值一致性
  -> 延迟与吞吐
  -> 外部功率与能耗
  -> ZCU104 代价模型校准
  -> 正式 HW-NAS
  -> 最终三个 Pareto 点完整复测
```

## 7. 完成定义

只有同时满足以下条件，才能称为“ZCU104 完整实验完成”：

1. 源码、协议、数据、候选和 checkpoint 完成 SHA256 冻结；
2. INT8 精度门与完整网络数值一致性通过；
3. ZCU104 exact-part Full Route 时序和资源通过；
4. 至少四个冻结部署候选完成板级全样本验证；
5. 板级无缺失样本、无 CRC 错误、无 INT8 logits/argmax 不明失配；
6. kernel、DMA、runtime 和 system E2E 延迟分开测量；
7. 至少三个候选完成同一仪器、同一协议的外部功率测量；
8. 同时报告整板总能耗和扣除 idle 后的动态能耗；
9. ZCU104 硬件代理具有独立 holdout 校准结果；
10. 新搜索产生的最终 Pareto 候选重新执行完整板级与功率链路；
11. 所有失败、超时和资源不可行结果均被保留；
12. 若声称适应强噪声和低对比度，冻结的 `sonar_robust_macro_f1`、各退化曲线和最差条件指标已完整报告；
13. ZCU104 PS/PL 结温与功率、延迟日志时间对齐，环境温度未测时明确标记 `NOT_MEASURED`；
14. 正式 evidence ledger 通过，而不仅是代码测试通过。
