# 全对标计划完成度与证据冲突审计

- 生成时间：2026-07-18T08:15:39+08:00。
- 有效总状态：`IN_PROGRESS_WITH_G1_CONFLICT_HOLD`。
- 正式证据增量：`0`；本文件只审计现有证据，不启动训练、HLS、Vivado、板卡或功耗实验。

## 结论

五篇主对标论文及六套环境已经锁定，但五篇外部方法均没有完成本项目统一协议下的正式重评。当前最接近完成的是闭集 G1：45 个软件产物都存在，预训练与 NAS 的 30 个单元通过专项独立审计；scratch 的 15 个单元因 `code_patch.diff` 被后续调用覆盖而出现 provenance 冲突。故即使测量优先总账显示 G1 PASS，本覆盖审计仍按 fail-closed 原则冻结为 `G1_CONFLICT_HOLD`。

```mermaid
flowchart LR
  A["R1 论文审计 PASS"] --> B["R2 环境锁定 PASS"]
  B --> C["R3 闭集 G1 冲突 HOLD"]
  C --> D["R4 开放集 未正式运行"]
  C --> E["R5 NAS 正式搜索 FROZEN"]
  E --> F["R6 HLS 真值 0/100"]
  F --> G["R7 板级与功耗 未正式测量"]
  D --> H["R9 正式归档 1/9 表、1/12 图"]
  G --> H
```

## 九项工作包矩阵

| 编号 | 工作包 | 有效状态 | 完成度 | 证据强度 | 下一准入条件 |
|---|---|---|---|---|---|
| R1 | 五篇主论文源码、版本、许可证与对应关系 | `PASS` | 5/5 审计卡 | 中等 | 保持固定 commit；缺许可证代码继续隔离 |
| R2 | 独立环境与依赖锁定 | `PASS` | 6/6 环境卡 | 中等 | 正式运行前再次核验解释器、CUDA、PyTorch 与锁文件 SHA |
| R3 | 闭集分类、校准与成本 | `G1_CONFLICT_HOLD` | 45/45 产物；30/45 独立审计通过；SURE 0/15 | 冲突，暂不定级 | 用户批准 provenance 修复、新源码冻结及 scratch-v2；另行批准 SURE 15 单元 |
| R4 | 开放集长尾声呐识别 | `FORMAL_NOT_STARTED` | 接口冒烟通过；CE+MSP/DMCL/PLUD 均 0/15 | 仅工程可行性 | 用户批准后按冻结 class-holdout manifest 运行三方法 15 单元 |
| R5 | 四种 NAS 方法与 exact HV | `FROZEN` | 正式 300 评估 × 10 paired seeds：0/40 方法种子单元 | 仅实现/契约 | Gate0、G2、G4 与 stage-3 用户批准全部闭合 |
| R6 | HLS/route 代理可靠性 | `TRUTH_NOT_COLLECTED` | 结构池 100/100；正式真值 0/100；先导 5/5 仅冻结输入 | 设计阶段 | 先补候选特异 operator 映射与完整网络生成器，再批准 HLS 采集 |
| R7 | AV7K325 route、板级延迟与外部功耗 | `NOT_MEASURED_FORMAL` | 正式候选 0/3；外部功率仪器 0/3 | 历史描述性证据 | 用户批准 checkpoint—INT8—HLS—bitstream 重建及板卡/功率仪器实验 |
| R8 | 测量优先门禁 | `IN_PROGRESS_WITH_G1_CONFLICT` | G0 PASS；G1 总账 PASS 但覆盖审计 HOLD；G2/G4 PENDING；G3 FROZEN；G5 PAUSED；功耗 NOT_MEASURED | 保守门禁 | 先解决 G1 审计冲突，再按 gate 顺序闭合 |
| R9 | 正式表格与图片归档 | `PARTIAL` | 正式表 1/9；正式图 1/12 | 可追溯但不完整 | 各实验族闭合后由原始逐样本/逐候选/时间序列数据生成 |

## G1 冲突的精确边界

- 测量优先总账：`PASS`，45/45。
- 45 单元专项审计：`FAIL`；产物 45/45，独立审计通过 30/45。
- scratch：`FAIL`，错误为 `tracked code patch hash mismatch`。
- 预训练：`PASS`；NAS：`PASS`。
- 冲突原因：总账只接受 protocol summary 的自报 claimability；专项审计还验证 manifest 绑定的 patch 文件 SHA。后者更严格，因此用于覆盖声明。
- 禁止结论：在冲突消除前，不得写成“G1 已完全通过”或把 scratch 纳入正式 T2 显著性比较。

## 五篇主论文现状

| 论文编号 | 方向 | 固定提交 | 许可证 | 源码冒烟 | 论文代码对应 | 本地正式结果 |
|---|---|---|---|---|---|---|
| hw_pr_nas_2023 | nas_pareto_surrogate | `296c6576fbae2b277e56c704ff3b6e648ec4c2be` | MIT | `BLOCKED_OFFICIAL_CODE_INCOMPLETE` | partial | 尚无 |
| sure_2024 | classification_robustness_calibration | `5ce0193bc93e73b1c7f1f53aeda8854e997011e2` | 缺失/未核验 | `PASS` | verified | 尚无 |
| harp_2023 | hls_proxy_reliability | `c8bffd9411917b125846429b4d6be4f21c7a7165` | BSD-3-Clause | `PASS` | verified | 尚无 |
| esda_2024 | fpga_artifact_power_chain | `b75c8c93ca258158c06a6434f5f0f084add02ee5` | MIT | `PASS` | verified | 尚无 |
| dmcl_sonar_oltr_2025 | sonar_open_long_tail | `eea8dc07ce007988150ac208cd09e00daedba2ca` | 缺失/未核验 | `PASS_SOURCE_PRESENT` | partial | 尚无 |

## 科学有效性风险

- 选择偏差：NAS 候选来自历史 fold0 选择，不能由 15 单元重评反推搜索策略无偏泛化。
- 构念混淆：搜索期延迟/功耗代理不得写成板级实测；历史 COM5 latency-only harness 不等于正式分类权重部署。
- 间接性：HW-PR、HARP 与 ESDA 均为 B/C 类迁移或证据链参考，作者原始数字不能与本项目 AV7K325/NKSID 直接排名。
- 不完整性：开放集、SNR 鲁棒性、HLS 真值、板级分类准确率和外部功耗尚无正式结果。
- 多重比较：T2/T3/T5/T6/T9 完整实验族形成后，才执行预先约定的 Holm 校正；当前诊断比较不替代完整实验族分析。

## 无需用户批准即可继续的工作

- 保持论文 commit、许可证卡、环境锁文件和中文档案审计有效。
- 完善只读审计、schema、图表生成器和不触发训练/硬件的单元测试。
- 保留 NAS 负结果及历史硬件证据的分层边界。

## 必须由用户批准的关键动作

1. 修改正式评测入口的 provenance 捕获逻辑、建立新源码冻结并决定是否 fresh scratch-v2 15 单元重跑。
2. 启动 SURE、CE+MSP、DMCL 或 PLUD 的正式 5 folds × 3 seeds 训练。
3. 启动四种 NAS 方法的 pilot/formal 搜索，或改变当前冻结候选与协议。
4. 启动完整网络 HLS/route 真值采集、板卡 COM5、bitstream 重建或外部功率仪器测量。

## 当前建议

优先请求用户批准第 1 项：先修复 provenance 覆盖缺陷并重新冻结，再决定是否重跑 scratch-v2。该步骤不改善 NAS 精度，但能消除总账与专项审计的逻辑矛盾；在此之前不建议启动板级正式实验。
