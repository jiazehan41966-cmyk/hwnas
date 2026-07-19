# rl_arch_135 软件—硬件跨层绑定分析

## 分析问题

本分析回答：完成 15/15 软件评测的 `rl_arch_135`，是否能与历史 Vivado route 和 COM5 记录可靠绑定，以及这些证据允许支持哪些结论。

## 结论

架构身份绑定通过。正式软件运行、harness manifest 和 harness spec 均绑定 candidate SHA256 `3e0f0d943ca31f659ee545a04d47bac985e6eebe508a3a1ba2d41460c818543a`；候选 encoding signature 重算为 `b58ae648c3d1bde1c9651dec6b45b447a56daadfd77ad951dc1752b71a7afb3b`，与最终硬件报告一致。

权重身份不通过。历史硬件 parameter manifest 明确记录 `parameter_mode=latency_only_deterministic`、`accuracy_claim=none`，没有绑定本次 15/15 的任何 checkpoint。因此可以把软件精度与硬件 route/延迟并列为“同一架构的不同证据层”，但不能声称正式软件权重已经部署到该 bitstream。

## 核心结果

- 软件：15 个 fold-seed 单元，NAS macro-F1 `0.693619 ± 0.029237`；预训练 MobileNetV2 `0.931632 ± 0.024188`。
- 配对差值：`-0.238013`，按 fold 分层的 10,000 次 bootstrap 95% CI `[-0.263968, -0.212667]`，NAS 胜出 `0/15`。
- route：PASS，WNS `0.094 ns`；LUT 18,598，FF 21,012，BRAM 127.5，DSP 612。
- COM5：5 次确定性 harness 测量均为 `9812402` cycles、`49.062010 ms`、checksum `33551839`。
- 搜索期延迟估计为 `7.286775 ms`，板级实测为 `49.062010 ms`，相差 `6.73×`。该比例仅描述这一候选，不能直接推广。
- 板级分类准确率：NOT_RUN。功耗：NOT_MEASURED。

## 证据质量问题

旧 measurement JSON 的 `vivado_hardware.stdout` 含 replacement character；受影响路径为 `$.vivado_hardware.stdout`。核心 frame、原始 CSV、bitstream SHA、route 报告和五次稳定性数值不依赖该文本字段，因此延迟证据仍可读取；但该编码缺陷必须保留在限制中，不能静默清洗旧证据。

## 决策含义

当前候选适合作为“架构可 route、板级延迟可测，但精度明显不足”的负结果。若要建立同一权重实例的端到端 FPGA 结论，下一阶段必须由用户另行批准并完成：固定一个 formal checkpoint、权重导出 SHA、PyTorch—INT8—HLS 数值等价、重新生成 bitstream、NKSID 板级分类准确率、延迟和外部仪器功耗。当前分析不授权这些动作。

## 证据边界表

| 证据层 | 状态 | 可支持结论 | 禁止结论 |
|---|---|---|---|
| 冻结架构身份 | PASS | 软件与硬件使用同一候选架构编码 | 不能由架构相同推断权重相同 |
| 软件分类 | PASS_15_OF_15 | 冻结架构的 NKSID 软件 macro-F1 | 不能证明 NAS 方法无偏泛化 |
| 权重绑定 | NOT_BOUND | 无 | 不能声称 formal checkpoint 已部署 |
| Vivado route | PASS | 该架构映射可 route 且资源/时序已记录 | 不能证明软件数值等价 |
| COM5 延迟 | PASS_LATENCY_ONLY | 确定性 harness bitstream 的板级端到端延迟 | 不能作为 NKSID 板级准确率 |
| 板级分类准确率 | NOT_RUN | 无 | 不能把 PyTorch 精度写成板级精度 |
| 功耗 | NOT_MEASURED | 搜索 proxy 与 Vivado estimate 仅作诊断 | 不能报告动态功耗或能耗实测 |
