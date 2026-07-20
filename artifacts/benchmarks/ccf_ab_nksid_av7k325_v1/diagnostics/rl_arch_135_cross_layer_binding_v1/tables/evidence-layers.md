| 证据层 | 状态 | 可支持结论 | 禁止结论 |
|---|---|---|---|
| 冻结架构身份 | PASS | 软件与硬件使用同一候选架构编码 | 不能由架构相同推断权重相同 |
| 软件分类 | PASS_15_OF_15 | 冻结架构的 NKSID 软件 macro-F1 | 不能证明 NAS 方法无偏泛化 |
| 权重绑定 | NOT_BOUND | 无 | 不能声称 formal checkpoint 已部署 |
| Vivado route | PASS | 该架构映射可 route 且资源/时序已记录 | 不能证明软件数值等价 |
| COM5 延迟 | PASS_LATENCY_ONLY | 确定性 harness bitstream 的板级端到端延迟 | 不能作为 NKSID 板级准确率 |
| 板级分类准确率 | NOT_RUN | 无 | 不能把 PyTorch 精度写成板级精度 |
| 功耗 | NOT_MEASURED | 搜索 proxy 与 Vivado estimate 仅作诊断 | 不能报告动态功耗或能耗实测 |
