# 图表目录

## figure-01-software-macro-f1

- 目的：展示 15 个完全配对的 fold-seed 软件分类结果。
- 数据源：`figures/figure-01-software-macro-f1_source.csv`。
- 读者应注意：所有配对线均从 NAS 指向更高的预训练基线。
- 解释：冻结架构的软件精度落后是跨 fold-seed 一致的，而非由单一 seed 驱动。
- 决策影响：保留该候选作为硬件可行负结果，不作为精度冠军。
- 限制：不是无偏 NAS 方法泛化证据。

## figure-02-hardware-evidence

- 目的：显示 route 后资源占用，并把搜索期延迟 proxy 与 COM5 板级延迟明确分层。
- 数据源：`figures/figure-02-hardware-evidence_source.csv`。
- 读者应注意：DSP 利用率最高；COM5 延迟明显高于搜索 proxy。
- 解释：该架构可以 route，但搜索 proxy 不能替代板级延迟。
- 决策影响：未来搜索需基于更多 route/COM5 样本校准；单候选比例不能硬编码为修正系数。
- 限制：bitstream 使用确定性 latency-only 参数，无板级准确率和功耗。
