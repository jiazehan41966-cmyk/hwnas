# 图片目录

## figure-01-three-method-paired-macro-f1

- 目的：展示三个方法在相同 15 个 fold-seed 单元上的配对结构。
- 数据源：`paired_units_source.csv`。
- 读者应注意：NAS 在所有单元低于两种 MobileNetV2；预训练与从零训练之间存在较小、非完全一致的差异。
- 决策意义：冻结 NAS 候选应按硬件可行但精度较弱的结果解释。
- 限制：纵轴放大至 0.55–1.00；SURE 尚未加入。

## figure-02-paired-contrast-forest

- 目的：展示三个预声明对比的均值差及 95% 分层 bootstrap CI。
- 数据源：`paired_contrasts.csv`。
- 读者应注意：涉及 NAS 的差异远大于预训练相对 scratch-v2 的差异。
- 决策意义：后续候选改进应优先解决 NAS 精度缺口，同时保留硬件证据边界。
- 限制：完整 T2 加入 SURE 后需重新计算实验族校正。
