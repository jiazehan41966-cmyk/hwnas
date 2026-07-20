# T2/F6 闭集分类图表合同（四方法完成前预声明，中文伴随档案）

- 英文原件：`closed_classification_figure_contract_20260716.md`；SHA256：`4ce6cb7d57bd6a97b4677408f892e2369e9f3b67fda84fe42b4b403d737d63a0`。
- 核心结论边界：在相同 NKSID 5 folds × 3 seeds 协议下，四个已审计闭集方法可能在 macro-F1、校准和失败排序质量上存在差异；60 个单元及全部审计完成前不声明优胜方法。
- 图形类型：F6 为双面板定量图；risk-coverage curve（风险—覆盖率曲线）是主证据，reliability diagram（可靠性图）用于校准验证。
- 输出：仅使用 Python/matplotlib；可编辑 SVG 为主，保留矢量 PDF 和 300 dpi PNG。
- 最终尺寸：180 mm × 82 mm。
- F6a：四方法各自的 15-bin pooled reliability curve（15 箱合并可靠性曲线）及 identity line（理想对角线）；精确箱计数保存在 source CSV。
- F6b：在共同 1%–100% coverage 网格上，对 15 个 fold-seed 单元求平均 risk-coverage curve，并使用 10,000 次 fold-stratified bootstrap 形成 95% CI 带。
- 统计：T2 对 15 个协议单元使用按 fold 分层的 bootstrap 95% CI；macro-F1 两两差异使用相同配对 bootstrap、精确配对 sign-flip permutation（符号翻转置换）及 T2 macro-F1 实验族内 Holm 校正。
- 源数据：所有指标从归档的逐样本 logits 与 targets 重新计算；保留每个 checkpoint、预测文件、运行记录和独立审计的 SHA256。
- 计算边界：checkpoint 字节数和 model-state tensor elements（模型状态张量元素数）从 checkpoint 文件直接测量。训练 wall-clock/GPU-hours 没有可信逐单元计时，必须记为 `NOT_RECORDED`，不得从文件时间戳反推。
- 审稿风险：同一 outer fold 的三个 seed 共享图像，五个 fold 来自同一数据集总体；pooled reliability bins 只作视觉摘要，推断仍以配对 fold-seed 单元为单位。闭集与开放集混淆证据均完整前，F5 不得发布。

本中文伴随档案不使 T2/F6 可用；当前仍等待四方法 60/60 及独立审计。
