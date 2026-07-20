# T4/F7/F8 声呐鲁棒性图表合同（正式 corruption 评测前预声明，中文伴随档案）

- 英文原件：`sonar_robustness_figure_contract_20260716.md`；SHA256：`806b910acfe13a97e9333c09e34f606e1704382829b4ac6541f52304ef7bd282`。
- 核心结论边界：在相同 NKSID outer folds 和确定性图像域 corruption（扰动）下，四个已审计闭集方法的 macro-F1 下降速度与不确定性可能不同；归档数据存在前不声明优胜方法。
- 图形类型：F7 是以 SNR 响应为主证据的双面板定量图；F8 是带追溯 meta 的图像样例板，只提供定性语境，不作为性能证据。
- 输出：使用 Python/matplotlib，面向 Nature/高水平期刊；可编辑 SVG 为主，保留矢量 PDF 和 300 dpi PNG。
- 最终尺寸：F7 为 180 mm × 82 mm；F8 为 180 mm × 205 mm，8 个类别行 × 7 个冻结条件列。
- F7a：AWGN 在 0、5、10、15、20 dB 下的 macro-F1；对 15 个 fold-seed 单元使用 10,000 次 fold-stratified bootstrap 95% CI。
- F7b：speckle 使用相同 SNR 网格和推断规则。
- F8：每个 NKSID 类别中全局 sample index 最小的样本；列依次为 clean、AWGN 10 dB、AWGN 0 dB、speckle 10 dB、speckle 0 dB、blur severity 5、contrast severity 5。
- 证据层级：F7 的配对方法曲线和不确定性是主证据；T4/T9 的逐单元 F1-SNR AUC、worst-case macro-F1 和相对 clean 下降是验证证据；F8 只用于目视核对 corruption 严重度与哈希追溯。
- 统计：10,000 次按 fold 分层的配对 bootstrap；15 个单元的精确配对 sign-flip permutation；AWGN 和 speckle 实验族内 Holm 校正；效应量为配对差异的 Cohen's dz。
- 源数据：每幅图使用一个组合 source CSV；逐样本预测、实际达到的 SNR、clipping/saturation ratio、checkpoint SHA 与变换 seed 保留在正式鲁棒性结果根目录。
- 图像完整性：F8 只允许确定性全图变换，禁止局部修饰、裁剪、按类别调节或根据模型结果后选；保留原图与渲染图 SHA256。
- 审稿风险：只有 15 个配对 fold-seed 单元，fold 共享同一数据集总体；置信带不能外推为独立采集站点结论。数值 F1-SNR AUC 排除 clean。PSNR/SSIM 仅因 corruption 存在精确配对的 resized-clean 输入而可用，不能证明真实场景去散斑质量。

本中文伴随档案不使 T4/F7/F8 可用；正式 corruption 评测仍受 G1 完成与用户决策门禁约束。
