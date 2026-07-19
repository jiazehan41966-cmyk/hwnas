# T3/F5 开放集图表合同（正式运行前预声明，中文伴随档案）

- 英文原件：`open_set_figure_contract_20260716.md`；SHA256：`2083d09a9ce9ecb0f52b636cc2c01d8090211b3d3dff857c9ee257e434b66ed0`。
- 核心结论边界：在相同冻结的 5-known/3-unknown manifest 下，CE+MSP、DMCL 和 PLUD 可能在已知类识别与未知类拒识上不同；45 个单元全部通过独立审计前不作方法排名。
- 图形类型：F5 为定量混淆矩阵网格，组合闭集分类语境与开放集检测证据。
- 输出：仅使用 Python/matplotlib；可编辑 SVG 为主，保留矢量 PDF 与 300 dpi PNG。
- 最终尺寸：180 mm × 150 mm，2 行 × 4 列。
- F5a–d：Scratch MobileNetV2、ImageNet 预训练 MobileNetV2、冻结 NAS 候选和 SURE 的行归一化 8×8 闭集混淆矩阵。
- F5e–g：CE+MSP、DMCL、PLUD 的行归一化 2×2 known-versus-unknown（已知/未知）检测矩阵；最后一个面板保留给公共色标与证据边界说明。
- 证据规则：开放集协议的五个已知类别随 outer fold 改变，因此不能跨 fold 合并“5 类 + unknown”的类别矩阵。F5 只能聚合 fold-invariant 的 known/unknown 判断；known-class macro-F1 与 OSCRmac 仍作为 T3/T9 的 fold-seed 指标。
- 统计：T3 均值使用 10,000 次 fold-stratified bootstrap；known macro-F1 与 OSCRmac 两两比较使用配对 fold-seed bootstrap、精确配对 sign-flip permutation，并在各指标实验族内作 Holm 校正。
- 源数据：每个 cell 的计数和归一化值都保留 method、task、true label、predicted label 与贡献样本数；所有输入审计、记录与预测 SHA 可追溯。
- 审稿风险：known/unknown 矩阵不能显示已知类之间的细分混淆，需由 T3 known macro-F1 补充；所有 fold 来自同一数据集总体。闭集和开放集输入均完整审计前，F5 必须保持不可用。

## 2026-07-17 冻结输入绑定

- 样本 manifest：`artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/manifests/open_long_tail_sample_manifest_v1.json.txt`。
- Manifest SHA256：`59878d48786129c983e976b1cf8f4fc03bda79bd9e05ec5671ab42dedc1f7a3e`。
- 独立审计：`open_long_tail_sample_manifest_v1_audit.json.txt`，SHA256 `63043398d989e10da319ab1a70bafa8204651e279bbf9cce28269ef1eb5f759f`。
- 审计结果：PASS；重新核验 2,617/2,617 个样本哈希、重建 15/15 个 fold-seed 成员关系、得到五组不同的 fold-specific unknown classes，错误为 0。
- 准入：CE+MSP、DMCL、PLUD 正式运行必须在 canonical integration 后绑定该 manifest SHA。本次输入冻结贡献 0 个正式结果单元，不使 T3/F5 可用。
