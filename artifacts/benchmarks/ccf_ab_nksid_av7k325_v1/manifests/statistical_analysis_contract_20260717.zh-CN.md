# 统计分析合同审计（中文伴随档案，2026-07-17）

- 英文原件：`statistical_analysis_contract_20260717.md`；SHA256：`f618ec4f7e86f0298297a732615fd16e3810e223ab96c78e8a1d8d03303f106a`。

## 范围与结论边界

本卡审计 T2、T3、T4、T9 的预声明统计工具，不包含跨方法结果。只有被比较方法都具有相同 15 个可声明 fold-seed 单元，并通过独立文件与 provenance 审计后，才允许正式分析。

## 冻结比较单位与问题

- 分析单位：一个配对 `(outer_fold, seed)`；5 folds × seeds 42、43、44，每方法 `n=15`。
- 闭集主问题：四种闭集方法任意两者的 macro-F1 差异。
- 开放集问题：known-class macro-F1、NMA、OSFM、OSCRmac、unknown AUROC 与 FPR95 的配对差异。
- 鲁棒性问题：加性噪声和 speckle 两个实验族中，预声明 normalized F1-SNR AUC 的配对差异。clean、blur、contrast 单独标记，不静默并入 SNR。
- 差异方向序列化为 `left_minus_right`；lower-is-better 指标必须结合声明方向解释，不能只看正负号。

## 预声明推断

- 描述统计：保留全部 15 个值、`mean ± sample standard deviation`、样本数与 fold-seed 标识。
- 不确定性：10,000 次配对 bootstrap，在各 outer-fold stratum 内独立重采样后合并；seed 由实验族与比较标识确定性派生。
- 假设检验：双侧配对 sign-flip permutation test，以均值差为统计量；`n=15` 时枚举全部 `2^15=32,768` 个符号分配，因此是精确检验，不是 10,000 次近似。
- 效应量：配对差异 SD 非零时，Cohen's `dz = mean(pairwise difference) / SD(pairwise difference)`。
- 多重性：在声明实验族内使用 Holm step-down family-wise correction；原始和校正 p-value 都必须保留，不删除不显著比较。
- measurement-first ledger 允许发布前，分析行保持 `PENDING_G1_LEDGER`。

## 绑定与 fail-closed 规则

- Builder 按完整 `(fold, seed)` 集合为方法建键，推断前拒绝缺失或不配对单元。
- 闭集、开放集与 corruption builder 在读指标前，分别核验 method、run fingerprint、source freeze、当前 prediction/checkpoint SHA。
- summary-only、作者论文数值、smoke 和未配对 seed 不得进入正式配对分析。
- 统计显著性不能代替实际幅度；每个正式 contrast 必须同时报告均值差、95% CI、Cohen's dz 与 p-value。

## 已执行验证

- 命令：`D:\software\python\python.exe -m pytest -q -p no:cacheprovider tests/test_benchmark_statistics.py tests/test_benchmark_metrics.py`。
- 环境：`PYTHONDONTWRITEBYTECODE=1`，通过 `PYTHONPATH` 提供项目 `src`，未修改活动 CUDA 训练环境。
- 结果：`10 passed in 0.88s`。
- 覆盖：确定性分层配对 bootstrap、精确 15 单元 permutation、单调 Holm 校正、不配对数组拒绝、已知 exact-HV front、dominated/out-of-reference HV point、目标方向、Pareto coverage/NDCG、校准摘要与开放集未知分离。

## 源绑定

- `src/hwnas_fpga/benchmarks/statistics.py`：`f8c9b2487ac0fac4eddccba85637414a55b60c3a43fde9b72621d2d49c607630`。
- `tests/test_benchmark_statistics.py`：`4df99dca5eeea9d30c23eb9cd44caf9ddc09c3d1a1ddb44fdadbc9ebcd06cc14`。
- `tests/test_benchmark_metrics.py`：`708831021badc9591e8949d98988504c3ef383a76e8be41732403e6bc2cb3f6f`。
- `build_closed_classification_artifacts.txt`：`d761ba16cb430b8daf2c1b9de7fe38cee58c8b775529be041b2834f797cd24fc`。
- `build_open_set_artifacts.txt`：`0f202516ca10e1476ff3d1a411c7a22784795f4538dcaa0d0cfc0948ace08a48`。
- `build_sonar_robustness_artifacts.txt`：`d4e6c88646a2fecc7efd7ee1a0ba6c934251f8084a09faaa976c308b720e468d`。

## 剩余 Gate

统计工具合同已通过，但 T2/T3/T4/T9 统计尚不可用；仍需适用方法的完整 15 单元 cohort、独立审计和重新生成的 source-data 表。本卡不能单独授权任何 p-value 或效应量结论。
