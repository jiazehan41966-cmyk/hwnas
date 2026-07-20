# NAS 四方法实现缺口与修订（中文伴随档案，2026-07-17）

- 英文原件：`nas_four_method_implementation_gap_20260717.md`；SHA256：`d4c10f0c5359e22156a12696ae6dab387d6651580e2cc26ee7576c88b0da9cdc`。

## 决定

T5 不得从当前双方法 launcher 启动。正式目标仍是 Random、RL、Aging Evolution、HW-PR-NAS paper-spec adapter；每 seed 42–51 各 300 次实际 evaluator call。缺失实现完成后、任何正式运行前必须新建 source freeze。

第四方法必须标为 `hw_pr_nas_paper_spec_low_data_adapter`，不能称为作者复现。官方仓库没有可执行作者 release，项目 300-call 预算也低于论文对新搜索空间所述数据期望。

## 当前覆盖

- Random/RL/Aging 单方法实现存在；HW-PR 只有 `benchmarks/hwpr.py` 中 Pareto-rank/ListMLE smoke helper，没有 acquisition loop。
- `run_aging_vs_rl_benchmark.py` 只跑 RL/Aging，默认预算 200；`compare_search_methods.py` 虽能聚合任意方法，但推断硬编码为 Aging−RL。
- 正式 `formal/search_comparison/comparison.json` 不存在。

## 作者源码审计与论文支持边界

官方 commit `296c6576...` 只有 main/add-license-1/v0.1.0；README 所述 surrogate/predictor 文件缺失，`search_algo.py` 调未定义 `valid_loss()` 且是单目标 tournament，`test.py` 调用不兼容入口。本地 helper 只实现 Pareto-rank target、listwise ordering、tabular encoding 与三层 MLP，不复现 architecture features+GCN+LSTM encoder。

论文描述统一 predictor、nondominated sorting 的 Pareto rank、三层 FC、listwise loss、batch 18，并指出新搜索空间通常至少需 500 个 ground-truth architecture。Primary DOI：`10.1145/3579853`。

## 等预算修订

1. 任何为新架构产生 `f_clean/f_robust/latency` 的调用计入同一 300 budget；rejected/duplicate/failed 分列。
2. Pilot：18 个初始评估 + 32 个 surrogate-guided；formal：50 初始 + 250 guided。
3. 目标严格为 `1-f_clean`、`1-f_robust`、`latency/limit`，LUT/DSP/BRAM 为硬约束。
4. Pareto-rank target 只能由已评估架构拟合；未评估候选可打分，不能贡献 truth 或 HV。
5. 初始设计与 proposal stream 确定且 hash-recorded；离散空间 Latin-hypercube adaptation 和 seed 必须归档。
6. 序列化训练行、predicted rank/score、Kendall tau、NDCG@k、top-k recall、update cadence、rejection、wall/GPU。
7. 每个 T5 行和 F2/F3 meta 标记 `paper_spec_reimplementation`、`author_runtime_ready=false`、`low_data_adaptation=true`、`author_code_numerical_result=false`。
8. 禁止与作者 FBNet/NAS-Bench 数值跨协议排名；只比较本项目统一协议下四方法。

## 新 source freeze 下所需代码

实现可 resume HW-PR searcher；launcher 泛化四方法并按 seed counterbalance、强制 300×10；推断扩展为 6 个 method pair 并做 Holm/效应量/CI；测试预算记账、未评估泄漏、resume equivalence、deterministic acquisition、全 pair Holm 与 fail-closed；先 10-call smoke，再 50-call×3 pilot，用户批准后才可在 G3 允许时运行 300-call×10 formal。

当前 T5/F2/F3 PENDING，G3 FROZEN；本卡只授权实现规划，不授权 NAS 性能实验。
