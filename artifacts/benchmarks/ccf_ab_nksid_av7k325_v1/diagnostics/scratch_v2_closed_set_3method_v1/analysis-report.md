# scratch-v2 三方法严格配对分析

## 分析问题

在完全相同的 5 outer folds × seeds 42–44（15 个配对单元）上，比较从零训练 MobileNetV2、ImageNet 预训练 MobileNetV2 与冻结 NAS 候选 `rl_arch_135` 的 macro_f1（宏平均 F1，越高越好）。旧 scratch 仅用于检验 scratch-v2 的数值复现，不作为独立统计方法。

## 关键发现

- scratch-v2：`0.909902 ± 0.024014`（SD，n=15）。
- 预训练 MobileNetV2：`0.931632 ± 0.024188`（SD，n=15）。
- 冻结 NAS：`0.693619 ± 0.029237`（SD，n=15）。
- scratch-v2 与旧 scratch 的 15 个 macro_f1 逐项完全一致；这证明数值复现，但不修复旧 patch provenance。
- 预训练相对 scratch-v2 的配对均值差为 `0.021730`，95% CI `[0.010925, 0.032833]`。
- scratch-v2 相对 NAS 的配对均值差为 `0.216283`，95% CI `[0.202079, 0.229091]`。

## 证据解释

观察层面，预训练 MobileNetV2 的平均 macro_f1 高于 scratch-v2，而两者都显著高于当前冻结 NAS 候选。统计支持来自 15 个相同 fold-seed 单元的配对差、按 fold 分层的 10,000 次 bootstrap CI、精确符号置换和 Holm 校正。该结果改变的决策是：`rl_arch_135` 可以继续作为硬件可行性与精度—资源权衡的负结果，但不能描述为精度竞争方法。

## 限制

SURE 尚未完成 15 单元，因此本分析不是完整 T2，也不能发布四方法最终排序或完整实验族的最终 Holm 校正。NKSID 缺少采集任务分组元数据，当前结论不支持 group-safe 泛化。板级准确率和功耗没有由本分析测量。
