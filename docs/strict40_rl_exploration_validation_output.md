# Strict40 RL Exploration 完整验证结果输出

日期：2026-05-10
实验：`formal_lut_strict40_nksid_rl50_eval5_explore_v1`
配置：`configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml`
结果目录：`results/formal_lut_strict40_nksid_rl50_eval5_explore_v1`
后处理汇总：`results/strict40_rl50_eval5_explore_final_analysis.json`

## 0. 汇报边界

这轮实验的目的不是证明最终 NAS 架构性能，而是验证 RL controller 的 exploration 是否从 collapse 恢复，并确认 strict40 LUT 搜索链路仍然稳定。

需要强调：

- `eval_epochs=5` 是 proxy，不是最终训练精度。
- 只有单次 run，没有多 seed 独立重复，因此不做显著性结论。
- 当前 strict40 搜索空间只有 4 个候选，因此这轮更适合证明 controller 能探索，而不是证明 RL 搜索能力已经充分。

## 1. 可讲结论：实验已经完整跑完，没有执行错误

佐证数据：

| 项目 | 结果 |
|---|---:|
| run status | completed |
| planned episodes | 50 |
| evaluated records | 50 |
| stdout | 包含 `RL NAS completed!` |
| stderr | 0 bytes |
| traceback | 无 |

可讲口径：

> 修复 exploration 后的 50-episode strict40 NKSID 验证已经完整跑完，run 状态为 completed，stderr 为空，没有 traceback 或失败阶段。

## 2. 可讲结论：RL exploration collapse 已经修复

修复前后对比：

| 方法 | episodes / candidates | feasible | unique candidates | 覆盖结论 |
|---|---:|---:|---:|---|
| 原始 RL baseline | 50 | 50/50 | 1/4 | collapse，只采到 1 个候选 |
| Random baseline | 50 | 50/50 | 4/4 | 搜索空间本身可覆盖 |
| 修复后 RL | 50 | 50/50 | 4/4 | exploration 恢复 |

修复后 RL 的候选覆盖：

| candidate | count | proportion |
|---|---:|---:|
| mbconv k3/e3 | 20 | 40% |
| mbconv k3/e6 | 12 | 24% |
| mbconv k5/e3 | 13 | 26% |
| mbconv k5/e6 | 5 | 10% |

可讲口径：

> 原始 RL 在 50 episodes 中只采到 1 个候选；修复后 50 episodes 覆盖了全部 4 个 strict-covered 候选。这个对比说明问题不是 LUT 或约束导致的不可行，而是 controller exploration；temperature、entropy、epsilon exploration 和 exploration bonus 已经把 collapse 拉回来了。

## 3. 可讲结论：strict40 LUT 搜索链路保持稳定

strict40 LUT 来源：

| 项目 | 数值 |
|---|---:|
| measured entries | 40 |
| defer_current_impl | 44 |
| board_result_status_overrides | 0 |
| status_authoritative | true |

已验证的 LUT integrity：

| 验证项 | 数据 |
|---|---:|
| deterministic 4-candidate smoke | 4/4 feasible |
| 每个候选 LUT hit/miss | 5/0 |
| deterministic true_miss | 0 |
| deterministic deferred_hit | 0 |
| resume 段 LUT hits | 25 |
| resume 段 LUT misses | 0 |
| resume 段 true_miss | 0 |
| resume 段 deferred_hit | 0 |

说明：

`lut_stats.json` 在 resume 后只记录最后 5 个 episode，因此 final 文件显示 `hits=25`。全量 50 个 episode 采样的都是 deterministic smoke 已验证的 4 个候选；每个候选 cost 查询为 5 个 strict LUT hits、0 miss。因此可以讲 strict40 覆盖的候选集合没有 true_miss/deferred_hit。

可讲口径：

> strict40 LUT 链路没有暴露 true_miss 或 deferred_hit。4 个候选已经通过确定性枚举验证，每个候选都是 5/5 strict LUT hit；最终 resume 段也保持 25/25 hits、0 miss。

## 4. 可讲结论：所有采样架构都满足硬件约束

佐证数据：

| 项目 | 结果 |
|---|---:|
| total evaluated | 50 |
| feasible | 50 |
| infeasible | 0 |
| feasible ratio | 50/50 |
| max_latency_ms constraint | 120.0 ms |

4 个候选的硬件代价：

| candidate | latency_ms | LUT | DSP | BRAM | power_w |
|---|---:|---:|---:|---:|---:|
| mbconv k3/e3 | 86.2467 | 12870 | 363 | 133 | 9.0312 |
| mbconv k3/e6 | 111.0041 | 13645 | 363 | 150 | 9.0312 |
| mbconv k5/e3 | 86.6792 | 13959 | 363 | 169 | 9.2472 |
| mbconv k5/e6 | 111.8362 | 14732 | 363 | 190 | 9.6252 |

可讲口径：

> 50 个采样架构全部 feasible。当前 4 个 strict-covered 候选都低于 120 ms latency 约束，资源也在 AV7K325 约束内；因此这轮没有因为硬件约束造成搜索空间坍缩。

## 5. 可讲结论：eval_epochs=5 的 proxy 有一定区分度，但仍不能作为最终精度结论

修复后 RL 的整体 proxy 分布：

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| macro_f1 | 0.4773 | 0.4796 | 0.3271 | 0.5741 |
| top1 | 0.7234 | 0.7327 | 0.5808 | 0.7904 |

按候选聚合：

| candidate | n | mean macro_f1 | min-max macro_f1 | mean top1 | min-max top1 |
|---|---:|---:|---:|---:|---:|
| mbconv k3/e3 | 20 | 0.4804 | 0.3619-0.5471 | 0.7238 | 0.5865-0.7712 |
| mbconv k3/e6 | 12 | 0.4809 | 0.3816-0.5312 | 0.7154 | 0.5808-0.7596 |
| mbconv k5/e3 | 13 | 0.4638 | 0.3271-0.5614 | 0.7268 | 0.6423-0.7904 |
| mbconv k5/e6 | 5 | 0.4915 | 0.3276-0.5741 | 0.7323 | 0.6096-0.7788 |

候选均值跨度：

| metric | 修复后 RL | Random baseline |
|---|---:|---:|
| macro_f1 mean range | 0.0277 | 0.0416 |
| top1 mean range | 0.0169 | 0.0444 |

可讲口径：

> eval_epochs=5 比 eval_epochs=1 更有分辨能力：修复后 RL 中候选均值的 macro_f1 跨度约 2.77%，top1 跨度约 1.69%；Random baseline 中跨度更大，macro_f1 约 4.16%，top1 约 4.44%。但 5 epoch 仍然是 proxy，有明显随机波动，不能作为最终架构精度结论。

## 6. 可讲结论：修复后 RL 的 top-K 与 Random 接近，但不能下“RL 优于 Random”的结论

top-K 对比：

| 方法 | top3 macro_f1 mean | top3 top1 mean | top5 macro_f1 mean |
|---|---:|---:|---:|
| 原始 RL baseline | 0.5720 | 0.7577 | 0.5619 |
| Random baseline | 0.5599 | 0.7679 | 0.5540 |
| 修复后 RL | 0.5609 | 0.7769 | 0.5519 |

解释：

- 原始 RL top-K 高不代表 RL 有效，因为它 50 次都是同一个候选的重复 proxy 训练，相当于从随机训练波动里挑最好的。
- 修复后 RL top3 macro_f1 与 Random top3 几乎相同：`0.5609` vs `0.5599`。
- 修复后 RL top3 top1 略高于 Random：`0.7769` vs `0.7679`，但差异很小，且不是多 seed 统计。

可讲口径：

> 修复后 RL 的 top-K proxy 与 Random 大致持平，说明 controller 恢复探索后没有明显劣于随机采样。但当前不能讲 RL 已经优于 Random，因为搜索空间太小、proxy 训练较短、没有多 seed 统计。

## 7. 可讲结论：Top 候选具备可追踪的硬件-精度记录

按 macro_f1 排序的 Top-5：

| rank | arch_id | candidate | macro_f1 | top1 | latency_ms | LUT | DSP | BRAM | power_w |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | rl_arch_12 | mbconv k5/e6 | 0.5741 | 0.7731 | 111.8362 | 14732 | 363 | 190 | 9.6252 |
| 2 | rl_arch_27 | mbconv k5/e3 | 0.5614 | 0.7865 | 86.6792 | 13959 | 363 | 169 | 9.2472 |
| 3 | rl_arch_42 | mbconv k3/e3 | 0.5471 | 0.7712 | 86.2467 | 12870 | 363 | 133 | 9.0312 |
| 4 | rl_arch_4 | mbconv k3/e3 | 0.5390 | 0.7596 | 86.2467 | 12870 | 363 | 133 | 9.0312 |
| 5 | rl_arch_44 | mbconv k3/e3 | 0.5382 | 0.6885 | 86.2467 | 12870 | 363 | 133 | 9.0312 |

可讲口径：

> Top-5 候选的精度 proxy 和硬件代价都已经记录下来，可以追踪到 arch_id、stage1 choice、latency、LUT/DSP/BRAM 和 power。若后续要做完整训练，优先应在更大搜索或更稳定 proxy 后再选 Top-3，而不是直接把这 4-candidate proxy 当最终架构选择。

## 8. 最终决策建议

可以直接讲的本轮结论：

1. strict40 LUT 已接入并能支撑搜索链路。
2. 当前 strict40 4-candidate 搜索空间全部 feasible。
3. 原始 RL exploration collapse 已被定位并修复。
4. 修复后 50 episodes 覆盖 4/4 候选，feasible ratio 为 50/50。
5. eval_epochs=5 有一定 proxy 分化，但不能替代完整训练。
6. 修复后 RL 与 Random top-K proxy 接近，尚不能证明 RL 优于 Random。

下一步建议：

- 如果目标是继续验证 controller：可以跑 200 episodes，看 coverage、top-K 和候选采样分布是否稳定。
- 如果目标是真正 NAS：应优先扩大 measured LUT 覆盖，再扩大 `stage_block_choices`；否则 4 个候选空间太小，RL 的价值很难体现。
- Top-3 完整 240 epoch 训练建议放在更大候选空间或更稳定 proxy 之后。
