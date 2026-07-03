# Strict40 LUT 接入与 RL Exploration 验证组会稿

日期：2026-05-10
主题：40 条 measured LUT 接入搜索链路、strict40 搜索验证、RL exploration 修复

## 开场总述

这周主要做了三件事。第一，把当前状态权威的 40 条 measured LUT 接入到 HW-NAS 搜索链路里，并且生成 strict40 LUT/status 文件。第二，在严格 LUT 模式下验证 4 个候选架构都能被 cost 查询和硬件约束评估。第三，定位并修复了 RL controller 的 exploration collapse，修复后 50 episodes 覆盖了全部 4 个 strict-covered 候选。

这轮实验的定位是链路和 controller 验证，不是最终 NAS 性能实验。所以 macro_f1 和 top1 只作为 5 epoch 短训练得到的 proxy 信号，不做最终架构优劣结论。

## 第 1 页：40 条 measured LUT 接入搜索链路

这一页我主要说明 strict40 LUT 是怎么定义的，以及为什么当前搜索空间要限制得很小。

我们这次没有再使用 board-result overlay，而是把 `board_measure_status_current_impl.csv` 作为权威状态来源。最终生成了两个文件：`formal_lut_strict40_v1.json` 和 `formal_lut_status_strict40_v1.json`。

关键数据是：

- 总候选算子 case 是 84 个；
- 其中 measured entries 是 40 个；
- defer_current_impl 是 44 个；
- board_result_status_overrides 是 0；
- status_authoritative 是 true。

这说明 strict40 的定义是严格的：只有 status CSV 里标记为 measured 的 40 个算子进入 LUT，不额外叠加已有 board result 里的 11 条结果。

右边这个 MBConv measured/deferred 表说明了当前 measured 覆盖不是完整的。不同 shape 下，有些 e3/k3、e6/k3、e3/k5、e6/k5 已经 measured，有些因为 csynth crash 或 OOM 被 defer。这个结果直接决定了当前 strict LUT 模式下不能随意扩大搜索空间。只要新增的 block choice 触碰到 deferred case，就会出现 true_miss 或 deferred_hit。

所以当前这一轮不是做大规模 NAS，而是先证明：40 条 measured LUT 能不能稳定接入搜索链路。

这一页底部的 trace metric 只做链路记录：短搜索里 observed macro_f1 是 0.4712，top1 是 0.7019，latency 是 86.25 ms，LUT 是 12870。这里不对 macro_f1/top1 下结论，因为短搜索只有 8 episodes，且 eval_epochs 只有 1。

本页结论可以讲成：

> strict40 LUT 已经按 status-authoritative 方式接入，40 条 measured、44 条 deferred，且没有 overlay。当前 measured 覆盖决定了搜索空间必须先收缩到 strict-covered 子集。

## 第 2 页：Strict40 Plumbing Baseline 和 RL Exploration

这一页建议标题改成：

> Strict40 plumbing baseline + RL exploration after fix

因为左半部分是 8 episode 的链路 baseline，右半部分是修复后 50 episode 的 RL exploration 覆盖，两者不是同一个实验。

左边 8ep baseline 的目的很简单，就是验证链路是否跑通。结果是：

- episodes 是 8；
- feasible 是 8；
- feasible ratio 是 8/8；
- LUT hits 是 40；
- hit rate 是 1.0；
- 没有 true_miss；
- 没有 deferred_hit。

这说明在当前 strict40 配置下，搜索链路可以正常完成 LUT 查询、硬件约束检查和 reward 计算。

右边是修复后 50 episodes 的候选覆盖。这里的候选覆盖指 RL controller 在 50 次采样中实际访问了多少种架构。当前 strict40 搜索空间只有 4 个 strict-covered 候选，主要差异是 stage1 的 MBConv block：

- mbconv k3/e3 被采样 20 次，占 40%；
- mbconv k3/e6 被采样 12 次，占 24%；
- mbconv k5/e3 被采样 13 次，占 26%；
- mbconv k5/e6 被采样 5 次，占 10%。

修复前的原始 RL 在 50 episodes 里只采到 1 个候选；修复后 50 episodes 覆盖了全部 4 个候选。这个对比说明，之前的问题不是 LUT 或硬件约束导致搜索空间不可行，而是 RL controller 的 exploration collapse。

修复方式包括 controller temperature、entropy regularization、epsilon exploration 和 exploration bonus。修复后候选覆盖恢复到 4/4，说明 controller 已经能实际探索搜索空间。

本页结论可以讲成：

> 8ep baseline 证明 strict40 查询链路能跑通；50ep exploration 验证证明修复后的 RL controller 能覆盖全部 4 个 strict-covered 候选，原来的 collapse 已经被拉回来了。

## 第 3 页：所有采样架构都满足硬件约束

这一页说明硬件可行性，不讨论最终精度。

完整 50 episode 验证里：

- total evaluated 是 50；
- feasible 是 50；
- infeasible 是 0；
- feasible ratio 是 50/50；
- latency 约束是 120 ms。

4 个候选的硬件代价分别是：

- mbconv k3/e3：latency 86.2467 ms，LUT 12870，DSP 363，BRAM 133，power 9.0312 W；
- mbconv k3/e6：latency 111.0041 ms，LUT 13645，DSP 363，BRAM 150，power 9.0312 W；
- mbconv k5/e3：latency 86.6792 ms，LUT 13959，DSP 363，BRAM 169，power 9.2472 W；
- mbconv k5/e6：latency 111.8362 ms，LUT 14732，DSP 363，BRAM 190，power 9.6252 W。

所有候选都低于 120 ms latency 约束，资源也在 AV7K325 的 LUT、DSP、BRAM 约束内。因此 50 次采样全部 feasible。

这里可以强调一个判断：如果 RL 只采到 1 个候选，但 Random 和修复后 RL 都能覆盖 4 个候选，而且 4 个候选全部 feasible，那么原因就不是约束太严或 LUT 缺失，而是原始 RL exploration 有问题。

本页结论可以讲成：

> 当前 4 个 strict-covered 候选全部满足硬件约束，50 个采样架构全部 feasible。因此搜索坍缩不是由硬件约束造成的。

## 建议补充页：原始 RL vs Random vs 修复后 RL

建议在第 2 页或第 3 页之间加一页对比表，这是这轮最有结论性的数据。

| 方法 | episodes | feasible | candidate coverage | 结论 |
|---|---:|---:|---:|---|
| 原始 RL | 50 | 50/50 | 1/4 | exploration collapse |
| Random | 50 | 50/50 | 4/4 | 搜索空间本身可覆盖 |
| 修复后 RL | 50 | 50/50 | 4/4 | exploration 恢复 |

这一页的讲稿：

原始 RL 50 episodes 全部 feasible，但是只访问了 1 个候选；Random baseline 同样 50 次采样，覆盖了全部 4 个候选；修复后的 RL 也覆盖了全部 4 个候选。

这个对比把问题拆清楚了。Random 能覆盖 4 个候选，说明搜索空间不是只有一个可行点；修复后 RL 能覆盖 4 个候选，说明 controller exploration 的改动有效。原始 RL 只覆盖 1 个候选，因此可以定位为 exploration collapse，而不是 LUT 或 constraint 的问题。

可讲结论：

> Random 和修复后 RL 都能覆盖 4/4 候选，原始 RL 只能覆盖 1/4。因此本轮定位并修复的是 RL exploration 问题。

## 建议补充页：Proxy 结果只用于诊断

建议再加一页，把 proxy 讲清楚，避免老师直接问“这个 macro_f1 能不能说明架构好”。

这里 proxy 指的是搜索阶段的近似评价信号。本实验中 proxy 是每个候选只训练 eval_epochs=5 后得到的 macro_f1 和 top1。它比 eval_epochs=1 更有参考价值，但仍不能替代完整训练。

修复后 RL 的 proxy 分布是：

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| macro_f1 | 0.4773 | 0.4796 | 0.3271 | 0.5741 |
| top1 | 0.7234 | 0.7327 | 0.5808 | 0.7904 |

候选均值跨度：

- macro_f1 mean range 是 0.0277，也就是约 2.77%；
- top1 mean range 是 0.0169，也就是约 1.69%。

Random baseline 中候选均值跨度更大：

- macro_f1 mean range 是 0.0416；
- top1 mean range 是 0.0444。

这说明 eval_epochs=5 已经有一定区分度，不像 eval_epochs=1 那样几乎只能看链路。但同一个候选多次短训练仍然有明显波动，所以这里不能把 top1 或 macro_f1 作为最终架构结论。

可讲结论：

> 5 epoch proxy 有一定区分度，但仍然只是搜索阶段的近似信号。本轮只用它判断搜索链路和候选差异是否可观测，不用它下最终精度结论。

## 第 4 页：下一步计划

这一页建议把“240 episodes”改成“200 episodes”或“补一轮更长搜索”。原因是前面讨论的是搜索 episodes，而后面 Top-3 是完整训练 240 epoch，这两个数字很容易混淆。

建议改成下面这个版本。

### 1. 跑一轮更长搜索，建议 200 episodes

目的：验证修复后的 controller 在更长搜索中是否稳定探索。

记录指标：

- candidate coverage；
- feasible ratio；
- top-K proxy；
- LUT true_miss/deferred_hit；
- 每个候选的采样比例；
- controller 是否再次 collapse。

判断标准：

- 如果 200 episodes 仍然覆盖 4/4 候选，说明 exploration 修复稳定；
- 如果重新集中到单一候选，需要继续调 temperature、entropy、epsilon 或 exploration bonus；
- 如果 top-K 与 Random 仍然接近，说明在 4 候选空间里 RL 优势难以体现。

### 2. 提高 proxy 可靠性

当前 eval_epochs=5 有一定区分度，但噪声仍然明显。下一步可以：

- 对候选做多 seed proxy eval；
- 或把 eval_epochs 提到 10；
- 记录候选均值、方差和 top-K 稳定性。

目标不是追求短训最高分，而是判断候选之间的 proxy 差异是否稳定。

### 3. 扩大 measured LUT 覆盖和搜索空间

这是做真正 NAS 的关键。

当前 strict40 只支持 4 个 strict-covered 候选，搜索空间太小。即使跑 200 episodes，也更像 controller validation，不足以证明 NAS 的搜索价值。

如果要扩大 stage_block_choices，就需要先补测更多 LUT。否则 strict LUT 模式下会遇到 true_miss 或 deferred_hit。

所以建议先列出新增候选需要的 missing/deferred LUT cases，优先补测能扩大 stage1/stage2 组合数的 MBConv variants。

### 4. Top-3 完整训练

Top-3 完整训练建议放在搜索空间扩大、proxy 更稳定之后。

完整训练设置：

- Top-3 架构；
- 每个架构完整 240 epoch；
- 固定 dataset split、fold、seed；
- 最终比较 macro_f1、top1、latency、LUT、DSP、BRAM、power。

这一页的最终口径：

> 下一步先做更长搜索验证 controller 稳定性，再提高 proxy 可靠性，并补测 LUT 扩大搜索空间。Top-3 的 240 epoch 完整训练应放在搜索空间和 proxy 更稳定之后。

## 结尾总结

本周的核心进展是，strict40 measured LUT 已经接入 HW-NAS 搜索链路，4 个 strict-covered 候选全部可评估并满足硬件约束；原始 RL 的 exploration collapse 已经定位并修复。修复后 50 episodes 覆盖 4/4 候选，feasible ratio 为 50/50，没有 true_miss 或 deferred_hit。

当前结论不包含最终架构精度判断，因为 eval_epochs=5 仍然只是 proxy。下一步需要做更长 episodes 的稳定性验证，同时补测更多 measured LUT 来扩大搜索空间，之后再选择 Top-3 做完整 240 epoch 训练。
