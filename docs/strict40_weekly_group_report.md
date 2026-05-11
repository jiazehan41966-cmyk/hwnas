# Strict40 LUT 接入与短搜索进展汇报

日期：2026-05-10
项目：HW-NAS / FPGA / NKSID 声呐图像分类
汇报定位：本轮是 strict40 LUT 接入、搜索链路与 RL exploration 诊断，不是最终 NAS 性能结论。

## 一、当前结论

1. **40 条 measured LUT 已经接入搜索链路。**
   `board_measure_status_current_impl.csv` 作为状态权威来源，生成了 strict40 LUT/status 文件；metadata 显示 `measured_entries=40`、`defer_current_impl=44`、`board_result_status_overrides=0`。短搜索和确定性枚举中均没有 `true_miss` 或 `deferred_hit`。

2. **strict40 当前可严格覆盖 4 个候选架构。**
   确定性枚举 4 个候选全部可评估，feasible ratio 为 `4/4`，每个候选均为 `5/5` LUT hit，`true_miss=0`、`deferred_hit=0`、`hit_rate=1.0`。

3. **原始 RL 存在 exploration collapse。**
   在 50 episodes、eval_epochs=5 的 baseline 对比中，RL 50 次全部采到同一个候选；Random 能覆盖全部 4 个候选。因此此前 RL top-K 指标不能解释为 RL 有效，只能说明链路能跑通。

4. **eval_epochs=5 已出现可观测但仍有噪声的 proxy 差异。**
   Random 覆盖 4 个候选时，候选均值的 `macro_f1` 跨度约 `4.16%`，`top1` 跨度约 `4.44%`，高于 1%。这说明 5 epoch 比 1 epoch 更有区分度，但仍不能替代完整训练结论。

5. **RL exploration 修复后已完成 50 episodes 验证。**
   已加入 temperature、entropy regularization、epsilon exploration 和 exploration bonus。恢复补跑后，run 已 finalize 为 `completed`，50 episodes 覆盖全部 4 个候选，feasible ratio 为 `50/50`，stderr 为空。该结果说明 exploration collapse 已被修复；但由于搜索空间仍只有 4 个候选，不能据此证明最终 NAS 性能优势。

## 二、已完成工作

### 1. strict40 LUT/status 文件固化

生成文件：

- `hls_lut_builder/results/formal_lut_strict40_v1.json`
- `hls_lut_builder/results/formal_lut_status_strict40_v1.json`

关键元数据：

| 项目 | 数值 |
|---|---:|
| candidate cases | 84 |
| measured entries | 40 |
| defer_current_impl | 44 |
| board_result_status_overrides | 0 |
| status_authoritative | true |

结论：strict40 的定义已经从“结果文件覆盖”改为“status CSV 权威”，没有把 11 条 board-result overlay 混入 measured。

### 2. strict40 NKSID 短搜索 baseline

配置：

- `configs/search/formal_lut_strict40_nksid_short_av7k325.yaml`

结果目录：

- `results/formal_lut_strict40_nksid_short_8ep`

关键结果：

| 指标 | 数值 |
|---|---:|
| episodes | 8 |
| feasible | 8 |
| infeasible | 0 |
| feasible ratio | 8/8 |
| LUT hits | 40 |
| LUT misses | 0 |
| true_miss | 0 |
| deferred_hit | 0 |
| hit rate | 1.0 |

trace metric，仅用于记录链路：

| metric | best observed |
|---|---:|
| macro_f1 | 0.4712 |
| top1 | 0.7019 |
| latency_ms | 86.2467 |
| LUT | 12870 |
| DSP | 363 |
| BRAM | 133 |
| power_w | 9.0312 |

说明：8 episodes × 4 candidates 只用于 plumbing validation，不能说明 RL 学习能力；eval_epochs=1 的 macro_f1/top1 不用于架构优劣判断。

### 3. deterministic 4-candidate enumeration smoke

脚本：

- `scripts/enumerate_strict40_candidates.py`

结果：

- `results/strict40_deterministic_4candidate_smoke/summary.json`
- `results/strict40_deterministic_4candidate_smoke/candidates.csv`

4 个候选全部 strict LUT covered：

| stage1 choice | feasible | latency_ms | LUT | DSP | BRAM | power_w | LUT hit/miss | true_miss | deferred_hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mbconv k3/e3 | true | 86.2467 | 12870 | 363 | 133 | 9.0312 | 5/0 | 0 | 0 |
| mbconv k5/e3 | true | 86.6792 | 13959 | 363 | 169 | 9.2472 | 5/0 | 0 | 0 |
| mbconv k3/e6 | true | 111.0041 | 13645 | 363 | 150 | 9.0312 | 5/0 | 0 | 0 |
| mbconv k5/e6 | true | 111.8362 | 14732 | 363 | 190 | 9.6252 | 5/0 | 0 | 0 |

结论：已经把“4 个候选都可被评估”从 RL 随机采样中拆出来验证了。

### 4. RL vs Random eval5 baseline 诊断

结果文件：

- `results/strict40_rl_vs_random_eval5_analysis.json`
- RL：`results/formal_lut_strict40_nksid_rl_50ep_eval5`
- Random：`results/formal_lut_strict40_nksid_random_50c_eval5`

整体对比：

| 方法 | completed | feasible ratio | unique candidates | LUT hit rate | true_miss | deferred_hit |
|---|---:|---:|---:|---:|---:|---:|
| RL baseline | yes | 50/50 | 1/4 | 1.0 | 0 | 0 |
| Random baseline | yes | 50/50 | 4/4 | 1.0 | 0 | 0 |

候选覆盖：

| 方法 | k3/e3 | k3/e6 | k5/e3 | k5/e6 |
|---|---:|---:|---:|---:|
| RL baseline | 50 | 0 | 0 | 0 |
| Random baseline | 18 | 11 | 9 | 12 |

Random 下的候选均值：

| candidate | n | mean macro_f1 | mean top1 | latency_ms | LUT | DSP | BRAM | power_w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k3/e3 | 18 | 0.4798 | 0.6974 | 86.2467 | 12870 | 363 | 133 | 9.0312 |
| k3/e6 | 11 | 0.4688 | 0.7038 | 111.0041 | 13645 | 363 | 150 | 9.0312 |
| k5/e3 | 9 | 0.4946 | 0.7378 | 86.6792 | 13959 | 363 | 169 | 9.2472 |
| k5/e6 | 12 | 0.4530 | 0.6934 | 111.8362 | 14732 | 363 | 190 | 9.6252 |

诊断结论：

- strict40 空间本身可行，Random feasible ratio 也是 50/50。
- RL 没有真正探索，50 episodes 只采到 1 个候选，属于 controller exploration 问题。
- RL top3 macro_f1 略高于 Random top3，但因为 RL 重复同一架构，该比较不能说明 RL 优于 Random。
- eval_epochs=5 的候选均值差异大于 1%，有一定 proxy 分辨能力，但仍需更长训练确认。

### 5. RL exploration 修复与完整验证

已加入：

- controller temperature
- entropy regularization
- epsilon exploration
- exploration bonus

配置：

- `configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml`

完整补跑结果：

| 项目 | 数值 |
|---|---:|
| episodes | 50 |
| completed records | 50 |
| feasible | 50 |
| infeasible | 0 |
| feasible ratio | 50/50 |
| unique candidates | 4/4 |
| macro_f1 mean | 0.4773 |
| macro_f1 min/max | 0.3271 / 0.5741 |
| top1 mean | 0.7234 |
| top1 min/max | 0.5808 / 0.7904 |

候选覆盖：

| k3/e3 | k3/e6 | k5/e3 | k5/e6 |
|---:|---:|---:|---:|
| 20 | 12 | 13 | 5 |

候选均值跨度：`macro_f1=0.0277`，`top1=0.0169`，均大于 1%，说明 eval_epochs=5 的 proxy 信号在这 4 个候选上有可观测差异，但仍有明显随机性。`lut_stats.json` 在 resume 后只记录续跑的 5 个 episode，因此显示 `hits=25`；续跑段 `misses=0`、`true_misses=0`、`deferred_hits=0`。全量候选明细已写入 `results/candidates.jsonl` 和 `results/candidates.csv`。

## 三、当前待办

1. **基于完整 50 episodes 结果做最终对比表。**
   当前修复后 RL 已覆盖 4/4 候选，feasible ratio 为 50/50。下一步应把修复后 RL、原始 RL、Random 三者整理到同一张对比表，并注明 resume 段 LUT 统计只覆盖最后 5 个 episode。

2. **补充 RL vs Random 的最终对比表。**
   对比维度：候选覆盖、feasible ratio、macro_f1/top1 分布、top-K、latency/LUT/DSP/BRAM/power。

3. **根据 50-episode 结果决策下一轮。**
   如果 RL 覆盖 3-4 个候选且 proxy 精度有分化，再扩大到 200 episodes；如果仍然塌缩，继续调 temperature、entropy、epsilon 或 exploration bonus。

4. **扩大搜索空间前先补 LUT 覆盖。**
   strict LUT 模式下不能依赖 interpolation 或 shape-only matching；如果想增加更多 stage_block_choices，需要先补测对应 measured LUT，或明确切换到非 strict/模型估计路径。

## 四、本周建议工作安排

### 周一：整理 exploration 50 episodes 终版

- 已补完 `formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml`。
- 输出 finalized analysis：coverage、feasible ratio、LUT integrity、macro_f1/top1 分布。
- 判据结果：50 episodes 覆盖 4/4 个候选，exploration 基本恢复。

### 周二：做 50ep RL vs Random 正式对比

- 固定 strict40、eval_epochs=5、NKSID fold 0。
- 对比 Random 与修复后 RL：feasible ratio、候选覆盖、top-K、proxy 分布。
- 不把 macro_f1/top1 当最终精度，只作为 proxy ranking 信号。

### 周三：准备 200 episodes 或补测 LUT 清单

- 若 50ep 通过，启动 200 episodes eval5 或 eval10。
- 若要扩大搜索空间，先列出新增 stage_block_choices 对应的缺失 LUT cases，决定补测优先级。

### 周四：Top-3 候选筛选条件

- 只有在 200 episodes 或稳定 proxy 下，才选择 Top-3 架构。
- Top-3 进入完整 240 epoch 训练前，需要记录架构、seed、fold、硬件代价、LUT hit/miss 统计。

### 周五：整理组会图表和论文材料

- 表 1：strict40 LUT 统计与链路完整性。
- 表 2：4 候选硬件代价。
- 表 3：RL vs Random coverage/feasible/proxy 对比。
- 图 1：候选覆盖柱状图。
- 图 2：macro_f1/top1 proxy 分布图。

## 五、组会可直接讲的结论

1. 本周已经证明 strict40 measured LUT 可以接入 HW-NAS cost/query/search 链路；在 strict 模式下，短搜索和枚举均实现 `true_miss=0`、`deferred_hit=0`、`hit_rate=1.0`。
2. 当前 strict40 只支持一个很小的 4-candidate search space。该空间全部 feasible，硬件代价差异清晰：latency 约 `86.25-111.84 ms`，LUT 约 `12870-14732`，BRAM 约 `133-190`，power 约 `9.03-9.63 W`。
3. 原始 RL 50 episodes 只探索 1 个候选，而 Random 覆盖 4 个候选，说明问题不在 LUT 或可行性约束，而在 RL controller exploration。
4. 修复 exploration 后，完整 50 episodes 覆盖了全部 4 个候选，说明 temperature/entropy/epsilon/bonus 方向有效；下一步应扩大 episodes 或搜索空间，而不是继续在 4 个候选上证明 RL。
5. 现阶段不对 macro_f1/top1 下最终架构结论；eval_epochs=5 只用于 proxy 诊断，完整模型优劣需要更大搜索和更长训练验证。

## 六、关键文件路径

- strict40 LUT：`hls_lut_builder/results/formal_lut_strict40_v1.json`
- strict40 status：`hls_lut_builder/results/formal_lut_status_strict40_v1.json`
- short config：`configs/search/formal_lut_strict40_nksid_short_av7k325.yaml`
- deterministic smoke script：`scripts/enumerate_strict40_candidates.py`
- deterministic smoke results：`results/strict40_deterministic_4candidate_smoke`
- RL vs Random analysis：`results/strict40_rl_vs_random_eval5_analysis.json`
- exploration config：`configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml`
- exploration final run：`results/formal_lut_strict40_nksid_rl50_eval5_explore_v1`
- exploration final analysis：`results/strict40_rl50_eval5_explore_final_analysis.json`
- baseline docs：`docs/strict40_plumbing_baseline.md`
- expansion plan：`docs/strict40_next_rl_expansion.md`
- eval5 findings：`docs/strict40_eval5_rl_random_findings.md`
