# Strict40 Frozen Baseline Manifest

日期：2026-05-11
冻结对象：strict40 measured LUT 接入、4-candidate deterministic smoke、RL exploration 修复后 50-episode 验证
本地归档包：`results/strict40_frozen_baseline_20260511.zip`

## 1. Frozen Scope

本次冻结保留以下内容：

- strict40 search configs
- strict40 LUT/status JSON
- strict40 short plumbing run
- deterministic 4-candidate smoke run
- RL exploration fixed 50-episode run
- final analysis JSON
- 组会汇报稿和结论说明

本次冻结只固定链路和 controller exploration 结论，不固定最终架构精度结论。

## 2. Artifact Inventory

| 类型 | 路径 | 状态 |
|---|---|---|
| short config | `configs/search/formal_lut_strict40_nksid_short_av7k325.yaml` | kept |
| RL explore config | `configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml` | kept |
| strict40 LUT | `hls_lut_builder/results/formal_lut_strict40_v1.json` | kept |
| strict40 status | `hls_lut_builder/results/formal_lut_status_strict40_v1.json` | kept |
| short run | `results/formal_lut_strict40_nksid_short_8ep` | kept |
| deterministic smoke | `results/strict40_deterministic_4candidate_smoke` | kept |
| RL fixed run | `results/formal_lut_strict40_nksid_rl50_eval5_explore_v1` | kept |
| final analysis | `results/strict40_rl50_eval5_explore_final_analysis.json` | kept |
| weekly report | `docs/strict40_weekly_group_report.md` | kept |
| exploration report | `docs/strict40_rl_exploration_validation_output.md` | kept |
| talk script | `docs/strict40_group_meeting_talk_script.md` | kept |
| frozen archive | `results/strict40_frozen_baseline_20260511.zip` | created |

## 3. Hashes

| Artifact | SHA256 |
|---|---|
| `hls_lut_builder/results/formal_lut_strict40_v1.json` | `554FC8466657458506BC05DD5E43930C99F93A1D106ABE9E876EB19AC0D90BD5` |
| `hls_lut_builder/results/formal_lut_status_strict40_v1.json` | `E2BE58A6567B06C2B817DC71B8505D6750ECC66D5FFB24D9150301BC2B49F015` |
| `configs/search/formal_lut_strict40_nksid_short_av7k325.yaml` | `A416DA0BAF2EB17599CA3953CAD1E8F1E531BE8CECA64EC5AC586D612D5799C0` |
| `configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml` | `DD086B7B38D3C6B6C4927C42E1C385225FBCA41618FFA102E8A5F55E9F2E4726` |
| `results/strict40_rl50_eval5_explore_final_analysis.json` | `F8E52721A9795CDE5A2F8C1A8217DAA9A8A60279645BE3D5F1EBF51D3E658D37` |
| `results/strict40_frozen_baseline_20260511.zip` | `73B37F988272D7F675542A68F9D1AE0620086A2B0CD87769706A00E7A3DA8E22` |

## 4. Fixed Results

### Strict40 LUT Status

| 指标 | 数值 |
|---|---:|
| candidate cases | 84 |
| measured entries | 40 |
| defer_current_impl | 44 |
| board_result_status_overrides | 0 |
| status_authoritative | true |

结论：strict40 LUT 已按 status-authoritative 方式接入；只使用 40 条 measured entries，没有 board-result overlay。

### Short Plumbing Baseline

| 指标 | 数值 |
|---|---:|
| episodes | 8 |
| feasible | 8 |
| feasible ratio | 8/8 |
| LUT hits | 40 |
| LUT misses | 0 |
| true_miss | 0 |
| deferred_hit | 0 |
| hit rate | 1.0 |

结论：strict40 LUT 查询、硬件约束检查、reward 计算链路跑通。

### Deterministic 4-Candidate Smoke

| 指标 | 数值 |
|---|---:|
| candidate count | 4 |
| feasible | 4 |
| infeasible | 0 |
| strict_lut_ok | true |
| per-candidate LUT hit/miss | 5/0 |
| true_miss | 0 |
| deferred_hit | 0 |

候选硬件代价：

| candidate | latency_ms | LUT | DSP | BRAM | power_w |
|---|---:|---:|---:|---:|---:|
| mbconv k3/e3 | 86.2467 | 12870 | 363 | 133 | 9.0312 |
| mbconv k5/e3 | 86.6792 | 13959 | 363 | 169 | 9.2472 |
| mbconv k3/e6 | 111.0041 | 13645 | 363 | 150 | 9.0312 |
| mbconv k5/e6 | 111.8362 | 14732 | 363 | 190 | 9.6252 |

结论：4 个 strict-covered 候选全部可被评估，且均满足硬件约束。

### RL Exploration Fixed Run

| 指标 | 数值 |
|---|---:|
| run status | completed |
| episodes | 50 |
| feasible | 50 |
| infeasible | 0 |
| feasible ratio | 50/50 |
| unique candidates | 4/4 |
| stderr | 0 bytes |

候选覆盖：

| candidate | count | proportion |
|---|---:|---:|
| mbconv k3/e3 | 20 | 40% |
| mbconv k3/e6 | 12 | 24% |
| mbconv k5/e3 | 13 | 26% |
| mbconv k5/e6 | 5 | 10% |

结论：原始 RL 50 episodes 只覆盖 1/4 候选；修复后 RL 50 episodes 覆盖 4/4 候选，controller exploration collapse 已修复。

备注：`results/formal_lut_strict40_nksid_rl50_eval5_explore_v1/results/lut_stats.json` 在 resume 后只记录最后 5 个 episode，因此显示 `hits=25`。resume 段为 `misses=0`、`true_misses=0`、`deferred_hits=0`。全量候选集合已由 deterministic smoke 验证为每个候选 `5/5` strict LUT hit。

## 5. Explicit Non-Conclusions

本轮不做以下结论：

- 不用 `macro_f1` 或 `top1` 判断最终架构优劣。
- 不声明 RL 已经优于 Random。
- 不声明 4-candidate strict40 空间足以支撑正式 NAS 结论。
- 不把 `eval_epochs=5` 当作完整训练结果。

本轮可以做的结论：

- strict40 measured LUT 已经接入搜索链路。
- 4 个 strict-covered 候选全部 feasible。
- RL controller exploration collapse 已被修复。
- 当前结果适合作为 strict40 plumbing/controller baseline。

## 6. Reproduction Commands

```powershell
python run_search.py --config configs/search/formal_lut_strict40_nksid_short_av7k325.yaml
python scripts/enumerate_strict40_candidates.py
python run_search.py --config configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml
python run_search.py --config configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml --resume
```

## 7. Next Work Gate

下一步可以进入两条线：

1. Controller 稳定性验证：在当前 strict40 4-candidate 空间跑更长搜索，例如 200 episodes，对比 Random baseline。
2. 正式 NAS 准备：补测更多 measured LUT，扩大 `stage_block_choices`，再做更有意义的搜索和 Top-3 完整 240 epoch 训练。
