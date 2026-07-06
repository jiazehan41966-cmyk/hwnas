# Proxy Reliability Audit v2：优化训练执行

状态：v2 manifest 已生成，正式观测 `0/1200`。本机无 CUDA，正式 GPU
worker 会拒绝启动；最小/中位/最大解析延迟架构的 CPU 1-epoch
benchmark 已完成，用时47.51/58.68/80.07秒，全部
`formal_eligible=false` 且未执行 outer evaluation。

v1 的 7200-budget-unit manifest 保留为历史记录，不覆盖、不删除。

## 优化目标

v1 对同一个 `(architecture, seed, outer_fold)` 分别训练
1/3/10/30/150 epoch。v2 改为一条150-epoch trajectory：

1. 初始化模型并用副本计算 NASWOT；
2. 训练到150 epoch；
3. 在1/3/10/30 epoch记录 best-so-far inner-validation proxy；
4. epoch150完成 inner selection 后只评价一次 outer fold；
5. 将一个trajectory展开为六个分析观测。

在 constant LR、同一初始化与同一数据顺序下，这保留了 prefix
定义，同时把正式训练轨迹从6000条降为1200条。总训练 epoch 从
232,800降为180,000。batch size 从16改为 canonical mobile_anchor
search 使用的32。

## 阶段

| stage | fold/seed cells | trajectories | evidence scope |
|---|---:|---:|---|
| `phase_a_signal_discovery` | fold0 × 5 seeds | 240 | architecture×seed signal |
| `phase_b_fold_robustness` | folds1–4 × seed42 | 192 | architecture×fold robustness |
| `phase_c_full_confirmation` | folds1–4 × seeds43–46 | 768 | complete crossed confirmation |
| total | 25 | 1200 | formal Gate 0 |

Phase A 可计算两因素 architecture×seed ICC；Phase B 可计算两因素
architecture×fold ICC。两者均保持 `not_ready`，只有 Phase C 补齐
完整交叉网格后才能给出正式 Gate 0 pass/fail。

## 不变量

- architecture sampling 不读取历史 proxy Top-K；
- 所有架构共用 seeds 与 split policy；
- 0/1/3/10/30 epoch 不消费 outer loader；
- outer fold 只在完整150-epoch trajectory 完成 inner selection 后评价一次；
- `--max-budget` 产生 `benchmark_completed`，collector 拒绝将其作为正式证据；
- hardware truth 与 classification truth 继续分表；
- v1、v2 manifest fingerprint 与 observations 不混用。

## 本地 benchmark

```powershell
python scripts/run_proxy_reliability_observation.py `
  --run-matrix results/proxy_reliability_gate0/manifest_v2/run_matrix.jsonl `
  --work-index 0 `
  --max-budget 1 `
  --device cpu `
  --num-workers 0
```

输出写入 `manifest_v2/benchmarks/`，不进入正式 collection。

## GPU Phase A

每张 GPU 启动一个 shard worker；以下示例为16张独立 GPU/pod：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_proxy_reliability_worker.py \
  --run-matrix results/proxy_reliability_gate0/manifest_v2/run_matrix.jsonl \
  --stage phase_a_signal_discovery \
  --shard-index 0 \
  --num-shards 16 \
  --device cuda
```

其余 worker 使用 shard index 1–15。worker 默认要求 CUDA、可断点重跑，
已完成且 fingerprint 相同的 observation 会直接复用。

Phase A 收集：

```powershell
python scripts/collect_proxy_reliability_observations.py `
  --run-matrix results/proxy_reliability_gate0/manifest_v2/run_matrix.jsonl `
  --stage phase_a_signal_discovery `
  --require-complete
```

完整 Phase C 后去掉 `--stage`，再运行正式 analyzer。Phase A/B 输出只能
用于阶段性去留决策，不能表述为完整 Gate 0 pass。
