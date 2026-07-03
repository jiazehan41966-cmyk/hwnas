# 搜索训练指标链路审计归档

归档时间：2026-05-17

审计状态：已完成静态审计；未修改源码；未运行训练、HLS、Vivado。

前置引用：本审计承接 `docs\audit_entry_config.md` 中确认的入口与配置闭环，以及 `docs\audit_candidate_hardware.md` 中确认的 `CandidateMetrics` / `CostEstimate` 口径。

## 搜索训练数据流图

```text
config YAML + CLI args
  -> run_search.py 解析 dataset/search/project/training
     E:\1\hwnas\hwnas\run_search.py:222
     E:\1\hwnas\hwnas\run_search.py:256
  -> build_constraints() / build_search_space() / build_cost_estimator()
     E:\1\hwnas\hwnas\run_search.py:302
     E:\1\hwnas\hwnas\run_search.py:356
  -> create_data_pipeline()
     E:\1\hwnas\hwnas\run_search.py:431
     E:\1\hwnas\hwnas\run_search.py:445
  -> create_searcher(random / rl / proxyless)
     E:\1\hwnas\hwnas\run_search.py:471
     E:\1\hwnas\hwnas\run_search.py:490
  -> searcher 产生候选并调用 estimator
     random:   E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:178
     rl:       E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1731
     proxyless:E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:584
  -> train_model() / proxyless argmax eval
     E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:237
     E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:351
  -> CandidateMetrics
     E:\1\hwnas\hwnas\src\hwnas_fpga\interfaces.py:31
     E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:82
  -> ExperimentTracker artifacts
     candidates.jsonl / candidates/*.json:
       E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:190
     best_candidate.json / best_model.pt:
       E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:229
     candidates.csv / summary.json:
       E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:262
  -> feasible_candidates -> Pareto
     E:\1\hwnas\hwnas\run_search.py:620
     E:\1\hwnas\hwnas\run_search.py:630
```

## 指标传播表

| 指标名 | 产生位置 | 写入位置 | 用于排序位置 | 输出 artifact / 证据 |
|---|---|---|---|---|
| `macro_f1` | `evaluate_classifier()` 由混淆矩阵计算 macro-F1：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:151`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:174` | random 写 `candidate.metrics.macro_f1`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:236`；RL 写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1354`；proxyless 写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:435` | `selection_metric=macro_f1/f1` 在训练和候选排序中解析：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:222`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:38` | `candidates.jsonl/json/csv`：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:190`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:302` |
| `top1` / `accuracy` | `top1` 由混淆矩阵对角线计算：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:173`；`accuracy` alias 映射到 `top1`：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:224` | `top1` 与 `accuracy` 同时存在；random 将 `train_model()` 返回值写到 `metrics.accuracy`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:222`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:235`；RL 同理：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1325`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1353`；proxyless 将 selection score 写入 `accuracy`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:431`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:434` | random 用 `_candidate_selection_score()`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:38`；Pareto 默认目标固定从 `accuracy` 开始：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:111` | best/summary 打印读取 `metrics.accuracy`：`E:\1\hwnas\hwnas\run_search.py:683`, `E:\1\hwnas\hwnas\run_search.py:716` |
| `weighted_f1` | `evaluate_classifier()` 计算：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:151`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:175` | random/RL/proxyless 分别写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:238`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1355`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:436` | alias 支持：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:229`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:344` | CSV 字段：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:304`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:330` |
| `top5` | `_topk_accuracy_hits()` 与 `evaluate_classifier()` 产生：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:143`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:215` | random/RL/proxyless 分别写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:240`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1357`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:438` | 不参与默认排序；仅作为评估记录 | CSV 和控制台输出：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:310`, `E:\1\hwnas\hwnas\run_search.py:690` |
| `latency_ms` / `LUT` / `DSP` / `BRAM` / `power_w` / `energy_mj` | `CostEstimate` 定义并由 estimator 产生：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:48`, `E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:67` | `CostEstimate.to_candidate_metrics()` 写硬件指标：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:82`；RL 手写同一字段：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1309`；proxyless 重新封装：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:433` | RL reward 使用 latency/energy/DSP/BRAM/LUT：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:525`；Pareto 目标按 weights/constraints 添加：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:125` | `baseline.json`、`candidates.csv`、`pareto_selection.json`：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:175`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:302`, `E:\1\hwnas\hwnas\run_search.py:645` |
| `feasibility` | `BaseSearcher.check_feasibility()`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:75`；RL 自有实现：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1391`；probe 以 `violations` 判定：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\probe.py:53` | searcher 三列表写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:247`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1368`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:455` | Pareto 只使用 `feasible_candidates`：`E:\1\hwnas\hwnas\run_search.py:620`, `E:\1\hwnas\hwnas\run_search.py:630` | JSONL 有 `feasible` 字段：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:199`；CSV 无该字段：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:304` |

## 搜索器差异表

| 搜索器 | 采样 | 训练/评估 | reward/loss | feasibility | 记录 |
|---|---|---|---|---|---|
| random | `SearchSpace.sample(require_feasible=True, max_feasible_attempts=32, prefer_lightweight=True)`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:178` | 可行后调用 `train_model()`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:218`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:222` | 无 reward；按 `_candidate_selection_score()` 选 best：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:297`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:305` | 继承 `BaseSearcher.check_feasibility()`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:75` | 每候选写 JSONL，并更新 search_state：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:288`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:320` |
| RL | controller 生成架构，不要求先采到可行：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1731` | 可行后调用 `train_model()`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1320`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1325` | `RewardFunction.compute_reward()` 以 accuracy/latency/energy/DSP/BRAM/LUT 计算；不可行惩罚：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:466`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:490`；更新 controller：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1755` | 自己复制 feasibility 逻辑，未继承 Base：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1391` | 记录 reward/loss/baseline/checkpoint：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1761`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1778` |
| proxyless | supernet warmup/search 后提取 argmax 架构：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:506`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:584` | 不调用 `train_model()`；内部 `_evaluate_argmax_summary()` 计算 top1/top5/F1：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:351`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:413` | weight loss + arch loss + hardware penalty：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:558`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:576` | 继承 Base，通过 `_build_candidate_from_supernet()` 检查：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:422`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:430` | 每 epoch 记录；可行才更新 best，但无可行时会 fallback：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:595`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:656` |

## Feasible / Infeasible 数据流

| 状态 | 数据流 | 证据 |
|---|---|---|
| feasible candidate | estimator -> feasibility true -> 可训练或已完成 proxyless epoch -> `feasible_candidates` -> best/Pareto/summary | random：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:219`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:249`；RL：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1322`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1369`；proxyless：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:462`；Pareto：`E:\1\hwnas\hwnas\run_search.py:620` |
| infeasible candidate | estimator -> feasibility false -> random/RL 不训练；proxyless 已训练 supernet 但候选判不可行 -> `infeasible_candidates` -> candidates artifacts / summary count | random：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:218`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:251`；RL：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1320`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1371`；proxyless：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:465`；summary：`E:\1\hwnas\hwnas\run_search.py:728` |
| Pareto 输入 | 只用 feasible | `E:\1\hwnas\hwnas\run_search.py:620`, `E:\1\hwnas\hwnas\run_search.py:630`, `E:\1\hwnas\hwnas\run_search.py:631` |
| CSV / candidates.json | 包含所有 evaluated candidates，但无 feasible 字段 | finalize 写 all candidates：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:273`；CSV 字段无 `feasible`：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:304` |

## 训练与重训练一致性

| 项目 | train_model / evaluate_classifier | retrain_architecture | 审计结论 |
|---|---|---|---|
| `class_weights` | `train_model()` 接收并转到 device，构造 weighted CE：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:245`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:254`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:257` | retrain 也转到 device：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:69`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:80` | loss 权重一致 |
| `early_stopping` | 按 `selection_metric` 的 best score early stopping：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:299`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:305` | 只按 `val_acc` 改善：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:133`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:144` | selection 口径不一致 |
| `best_epoch` | 写入 history：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:271`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:302` | 写入 metrics：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:97`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:137` | 字段存在，但位置和 schema 不一致 |
| `best_eval` | 保存完整 `loss/top1/top5/macro_f1/weighted_f1/num_samples`：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:303`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:314` | 无 `best_eval`，仅 `best_val_acc/best_val_loss`：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:97`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:149` | macro_f1/top5 无法闭环复核 |
| `top5` | `evaluate_classifier()` 支持 `topk=5`：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:188`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:215` | retrain 的 `evaluate_model()` 只返回 loss/acc：`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:32`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:55` | 不一致 |

## NKSID 数据与 seed 传播

| 字段 | run_search 传入 | run_retrain 传入 | dataset/runtime 消费 | 结论 |
|---|---|---|---|---|
| `data_dir` | `E:\1\hwnas\hwnas\run_search.py:249`, `E:\1\hwnas\hwnas\run_search.py:434` | `E:\1\hwnas\hwnas\run_retrain.py:91`, `E:\1\hwnas\hwnas\run_retrain.py:122` | NKSID 必须有 data_dir：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:362` | 正确传入 |
| `k-fold/fold/use_kfold` | `E:\1\hwnas\hwnas\run_search.py:248`, `E:\1\hwnas\hwnas\run_search.py:250`, `E:\1\hwnas\hwnas\run_search.py:439` | `E:\1\hwnas\hwnas\run_retrain.py:90`, `E:\1\hwnas\hwnas\run_retrain.py:92`, `E:\1\hwnas\hwnas\run_retrain.py:127` | `create_nksid_dataloaders()` 接收并传入 `NKSIDDataset`：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:672`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:753` | 正确传入 |
| `valid_size/split_seed` | `E:\1\hwnas\hwnas\run_search.py:251`, `E:\1\hwnas\hwnas\run_search.py:252`, `E:\1\hwnas\hwnas\run_search.py:443` | `E:\1\hwnas\hwnas\run_retrain.py:93`, `E:\1\hwnas\hwnas\run_retrain.py:94`, `E:\1\hwnas\hwnas\run_retrain.py:131` | random valid split 用 `split_seed + fold`：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:733` | 正确传入 |
| `num_workers` | `E:\1\hwnas\hwnas\run_search.py:247`, `E:\1\hwnas\hwnas\run_search.py:440` | `E:\1\hwnas\hwnas\run_retrain.py:89`, `E:\1\hwnas\hwnas\run_retrain.py:128` | DataLoader 消费：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:776`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:784` | 正确传入 |
| `image_size` | `E:\1\hwnas\hwnas\run_search.py:244`, `E:\1\hwnas\hwnas\run_search.py:436` | `E:\1\hwnas\hwnas\run_retrain.py:86`, `E:\1\hwnas\hwnas\run_retrain.py:124` | transform resize：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:124`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:174` | 正确传入 |
| `input_channels` | `E:\1\hwnas\hwnas\run_search.py:245`, `E:\1\hwnas\hwnas\run_search.py:438` | `E:\1\hwnas\hwnas\run_retrain.py:87`, `E:\1\hwnas\hwnas\run_retrain.py:126` | dummy 分支使用：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:380`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:385`；NKSID dataset 参数名是 `output_channels`，默认 1：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:249`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:258` | NKSID 未正确传入 input_channels |
| `class_weights` | 写入 dataset summary 并传 searcher：`E:\1\hwnas\hwnas\run_search.py:467`, `E:\1\hwnas\hwnas\run_search.py:573` | 传入 retrain：`E:\1\hwnas\hwnas\run_retrain.py:183` | NKSID 计算权重：`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:509`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:749` | 搜索和重训均使用 |

## RL resume 状态

已恢复状态：

| 状态 | 证据 |
|---|---|
| `evaluated_candidates/feasible_candidates/infeasible_candidates` | `E:\1\hwnas\hwnas\run_search.py:106`, `E:\1\hwnas\hwnas\run_search.py:117` |
| `best_candidate` | `E:\1\hwnas\hwnas\run_search.py:119` |
| `best_reward` | `E:\1\hwnas\hwnas\run_search.py:120` |
| `baseline` | `E:\1\hwnas\hwnas\run_search.py:121` |
| controller 参数 | `E:\1\hwnas\hwnas\run_search.py:122` |
| optimizer 参数 | `E:\1\hwnas\hwnas\run_search.py:123` |
| reward stats 由候选记录重算 | `E:\1\hwnas\hwnas\run_search.py:125`, `E:\1\hwnas\hwnas\run_search.py:154` |
| next episode | `E:\1\hwnas\hwnas\run_search.py:156` |

未恢复状态：

| 状态 | 证据 |
|---|---|
| `architecture_visit_counts` | 初始化在 `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:606`，resume 函数未写该字段：`E:\1\hwnas\hwnas\run_search.py:80` 到 `E:\1\hwnas\hwnas\run_search.py:157` |
| `current_episode` | 初始化在 `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:605`，resume 只返回 `resumed_episode`：`E:\1\hwnas\hwnas\run_search.py:156` |
| Python/NumPy/Torch RNG state | checkpoint 只保存 controller/optimizer/baseline/reward：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1778`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1787` |
| `last_training_history/last_cost_estimate` | 初始化在 `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:655`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:657`，resume 未恢复：`E:\1\hwnas\hwnas\run_search.py:80` 到 `E:\1\hwnas\hwnas\run_search.py:157` |

## Pareto 目标构建

| 检查点 | 证据 | 结论 |
|---|---|---|
| 默认目标 | `objectives=["accuracy"]`，`directions=["max"]`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:117` | 默认仍依赖 `metrics.accuracy`，会受 selection score 语义影响 |
| `objective_weights` 扩展 latency/energy/resource/power/memory/offchip | `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:125` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:150` | 目标构建覆盖 latency、energy、DSP、BRAM、LUT、power、bandwidth、offchip |
| constraints 也会触发目标加入 | `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:125`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:133`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:139` | 与 constraints 有联动 |
| selector 使用点 | `E:\1\hwnas\hwnas\run_search.py:616`, `E:\1\hwnas\hwnas\run_search.py:625`, `E:\1\hwnas\hwnas\run_search.py:630` | run_search 使用同一 objectives/directions 计算 front、selector、rank |
| selector rank tie-break | `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:441`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\pareto.py:447` | rank 选择同排时仍按 `metrics.accuracy`，不是显式 selection_metric |

## search_space_probe 口径

| 检查点 | 证据 | 结论 |
|---|---|---|
| 脚本说明 | `Sample architectures from a search space and report FPGA feasibility`：`E:\1\hwnas\hwnas\run_search_space_probe.py:2` | 定位为硬件可行性抽样 |
| 不加载数据、不训练 | probe main 只 build constraints/search_space/estimator 后调用 `probe_search_space()`：`E:\1\hwnas\hwnas\run_search_space_probe.py:137`, `E:\1\hwnas\hwnas\run_search_space_probe.py:175` | 不产生训练指标 |
| probe 内部采样不要求可行，直接估计 cost | `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\probe.py:38`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\probe.py:47` | 只统计 feasibility 和硬件指标 |
| artifact 使用 SearchCandidate | `E:\1\hwnas\hwnas\run_search_space_probe.py:186`, `E:\1\hwnas\hwnas\run_search_space_probe.py:198` | 可能被误读成训练搜索候选 |
| 明确 mode | `best_candidate=None`，`mode=space_probe`：`E:\1\hwnas\hwnas\run_search_space_probe.py:210`, `E:\1\hwnas\hwnas\run_search_space_probe.py:249` | 有 schema 证据表明不是训练结果 |

## 发现列表

### P1

现象：`CandidateMetrics.accuracy` 实际保存的是 selection score；当 `selection_metric=macro_f1` 或 `weighted_f1` 时，`accuracy` 不再等于 top1。
证据：`train_model()` 用 `_resolve_selection_score()` 返回 best score，`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:222`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:299`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:325`；random 写 `candidate.metrics.accuracy = accuracy`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:235`；RL 写 `accuracy=accuracy`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1353`；proxyless 写 `accuracy=float(selection_score)`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:431`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:434`。
影响的指标或 artifact：`accuracy`、`top1`、`macro_f1`、`best_candidate.json`、`summary.json`、`pareto_selection.json`、`candidates.csv`。
具体动作：在 `CandidateMetrics` 中增加 `selection_metric/selection_score`；`accuracy` 或 `top1` 固定表示 top1；best 选择、Pareto tie-break、控制台打印改为显式读取 `selection_score` 或 `selection_metric`。

### P1

现象：random/proxyless 使用 `BaseSearcher.check_feasibility()`，RL 自己复制一份，且 RL 的直接约束检查缺少 `max_energy_mj` 和 `max_model_size_mb` 分支。
证据：Base 检查包含 `max_energy_mj` 和 `max_model_size_mb`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:84`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:90`；RL 自有检查从 `max_latency_ms` 跳到 `max_dsp`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1391`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1395`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1399`；约束字段来源包含 energy/model size，`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:63`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:68`。
影响的指标或 artifact：`feasibility`、`infeasible` count、RL reward、Pareto 输入、`summary.json`。
具体动作：抽出单一 feasibility helper 或让 RL 继承/复用 Base；直接检查和 `cost_estimate.violations` 保持同一字段集合。

### P1

现象：RL 的 `infeasible_penalty_mode=violation_ratio` 没有传入实际 violation ratio，当前会退化成默认 1.0。
证据：`compute_reward()` 支持 `constraint_violation_ratio`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:466`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:475`；实际 ratio 计算函数存在，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:682`；搜索时调用 reward 未传该参数，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1744`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\rl_searcher.py:1753`。
影响的指标或 artifact：RL reward、controller loss、`candidates.jsonl.extra.reward`、resume 后 reward stats。
具体动作：在 RL 搜索 reward 调用中传入 `constraint_violation_ratio=self._compute_constraint_violation_ratio(self.last_cost_estimate)`，并在 `extra` 中记录该 ratio。

### P2

现象：NKSID 的 `input_channels` 从配置读入并传给 `create_data_pipeline()`，但 NKSID 数据集实际使用 `output_channels` 默认值 1，配置的 3 通道不会生效。
证据：run_search 读取 `input_channels` 并传 pipeline，`E:\1\hwnas\hwnas\run_search.py:245`, `E:\1\hwnas\hwnas\run_search.py:438`；run_retrain 同样传入，`E:\1\hwnas\hwnas\run_retrain.py:87`, `E:\1\hwnas\hwnas\run_retrain.py:126`；runtime NKSID 分支调用 `create_nksid_dataloaders()` 未传 `input_channels`，`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:366`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:375`；NKSIDDataset 默认 `output_channels=1`，`E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:249`, `E:\1\hwnas\hwnas\src\hwnas_fpga\data\dataset.py:258`。
影响的指标或 artifact：reproducibility、模型输入通道、搜索/重训一致性、`dataset_summary.json`。
具体动作：给 `create_nksid_dataloaders()` 增加 `input_channels` 或 `output_channels` 参数，并在所有 `NKSIDDataset(...)` 构造点传入。

### P2

现象：`retrain_architecture()` 不产生 macro_f1、weighted_f1、top5、best_eval；只按 `val_acc` early stopping，与搜索评估 schema 不一致。
证据：搜索训练保存完整 `best_eval`，`E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:283`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\trainer.py:303`；重训 `evaluate_model()` 只返回 loss/acc，`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:32`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:55`；重训 metrics 只有 best/final val acc/loss，`E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:149`, `E:\1\hwnas\hwnas\src\hwnas_fpga\training\retrain.py:156`。
影响的指标或 artifact：`retrain_summary.json`、`final_best_model.pt`、macro_f1/top1/top5 对比闭环。
具体动作：重训复用 `evaluate_classifier()`，输出与搜索一致的 `best_eval/best_epoch/top1/top5/macro_f1/weighted_f1`。

### P2

现象：infeasible 候选会进入 `candidates.json` 和 `candidates.csv`，但聚合 JSON/CSV 没有 `feasible` 字段；只有 JSONL 单条记录有。
证据：record_candidate 写 `feasible`，`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:190`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:201`；finalize 写所有 candidates，`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:273`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:275`；CSV 字段列表无 `feasible`，`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:304`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:319`。
影响的指标或 artifact：`candidates.json`、`candidates.csv`、`summary.json` 可复查性。
具体动作：聚合候选和 CSV 增加 `feasible`、`violations`；或额外写 `feasible_candidates.json` 与 `infeasible_candidates.json`。

### P2

现象：proxyless 若没有可行候选，会返回不可行 best candidate。
证据：可行候选才更新 best，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:597`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:600`；无 best 时 fallback 到所有 evaluated，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:656`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:660`；run_search 将返回的 `best_candidate` 写 summary，`E:\1\hwnas\hwnas\run_search.py:728`, `E:\1\hwnas\hwnas\run_search.py:734`。
影响的指标或 artifact：`summary.json.best_candidate`、`best_candidate.json`、后续 `run_retrain.py` 输入风险。
具体动作：没有可行候选时返回 `None`；或在 best artifact 中显式写 `feasible=false` 并让 `run_retrain.py` 默认拒绝消费不可行候选。

### P3

现象：`search_space_probe` 只做硬件抽样，但复用 `SearchCandidate` 和 candidates artifact，容易被误读为训练搜索结果。
证据：probe 内部不训练，只 `sample()` 和 `estimate()`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\probe.py:38`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\probe.py:47`；脚本构造 `SearchCandidate` 并写 `record_candidate()`，`E:\1\hwnas\hwnas\run_search_space_probe.py:186`, `E:\1\hwnas\hwnas\run_search_space_probe.py:198`；`history=None`，`mode=space_probe`，`best_candidate=None`，`E:\1\hwnas\hwnas\run_search_space_probe.py:202`, `E:\1\hwnas\hwnas\run_search_space_probe.py:216`, `E:\1\hwnas\hwnas\run_search_space_probe.py:254`。
影响的指标或 artifact：`candidates.jsonl`、`candidates.csv`、`summary.json`。
具体动作：在 probe artifact schema 增加 `artifact_type=space_probe` 和 `trained=false`；或把 CSV 命名为 `probe_candidates.csv`。

## 无法静态确认项

- 未运行正式训练，因此无法确认具体 run 中 `macro_f1/top1/top5` 的数值正确性；需要 `candidates.jsonl` 中包含 `history.best_eval` 的实际运行 artifact。
- 未运行 RL resume，因此无法动态确认 controller/optimizer/baseline 恢复后 reward 曲线是否连续；静态可确认的恢复字段见上文 RL resume 表。
- 未检查历史 results 目录，因此不判断旧 run artifact 是否符合当前 schema。
- 未审计 HLS/Vivado 生产链，不判断 formal LUT 内容真实性。

## 最小复现实验建议表

| 命令 | 预期检查项 | 不需要长训练的原因 |
|---|---|---|
| `python -m py_compile run_search.py run_retrain.py run_search_space_probe.py src\hwnas_fpga\search\searcher.py src\hwnas_fpga\search\rl_searcher.py src\hwnas_fpga\search\proxyless_searcher.py src\hwnas_fpga\training\trainer.py src\hwnas_fpga\training\retrain.py src\hwnas_fpga\data\dataset.py` | 语法/导入层面闭环 | 只编译，不训练；本次审计已执行通过。 |
| `python -m pytest tests\test_search.py::SearchFactoryTests::test_build_pareto_objectives_from_weights_and_constraints -q` | `objective_weights` 与 constraints 是否共同构建 Pareto objectives | 单元测试只构造对象。 |
| `python -m pytest tests\test_search.py::SearchFactoryTests::test_rl_searcher_updates_controller_once_with_exploration_bonus -q` | RL reward、exploration bonus、controller update 次数 | 使用 mock/fixed evaluate，不跑训练。 |
| `python -m pytest tests\test_search_space_probe_cli.py::SearchSpaceProbeCliTests::test_probe_cli_generates_summary -q` | probe 生成 `probe_summary.json`，并确认 `total_samples` | 8 个硬件样本，无训练。 |
| `python -m pytest tests\test_dataset.py::NKSIDDatasetTests -q` | NKSID root、loader 基本行为 | 使用测试临时小数据。 |
| `python -m pytest tests\test_retrain.py::RetrainWorkflowTests::test_load_architecture_and_retrain -q` | retrain artifact 与短训练 smoke | 测试内小模型/小数据，不是正式长训练。 |
