# Proxy Reliability Audit（Gate 0）

状态：基础设施与真实数据零成本冒烟已完成；正式 48 架构 × 5 seeds ×
5 outer folds × 6 budgets 证据网格尚未运行，因此当前 Gate 0 为
`not_ready`，不能据此宣称 RL 优于、等于或劣于 Random。

执行更新：原始7200-budget-unit设计作为 v1 历史冻结；正式执行改用
prefix-consistent v2（1200条trajectory、Phase A/B/C 分阶段）。详见
`docs/PROXY_RELIABILITY_AUDIT_V2.md`。

## 1. 要回答的问题

搜索链条写作：

```text
architecture a
  -> classification proxy P_b(a, seed, outer_fold)
  -> search update
  -> shortlist
  -> full-training outer-fold truth T_150(a, seed, outer_fold)
```

Gate 0 先判断两个前提：

1. 完整训练后，架构间差异是否稳定大于 seed、outer fold 与交互噪声；
2. 0/1/3/10/30/150 epoch 的代理是否能保持 150 epoch untouched-outer
   truth 的架构排序，特别是 Top-K 区域。

只有这两个前提通过后，才有方法学理由比较 RL、Random、Proxyless
或相对排序预测器。控制器能否表达架构变量依赖属于后续搜索器审计，
不是 Gate 0 通过的替代条件。

## 1.1 现有 RL 证据与代码风险

当前代码仍有两个独立风险，但它们不能代替代理可靠性实测：

- `Controller.forward(stage_idx, block_idx)` 只根据位置产生 channel、
  depth、kernel、expand、operator 的 logits；没有把已经采样的上游
  architecture choices 作为条件，因此是 position-wise factorized
  policy，不能显式表示跨 stage 决策依赖。
- `RewardFunction.compute_reward` 用搜索迄今观察到的 running maximum
  分别归一化 accuracy、latency、energy、DSP、BRAM、LUT。同一个候选在
  搜索早期和晚期可能得到不同尺度的 reward，REINFORCE 的目标因此
  非平稳。

strict40 旧实验还存在一个额外偏差：RL 重复采到同一候选后，从多次
随机短训中取最佳值，而 Random 覆盖四个候选；二者的 Top-K 不是公平
的架构级比较。它足以否定“RL 已优于 Random”，但不足以证明二者
方法学等价。

还需修正“mobile_anchor 空间极小”这一表述：极小的是 strict40 的
4-candidate plumbing space。当前 semantic-safe mobile_anchor 的已归档
精确基数是 15,728,640（见
`artifacts/hw_surrogate_calibration_v2/probes/probe_manifest.json`）。
空间更大并不会创造代理信号；若 Gate 0 失败，
它只会提高无信息搜索的样本复杂度。

## 2. 冻结设计

- 架构数：48。
- seeds：42、43、44、45、46；所有架构共用。
- outer folds：0–4；使用
  `nksid_outer5fold_inner_contiguous_v1`。
- budgets：0、1、3、10、30、150 epoch。
- truth：仅 150 epoch 单元在 inner-validation 选出最佳 epoch 后，
  对 untouched outer fold 评价一次。
- 1/3/10/30 epoch：只产生 inner-validation proxy，禁止消费 outer
  loader。
- 0 epoch：NASWOT 二值激活核 log-determinant；只使用 inner-training
  calibration batch。
- 正预算训练：独立 exact-budget run、constant LR AdamW、同一 seed
  与数据顺序前缀、同一类别权重规则。预算以外的 recipe 字段必须一致。

协议源：

- `configs/audit/proxy_reliability_gate0.yaml`
- `scripts/generate_proxy_reliability_manifest.py`
- `scripts/run_proxy_reliability_observation.py`
- `scripts/collect_proxy_reliability_observations.py`
- `scripts/analyze_proxy_reliability.py`

## 3. 架构抽样

先从当前 policy-filtered search space 均匀抽取 4096 个去重候选，再按下列
标准化特征做 feasibility quota + maximin 选择：

- latency、DSP、BRAM、LUT；
- stage width、总 depth；
- operator mix；
- 最大资源利用率与 feasibility margin。

选择过程不读取任何 classification proxy 或历史 proxy Top-K。

当前范围限制：部署语义策略把 `denoise` 和 `edge` 标为 paused，原因是
PyTorch 与已接纳 HLS 模板不等价；因此现行 deployment-safe
mobile_anchor 空间只包含 `mbconv` 和 `skip`。正式 manifest 会记录实际
operator coverage 及 quota 重分配。若要审计四算子分类空间，必须另立
一个 classification-only manifest，不能把它的硬件结论并入当前审计。

## 4. 方差分解与 ICC

对每个分类指标拟合平衡三因素交叉设计：

```text
y = μ + A + S + F + A×S + A×F + S×F + residual
```

其中 `A` 为 architecture，`S` 为 seed，`F` 为 outer fold。一格一个观测
时，residual 包含三阶交互及未建模噪声。分析同时输出：

- absolute / relative single-observation ICC；
- 5 seeds、单一 outer fold 的 absolute / relative mean ICC；
- 单一 seed、5 outer folds 的 absolute / relative mean ICC；
- 按实际 5 seeds × 5 folds 平均后的 absolute / relative mean ICC；
- 原始和非负截断后的方差分量；
- 以 architecture 为一级抽样单位的 bootstrap 置信区间。

正式 gate 使用 relative mean ICC；single ICC 仍保留，用于判断单次短训
是否稳定。

## 5. 排序可靠性

对同一 `(architecture, seed, outer_fold, metric)` 将 `P_b` 与
`T_150` 匹配，再按 architecture 汇总。每个 budget 至少报告：

- Spearman ρ；
- Kendall τ-b；
- Pairwise Accuracy；
- Top-5 / Top-10 Recall 及随机期望；
- NDCG@5 / NDCG@10；
- regret@5 / regret@10；
- 95% bootstrap CI。

`regret@K` 在本次 audited set 内定义为：

```text
best T_150 among all audited architectures
-
best T_150 among architectures selected by proxy Top-K
```

bootstrap 先重采样 architecture，再在该 architecture 内重采样 outer
fold 与 seed。架构对只用于描述 concordance，不作为独立重复。

## 6. 预注册判据

`macro_f1` 的正式 Gate 0 要求：

- architecture ≥ 40、seed ≥ 5、outer fold ≥ 5；
- 0/1/3/10/30/150 六个 budget 都有完整匹配网格；
- relative mean ICC ≥ 0.60；
- 至少一个 budget 同时达到 Kendall τ-b ≥ 0.30、
  Pairwise Accuracy ≥ 0.65、regret@5 ≤ 0.02。

正式判定使用 bootstrap 95% CI 的保守边界：ICC、τ-b、Pairwise
Accuracy 使用下界，regret@5 使用上界；点估计与完整 CI 同时报告。
少于预注册的 2000 次 bootstrap 时仍判为 `not_ready`。

结果状态：

- `not_ready`：网格缺失或格式/recipe/outer 使用不合法；
- `fail_no_stable_architecture_signal`：真实性能主要由噪声决定；
- `fail_no_usable_proxy_budget`：有架构差异，但代理排序不可用；
- `pass`：可进入搜索算法比较；这本身不证明 RL 优越。

## 7. 硬件可靠性保持独立

硬件表单独比较 analytic proxy 与 HLS/post-route/COM5 truth，输出每项
资源的 bias、MAE、RMSE、MAPE、ρ、τ-b，以及 feasibility confusion
matrix 和 Pareto precision/recall。truth 来源字段为必填：

- `truth_latency_source`；
- `truth_resource_source`；
- `truth_feasibility_source`。

COM5 latency、post-route resources、分类 macro_f1/top1 和 power
不得合并为一个标量奖励。没有外部功率测量时，power/energy 仍是
`not measured`。

## 8. 执行命令

生成正式冻结 manifest：

```powershell
python scripts/generate_proxy_reliability_manifest.py
```

调度器中每个任务只运行一个 work index：

```powershell
python scripts/run_proxy_reliability_observation.py `
  --run-matrix results/proxy_reliability_gate0/manifest_v1/run_matrix.jsonl `
  --work-index 0
```

收集完整性：

```powershell
python scripts/collect_proxy_reliability_observations.py `
  --run-matrix results/proxy_reliability_gate0/manifest_v1/run_matrix.jsonl `
  --require-complete
```

正式分析：

```powershell
python scripts/analyze_proxy_reliability.py `
  --classification-csv results/proxy_reliability_gate0/manifest_v1/collected/classification_observations.csv `
  --hardware-csv artifacts/proxy_reliability_gate0/hardware_observations.csv `
  --output-dir results/proxy_reliability_gate0/manifest_v1/analysis `
  --require-pass
```

`--require-complete` 和 `--require-pass` 未满足时返回非零退出码，供调度器
或 CI 阻断后续搜索比较。
