# 正式实验与证据流转契约

本文件规定后续实验必须通过的六道门。任何一道门未通过，相关结论保持
`PENDING`、`NOT_MEASURED` 或 `not_ready`，不得用历史结果补齐。

## G1 数据隔离与嵌套搜索

正式 NKSID 搜索只能使用 `train + inner_val`。外层测试折不参与架构选择和
最佳 epoch 选择。固定架构完成 5 折×3 种子只能评价该架构；只有每个外层单元
都独立搜索并通过嵌套选择清单校验，才能令
`nas_generalization_claimable=true`。

## G2 候选身份与指标绑定

候选身份由 `arch_id + architecture_sha256` 共同确定，文件名不是证据。分类、
PSNR/SSIM、HLS/Route、板级分类和功率记录必须携带相同候选身份，并绑定源文件
SHA256。

使用：

```powershell
python scripts/audit_final_candidate_evidence.py `
  --manifest <final_candidate_evidence_manifest.json>
```

审计器会拒绝跨候选拼接。例如 `rl_arch_135` 的分类指标不能与
`rl_arch_60` 的 24.836150 ms 合并成一个联合工作点。

最终候选指标表固定保留：

- 软件 Macro-F1、Top-1；
- 全样本板级 Macro-F1、Top-1；
- PSNR、SSIM，并明确 `paired_clean_target` 或 `input_as_reference`；
- latency、LUT、DSP、BRAM；
- power、energy；未完成外部仪表测量时保持 `NOT_MEASURED`。

## G3 代理筛选与误删保护

未通过独立校准的资源/时延代理只能用于排序和奖励，不能直接证明候选不可部署。
`defer_uncalibrated_hardware_constraints=true` 时：

- 算子语义/部署策略违规和可精确计算的模型大小仍可硬拒绝；
- 代理预测超限记入 `proxy_constraint_violations`；
- 候选状态记为 `needs_hls`，进入 HLS/Route 复核。

搜索产物中的兼容字段 `feasible=true` 只解释为 `search_admissible`（允许继续
搜索/训练），不解释为 FPGA 已可部署；正式硬件可行性只能由候选级 HLS 与
Full Route 证据给出。

未来启用 `certified_reject` 前，校准报告必须至少给出资源与时延预测误差、排序
相关性、false-infeasible（错误判不可行）率及安全裕量；不确定区间继续送 HLS。

## G4 搜索方法公平预算

RL、Random、Aging Evolution 和 Proxyless 的正式比较必须使用完全相同的：

- 数据折、种子和 `frozen_inner` 划分；
- 候选空间及算子准入策略；
- 候选评估次数；
- 每个候选的代理训练 epoch 和 batch size；
- 分类选择指标、Pareto 目标和硬件约束；
- 硬件代价模型及版本。

`scripts/compare_search_methods.py` 的正式 CLI 默认要求四种方法齐全，并要求每个
运行声明同一个 `search.formal_budget`：

```yaml
search:
  formal_budget:
    candidate_evaluations: 200
    model_train_epochs_per_evaluation: 3
```

实际候选数必须等于声明预算。GPU-hours 作为独立计算成本结果报告，不得用更高
计算预算换来的结果冒充搜索策略优势。

当前 Proxyless 路径属于 `weight_sharing_supernet`（权重共享超网），而 RL、
Random、Aging 的现有路径属于 `independent_candidate_train`（候选独立训练）。
两者训练预算单位并不等价，因此比较审计会令 `same_evaluator_mode=false` 并阻止
正式优劣结论。只有实现共同的独立候选评价器，或预先冻结另一种可交换的计算
预算协议后，Proxyless 才能进入正式四方法结论；不能仅把 `epochs` 数字写成相同
就视为公平。

## G5 板级全样本分类

固定输入 COM5 重复测试只证明传输、状态码、checksum 和时延稳定性，不能报告
完整 NKSID 板级准确率。

正式板级分类必须使用
`hls_lut_builder/board_harness/scripts/run_dynamic_validation.py`，覆盖冻结清单中
每一个样本，并验证状态、CRC、样本 ID 和 INT8 logits。运行还必须提供 deployment
binding，把候选、checkpoint、HLS parity、bitstream 和验证清单按 SHA256 绑定。
缺少绑定时，即使所有样本数值一致，也不能令 `claimable=true`。

## G6 结论分层

正式报告保持以下证据层互不替代：

1. 搜索代理；
2. 充分重训练/冻结协议分类；
3. PSNR/SSIM 图像质量；
4. HLS 与 Full Route；
5. COM5 固定输入稳定性；
6. 全样本板级分类；
7. 外部仪表功率与单次推理能耗。

只有同一候选、同一冻结链路的证据可以合并成候选级联合工作点。嵌套 NAS 的
方法级泛化结果必须单列，不能伪装成某一个部署候选的共同结果。


