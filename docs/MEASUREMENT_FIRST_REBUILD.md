# HW-NAS 测量优先重建

本文件记录“先修测量仪器，再做优化”的可执行入口与证据边界。当前权威
机器可读状态为
`artifacts/measurement_first_rebuild/status.json`。软件门禁已实现不等于
实验已经完成；缺失的训练、csynth、full-route、COM5 或功率计数据一律保持
`PENDING` / `NOT_MEASURED`。

## 当前门禁

| 门禁 | 当前状态 | 判定 |
|---|---|---|
| G0 冻结协议 | PASS | 唯一正式入口为 `run_eval_protocol.py` |
| G1 精度基线 | PASS | 45/45 个 fold/seed 任务通过完整性与来源检查 |
| G2 硬件测量 | PENDING | 4 个独立探针和候选 HLS 100% 覆盖尚未完成 |
| G3 搜索 | FROZEN | 语义安全空间精确规模为 15,728,640 |
| G4 INT8 板级闭环 | PENDING | PTQ/QAT、HLS parity、全验证集板测均须通过 |
| 功耗 | NOT_MEASURED | 未通过真实外部仪表 CSV 验收 |
| G5 声呐消融 | PAUSED | `denoise`/`edge` 仍不得进入新搜索 |

运行 `python scripts/audit_measurement_first_gates.py` 可刷新该表对应的 JSON
和 Markdown 状态，不会把缺失证据补成通过。

## G0—G1：分类评估

冻结合同位于 `configs/evaluation/nksid_frozen_protocol_v1.yaml`。正式结果必须
同时满足：

- outer folds 为 0–4，seeds 为 42、43、44；
- outer validation 只用于最终报告；
- checkpoint、数据清单、split、配置和代码提交均可追溯；
- `rl_arch_135` 明确标记为 `legacy_fold0_selected`，只解释为冻结历史架构基准。

三个基线已完成；以下入口保留用于同一冻结协议下的可恢复复现：

```powershell
python scripts/run_g1_baselines.py
```

该入口规划 3×15=45 个任务，默认 AMP、batch size 8、gradient accumulation
4。预训练权重加载失败会直接报错，不会静默回退。当前权威摘要为：

- scratch：`results/protocol/g1_clean_20260718/g1_mobilenet_v2_scratch_v2/protocol_summary.json`；
- pretrained：`results/protocol/g1_clean_20260711/g1_mobilenet_v2_grayscale_imagenet/protocol_summary.json`；
- legacy-selected NAS：`results/protocol/g1_clean_20260711/g1_rl_arch_135_legacy_selected/protocol_summary.json`。

重新完成同口径运行后可执行：

```powershell
python scripts/analyze_protocol_baselines.py `
  --scratch-run results/protocol/g1_mobilenet_v2_scratch `
  --pretrained-run results/protocol/g1_mobilenet_v2_grayscale_imagenet `
  --nas-run results/protocol/g1_rl_arch_135_legacy_selected
```

## G2：分层硬件测量

`calibration_v2.json` 分开保存 analytic→HLS、HLS→post-route 和完整网络
composed-HLS→COM5 证据。旧单倍率仅保留为 `legacy_calibration_v1`，正式配置
不再使用 `latency_scale=7.155` 等倍率淘汰候选。

四个冻结探针位于
`artifacts/hw_surrogate_calibration_v2/probes/`。候选 HLS 链使用：

```powershell
python scripts/run_candidate_hls_shortlist.py `
  --candidate artifacts/hw_surrogate_calibration_v2/probes/calibration_probe_01.candidate.json `
  --candidate artifacts/hw_surrogate_calibration_v2/probes/calibration_probe_02.candidate.json `
  --candidate artifacts/hw_surrogate_calibration_v2/probes/calibration_probe_03.candidate.json `
  --candidate artifacts/hw_surrogate_calibration_v2/probes/calibration_probe_04.candidate.json `
  --output-dir results/calibration_probe_hls_shortlist `
  --run-synthesis
```

所有非 skip 层必须有真实 csynth 报告；任何缺失都令
`evidence_complete=false`，禁止解析值回填。route/COM5 完成后，按
`full_network_evidence.schema.json` 写入
`full_network_evidence.jsonl`，再运行：

```powershell
python scripts/build_hw_calibration_v2.py
```

只有独立探针误杀率为 0、P90 APE≤25%，且 latency 的 Spearman≥0.8 时，
相应指标才可用于区间硬筛；否则自动 `pass-through → HLS`。

## G3：搜索冻结

`run_search.py` 对真实 NKSID 搜索执行仓库级门禁。G2、G4 与独立的阶段3
重规划批准文件未同时通过前，RL、Random、Proxyless 不能启动新的正式搜索；
dummy 运行仅允许作为 smoke，永远不可声明。历史方法对比保持
`legacy exploratory evidence`。

## G4：INT8、HLS 与板级准确率

第一版对象被固定为 `rl_arch_193 / fold1 / seed42`。软件精度入口：

```powershell
python run_eval_int8.py `
  --protocol-run results/protocol/<rl_arch_193_protocol_run> `
  --candidate-path hls_lut_builder/board_harness/results/pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp/candidates/007_rl_arch_193.candidate.json
```

PTQ 的 macro_f1 与 top1 下降均不得超过 0.02；失败时自动进入 20 epoch
低学习率 QAT，QAT 下降超过 0.05 则阻断部署。

逐层整数模拟器与 HLS testbench 的真实、边界和随机张量记录写为 JSONL，
并通过：

```powershell
python scripts/audit_int8_hls_parity.py --records <parity_records.jsonl>
```

只有 parity 为 PASS，`run_dynamic_validation.py` 才接受全验证集板测。动态
UART v1 支持 `LOAD_RUN`、`RUN_REPEAT`、`PING`、CRC32、重试和断点续跑。
50,176 字节输入使用 Xilinx XPM block RAM；隔离综合实测为 16 个 RAMB36，
不是 LUT 存储。该结果只证明输入缓存实现，不代替完整网络 route。

## 功耗

`run_power_repeat.py` 先在测量区间外完成 UART 上传，再连续执行至少 1,000
次推理。真实 CSV 必须通过 `scripts/import_power_measurement.py` 的 3 次 idle
和 3 次 active 验收。在此之前 power/energy 保持 `not measured`，至少三个
候选在同一仪表与协议下通过后才可重返 Pareto。

## G5：声呐算子

详细执行方案（四路消融配置、折叠部署证据链、指标协议）见
[SONAR_OPERATOR_G5_EXPERIMENT_PLAN.md](SONAR_OPERATOR_G5_EXPERIMENT_PLAN.md)。

`scripts/audit_sonar_operator_gate.py` 同时检查：

- 软件/HLS 使用同一 INT8 规范并完成权重导出；
- 真实、边界、随机输入逐元素零差异；
- MBConv control 的输出 shape 相同，参数量和 MACs 误差均≤5%；
- 四组实验均完成 5 folds×3 seeds；
- 配对分层 bootstrap 与三项 Holm 校正；
- macro_f1 实际增益和 HLS route 可行性。

当前模板审计结果为 `BLOCKED`，因此历史简化 `denoise`/`edge` 的 route/COM5
不能作为训练期算子已部署的证据。
