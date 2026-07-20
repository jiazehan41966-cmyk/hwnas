# G2 A8 到 T6 准入审计（中文伴随档案，2026-07-17）

- 英文原件：`g2_a8_to_t6_admission_20260717.md`；SHA256：`a2ad0e0103d14e2f64e642ab4d2a616baa3924d009601ea49369803263031bdc`。

## 决定

修正后的 8 条 A8 记录可以作为 G2 calibration（校准）以及完整网络 route/COM5 证据，但对 T6 正式样本数的增量为 `0`。禁止把它们复制到 T6 完整网络 HLS/route ledger，并伪装成每行都具有完整网络 HLS synthesis。

## 来源 cohort

- Ledger：`results/experiment_cycle_20260712_v2/calibration_v2_a8_corrected/full_network_evidence.jsonl`。
- SHA256：`11ecb6c3872eaef2f88066e1a6c6c527044978e3102fb471f9a3d9d7b4891d52`。
- 审计状态：`PASS_READY_FOR_CALIBRATION_REBUILD_CORRECTED`。
- 8 行，包括 4 个 fit 与 4 个 independent probe；8/8 个唯一 architecture/fingerprint。
- 只有一个架构族：`mainline_mbconv_skip`。
- A8 cohort 的 source freeze：PASS 508/508。

## T6 证据分类

| 字段或不变量 | A8 证据 | T6 准入 |
|---|---|---|
| Canonical architecture hash | 有 | 可用 |
| 候选专用完整网络 route | 有，timing-clean 且有 routed bitstream | 仅作 route-side 辅助证据 |
| Route LUT/DSP/BRAM/FF、WNS | 有 | 可用 |
| Route TNS | corrected ledger 未序列化 | 缺失 |
| COM5 cycles/latency/status/checksum | 有，含五次稳定性审计 | 仅作 board-side 辅助证据 |
| 完整网络 HLS source hash | 无 | 不通过 |
| 单一完整网络 HLS top csynth report | 无 | 不通过 |
| HLS cycles/LUT/DSP/BRAM | 仅为 6 个算子 csynth 的求和 | 组合估计，不是完整网络 csynth |
| HLS FF、network-level II | 无 | 不通过 |
| HLS failure category/elapsed-time 合同 | 无 | 不通过 |
| T6 `claimability_status=CLAIMABLE` | 无 | 不通过 |
| 至少 5 个架构族 | 仅 1 个 | 不通过 |
| 正式样本阈值 | 0/100 | 不通过 |

## 不能视为完整网络 HLS 样本的原因

每个 `candidate_hls_report.json` 将 stem、stage blocks、global average pool 和 classifier 映射到独立 cached operator kernel；cycles/resources 是各组件 csynth XML 的算术和，结构 skip 记为 0。没有候选专用完整网络 HLS top function、生成的完整网络 HLS source，也没有能覆盖跨层 FIFO、调度、共享和 network-level II 的单一 csynth report。

下游 Vivado harness 是候选专用完整网络并真实 route，因此 route 与 COM5 层仍有效；但这不能倒推算子求和 HLS 成为完整网络 HLS measurement。

## 允许与禁止复用

- 允许：保留 8 行用于修正后的 G2 calibration，并在原 A8 source freeze 与 latency-only 边界下使用 route 资源、WNS、COM5 latency。
- 允许：把 component HLS aggregates 作为明确标注的 analytic/composite predictor input，用于诊断 LUT/BRAM/latency calibration 失败。
- 禁止：把这些行放入 T6 grouped CV、HARP 训练、T6 denominator 或 `<30`、`30–99`、`≥100` 阈值。

## 合格 T6 行的最低收集合同

未来每行必须绑定：候选与 canonical architecture SHA；完整网络 HLS source tree 和生成配置/命令 SHA；单一候选 HLS top csynth report（cycles、latency、II、LUT、DSP、BRAM、FF）；候选 route report（资源、WNS/TNS、实际时钟、状态、失败类别）；工具版本、`xc7k325t-ffg900-2`、时钟策略和时间戳；语义等价结果及失败阶段；架构族标签；明确 claimability。只有从磁盘重算哈希且为 `CLAIMABLE` 的行进入正式计数，HLS/route 失败也必须保留为 ledger 行。

## Gate 状态

T6/F4 保持 PENDING。A8 cohort 改善了对当前 proxy 的理解，但不降低 100 个完整网络正式样本要求，也不关闭 G2。
