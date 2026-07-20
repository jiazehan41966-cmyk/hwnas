# 硬件论文图表生成器缺口（中文伴随档案，2026-07-17）

- 英文原件：`hardware_artifact_builder_gap_20260717.md`；SHA256：`6ce01d1694b4ae5eb4ebfd09c57a6d51c5f2640e0907cb68405d0996c8929063`。

## 当前发现

仓库登记了 T6–T8 与 F4/F9–F12 标题，并已有 fail-closed HLS/route、board latency、external power 收集合同，但没有单一正式 builder 将接纳硬件证据转成全部表图格式。Phase0 script 与 smoke plot 不能替代，禁止手工拼装。所有命名 artifact 仍 PENDING。

## 未来 builder 的权威输入

1. 经 `audit_hardware_collection.txt` 接纳的 HLS/route truth CSV：≥100 条 claimable 完整网络行、5 个冻结架构族；
2. grouped-CV prediction CSV：每 `sample_id/method/target` 恰好一个 held-out prediction，并绑定 predictor/config/code/fold；
3. grouped proxy JSON：analytic/LUT、linear、gradient boosting、HARP-GNN 的 MAE/RMSE/sMAPE/rho/tau/top-k/false-feasible/false-infeasible/CI；
4. 三候选 selection manifest 与共同 route/HLS truth；
5. 通过 board auditor 的 latency CSV 与独立 full-validation accuracy audit；
6. 三个 power manifest、canonical summary、schema-v2 UTC audit；
7. source freeze、environment/toolchain、measurement-first ledger。

Prediction CSV 最低 schema 保留 campaign/sample/family/fold/method/target/predicted/measured/feasibility/predictor SHA/truth SHA/project/source freeze/claimability；训练 fold prediction 禁止混入正式 T6/F4。

## 输出映射

| Artifact | 必须内容 |
|---|---|
| T6 | target/method、n、MAE、RMSE、sMAPE、rho、tau、top-k、false-feasible/infeasible、CI、status |
| F4 | held-out predicted-versus-measured、identity line、calibration/CI、family/fold marker |
| T7 | 三角色的 HLS/route/board、resources、WNS/TNS/clock、latency p50/p95/p99、FPS、error rate、macro-F1/top1 |
| F9 | LUT/DSP/BRAM/FF 绝对值与三角色 utilization |
| F10 | 每角色 latency distribution 与 ECDF，含 n 和失败推理 |
| T8 | idle/active/dynamic W、total/dynamic mJ/inference、FPS/W、temperature、block/sample/inference 数 |
| F11 | 绑定 raw meter CSV，显示 idle/active 时间序列和 receipt-aligned 区间 |
| F12 | 三候选 macro-F1、board latency、dynamic energy Pareto，仅 power PASS 行 |

T7/T8 measured 列禁止使用 Vivado estimated power、GPU diagnostic、作者 ZCU102 数值或搜索 proxy；T6 禁止把 A8 operator sum 当完整网络 truth。

## 归档合同

- T6–T8 同一 row object 生成 CSV/Markdown/LaTeX。
- F4/F9–F12 输出 300 dpi PNG、vector PDF、`*_source.csv`、中文 `*_meta.json`。
- Meta 记录中文标题、caption、支持结论、限制、generator/input/output SHA。
- Build manifest 记录命令、Python 环境、source freeze、工具版本、时间与 claimability。
- 先写 staging，所有输出验证后 atomic promote；缺输入、audit 非 PASS、duplicate held-out、样本/族不足、role mismatch 或 UTC failure 时不碰现有正式 artifact 并返回非零。

## 所需实现

在新 source freeze 下新增 canonical `scripts/build_hardware_benchmark_artifacts.py` 及测试；synthetic smoke 只能放 `results/.../smoke/` 且标非科学并目视 QA；T6/T7/T8 source audit PASS 后才运行 real builder 并重建 readiness/ledger。

实现 builder 不需要板卡或外部仪器；但真实 T7/T8/F9–F12 在测量存在前不可能生成。
