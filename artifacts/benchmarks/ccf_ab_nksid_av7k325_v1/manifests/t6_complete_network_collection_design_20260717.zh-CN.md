# T6 完整网络 HLS/route 收集设计（中文伴随档案，2026-07-17）

- 英文原件：`t6_complete_network_collection_design_20260717.md`；当前实际 SHA256：`fb567a57a40a189703f66991b86ca8b081002f705eca834aafb9126b449c1f2f`。

## 目标与边界

收集至少 100 条唯一、semantic-safe 完整网络 HLS/route 行，用于按架构族 grouped five-fold proxy evaluation。执行目标预声明为 125 个候选（每个 macro family 25 个），使 synthesis/route failure 可见，且成功阈值不依赖选择性替换。

这是收集设计，不是已收集证据。只有磁盘审计器重算全部 SHA 并接纳至少 100 条 claimable complete row 后，T6/F4 才可能结束 PENDING。

## 历史行不足原因

- 29 个 board-harness candidate 文件只有 25 个唯一 arch ID、9 个详细 encoding、4 个顶层 operator sequence。
- 历史 RL200 evaluator 虽有 200 行，但只有 4 个唯一 network encoding；重复训练同一 encoding 不能计作不同 HLS network。
- 修正 A8 有 8 个唯一路由网络，但 HLS 是 component-summed operator estimate，对完整网络 HLS 计数增量为 0。
- `denoise`、`edge` 在 PyTorch-to-fixed-point/HLS numeric parity 与匹配 weight export 证据存在前不进入 semantic-safe T6 pool。

## 冻结候选域与五个 macro family

- 基础：matched RL/Aging 使用的 7-stage `mobile_anchor`。
- 允许算子：`mbconv` 与合法 identity `skip`；mbconv kernel 3/5、expand 1/2；每 stage width 0.75×/1.0×；使用 profile 的 shallow/deep depth。
- Skip 仅在 stride=1 且 input/output channel 相同合法；每个候选必须先通过 search-space validator。
- Target：`xc7k325t-ffg900-2`，Vitis HLS/Vivado 2023.2，固定 clock policy 与 generator version。

| Family ID | Width | Optional-stage depth | 目标数 |
|---|---|---|---:|
| `ma_wlow_dshallow` | 7 stage 全 0.75× | 每 stage 最小合法 depth | 25 |
| `ma_whigh_dshallow` | 7 stage 全 1.0× | 每 stage 最小合法 depth | 25 |
| `ma_walternating_dmixed` | 0.75×/1.0× 交替 | 可变 stage shallow/deep 交替 | 25 |
| `ma_wlow_ddeep` | 7 stage 全 0.75× | 可变 stage 最大 depth | 25 |
| `ma_whigh_ddeep` | 7 stage 全 1.0× | 可变 stage 最大 depth | 25 |

High/deep family 即使失败更多也保留，避免观察失败后删除导致 false-feasible/false-infeasible 偏倚。

## 确定性采样与分阶段执行

按每族冻结 seed 枚举或 rejection-sample 合法唯一 encoding；按 canonical architecture SHA 去重；尽量平衡 kernel、expand 与合法 skip-count bin；每族 25 个，不使用 accuracy、proxy resource、HLS 或 route outcome；首个 synthesis 前冻结有序 manifest。只有 pre-HLS semantic/schema failure 可替换，并保留 rejected row 与原因。

- Stage A：每族 1 个；五个候选均通过 source generation、semantic equivalence、完整网络 csynth 与 route provenance 后才能扩大。
- Stage B：每族 5 个，共 25；若 complete claimable <30，只作描述。
- Stage C：每族 15 个，共 75；结果 exploratory，不能支持正式 grouped inference。
- Stage D：每族 25 个，共 125；审计后至少 100 条 complete claimable 才进行正式 grouped five-fold 分析。

任何阶段不得静默删除失败；保留 architecture/source/config SHA、elapsed time、最后有效 report 和 normalized failure category。

## 单行合同与 grouped 分析

每行必须绑定 paper/method、candidate/pool SHA、`network_scope=COMPLETE_NETWORK`、完整网络 HLS top/source、semantic-equivalence report、source freeze、project commit/code state、command/config、csynth report 与 HLS cycles/II/LUT/DSP/BRAM/FF、route report 与 WNS/TNS/clock/resources/status、failure stage、工具版本、elapsed time、claimability。

Csynth PASS 必须有完整 report/metrics；route PASS 还需 route report/metrics；合法 csynth failure 可没有后续 report；route failure 保留成功 csynth 与 HLS 指标。所有失败必须有 `failure_stage/category`，进入 all-target failure denominator，但不进入 complete regression-target count。Component aggregate 只能作 predictor feature，measured target 只来自候选完整网络 top 与 route。

Grouped 5-fold 按五个固定 family，不随机拆行。方法为 analytic/LUT、linear regression、gradient boosting、HARP-GNN；指标为 MAE、RMSE、sMAPE、Spearman rho、Kendall tau、top-k recall、false-feasible、false-infeasible 与 calibration curve。HARP input 必须从候选 HLS C/C++ 通过支持的 LLVM/program graph 路径产生，NAS JSON 不是有效图输入。

13 个 target：HLS cycles/II/LUT/DSP/BRAM/FF，route WNS/TNS/achieved clock/LUT/DSP/BRAM/FF。100 条 complete row 需恰好 5,200 条 held-out prediction，产生 52 个 method-target metric row。Feasibility 由 held-out prediction/truth 重算：LUT≤203800、FF≤407600、BRAM≤445、DSP≤700、WNS≥0、clock≥200 MHz；物理 DSP 容量 840 单列，不能替代 700 gate。

## 执行边界

当前不启动该 collection。必须先在新 source freeze 下实现并 smoke 完整网络 HLS generator、冻结候选 manifest。HLS/route 阶段不需要 AV7K325 板卡；板卡只在后三候选 COM5 latency 与 power 阶段需要。合成 100-row/5,200-prediction 合同测试对真实 T6 增量为 0。
