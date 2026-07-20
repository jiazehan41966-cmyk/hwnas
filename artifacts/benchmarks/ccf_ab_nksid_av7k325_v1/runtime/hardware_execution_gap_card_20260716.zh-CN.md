# HLS / AV7K325 / 功耗执行缺口卡（中文伴随档案）

- 英文原件：`hardware_execution_gap_card_20260716.md`；SHA256：`0dfa37ebc1e2a2d19661c7020c82ebda00c8c0b6e0d2e2d9aaee53aa2e879036`。

## 当前只读状态（2026-07-17）

- 修正 A8 ledger 有 8 个唯一 route/COM5 行（4 fit+4 probe），但 HLS 是 component-summed operator estimate，不是 candidate-specific complete-network HLS top，因此 T6 增量 0；正式阈值仍为 ≥100 条、≥5 架构族。
- G2 PENDING、G3 FROZEN、G4 PENDING、外部功耗 NOT_MEASURED。
- HARP checkout/独立 CUDA 环境通过 LLVM program-graph smoke，只证明 adapter/environment 形状，不证明项目样本预测精度。
- A8 的 4 个独立 probe 均通过 source-frozen route 与 COM5 5/5 稳定性，外部 probe 数量门已闭合；但 G2 质量门仍失败。DSP interval screening PASS（P90 APE 12.46%、rho 1.0）；LUT 44.30%、BRAM 85.33%、latency 55.98% 且 rho 0.4 失败。因此只有 DSP 可 hard-screen；LUT/BRAM/latency 必须进入真实 HLS/route，不得把估计标为实测。
- 自动发现匹配 Vivado/Vitis HLS 2023.2；formal part 为 `xc7k325t-ffg900-2`，`ffg676-2` 只用于历史/operator LUT sampling。
- Windows 检测到健康 CP210x COM5；只证明 UART bridge。FTDI/JTAG 经 hw_server 看到 Digilent target 和 `xc7k325t_0`；未烧写 bitstream、未执行 inference。
- Trivial HLS kernel 在 ffg900-2 上 II=1；trivial RTL synth/place/route WNS 4.091 ns、TNS 0、36/36 nets、9 项 SHA PASS。两者均为 `SMOKE_ONLY_NOT_FOR_SCIENTIFIC_CLAIMS`，T6/T7 增量 0。
- 已冻结结构 DOE：100 unique encoding、5 架构族×20、5 个 leave-one-family-out fold；没有 semantic equivalence/HLS/route truth，T6 仍 0/100。

## 样本级别与收集 Gate

- <30 条完整网络：仅描述；30–99：exploratory；≥100 且 grouped 5-fold/provenance PASS：才可正式比较。
- 每行绑定 paper/method、candidate/pool SHA、complete-network scope/top、generated source、semantic equivalence、source freeze/project state、tool/part/clock/command/config/report/failure。
- HLS cycles/II/LUT/DSP/BRAM/FF 与 route WNS/TNS/clock/resources/status 是不同 target；合法失败保留，stage-aware 审计不伪造未产生 report。
- Proxy 方法：analytic/LUT、linear regression、gradient boosting、HARP-GNN；按架构族 grouped 5-fold，不允许随机 row split。T6/F4 在门槛前不可用，HARP 作者数值不能填项目 T6。

## 三个部署候选

1. `accuracy_first`：route-feasible 中 macro-F1 最高。
2. `knee_point`：归一化 accuracy/latency/resource front 的 knee。
3. `resource_min`：macro-F1 距最佳不超过 1 个百分点时 LUT 最低。

三候选必须使用相同 AV7K325 toolchain、clock、bitstream flow、COM5 harness、外部仪器协议。ESDA/ZCU102 只作 C 类流程参考，不与 AV7K325 排名。

## 板级与功耗验收

- Board latency：每角色固定且不同候选；相同 sample/target map；每角色 ≥1,000 条 timestamp；route-clean/bitstream verified 后报告 p50/p95/p99、FPS、error rate。
- Power：同一外部仪器、同一 wiring/range/sample-rate；每候选 ≥3 idle + 3 active block，每 active ≥1,000 inference；active CSV UTC 与 RUN_REPEAT receipt 对齐。
- 原始 meter time series 必须保留；主指标 dynamic mJ/inference，次指标 idle/active/dynamic W、FPS/W、temperature。
- 搜索期 power/energy estimate 永远是 diagnostic，不得标成 measured。原始数据和 instrument/tool manifest 缺失前，T7/T8/F9–F12 不可用。

## 已关闭与未关闭外部条件

已关闭：Vivado/HLS executable discovery、part database、license/report path、JTAG cable/target enumeration。仍需：semantic-safe complete-network candidate、冻结 candidate/bitstream、已连接外部功率仪器及经验证 acquisition command。当前严格 preflight 因无外部仪器命令 exit 2。

## 已准备控制与边界

已有 read-only preflight、part/target query、smoke manifest、power v2 schema、runbook、HLS/route/board/power templates、硬件收集审计和 grouped prediction contract。合成测试包括 100 truth/5,200 prediction/52 metric 以及三角色×1,000 board rows、预期 120 mJ/inference power path；它们只证明合同，真实样本增量 0。

没有 T6–T8/F4/F9–F12 正式 builder；未来必须在新 source freeze 下由单一 hash-bound builder 生成，禁止手工表图。

## 当前直接结论

五族 pilot 的 plan-only 结果均为 `not_generated_mapping_incomplete`，每个至少缺 6 个 component row、无 candidate-HLS mapping，并拒绝 arch-84 bitstream 复用。下一任务不是操作板卡，而是在新 source freeze 下实现 source-linked candidate-specific HLS mapping 与 semantic equivalence。现有 stitched RTL planner 也不是可直接供 HARP 使用的完整网络 HLS C/C++ generator。
