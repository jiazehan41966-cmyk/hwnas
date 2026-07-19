# T8 外部功耗收集合同修复（中文伴随档案，2026-07-17）

- 英文原件：`t8_power_contract_patch_20260717.md`；SHA256：`46067647c115a9a59b96cff221641de94e05d97e835f6dd1565853a8fa5946aa`。

## 原问题

Canonical power calculator 已要求 3 个 idle/active block、每 block 60 秒、至少 1,000 次推理、外部 board-input 测量和 receipt 数量，但没有证明 active CSV timestamp 与 `RUN_REPEAT` UTC 区间对齐，没有在 manifest 中绑定 CSV/receipt SHA，没有把 protocol fingerprint 绑定到 protocol 文件，也未要求三个 manifest 对应三种冻结候选角色。名义 PASS 的 campaign 可能把 active 前后样本纳入动态能量，或混合无关候选。

## Schema-v2 与审计修复

候选 manifest 新增：candidate role 和 selection/candidate/checkpoint provenance；source freeze、project commit/code state、data/split/route；bitstream/payload/parity path/SHA；共享 protocol manifest path/SHA（其 SHA 即 protocol fingerprint）；UTC Unix epoch timestamp basis；instrument serial/ranges 与 calibration certificate；所有 idle/active CSV 和 RUN_REPEAT receipt 的 SHA256。

`audit_power_campaign_v2.txt` 在 canonical calculator 之外检查：

- `accuracy_first`、`knee_point`、`resource_min` 各且仅一个 manifest；
- 三个不同 candidate ID，共享 selection manifest、source freeze、code/data/split 与 measurement protocol；
- 三候选使用同一 instrument identity 与 calibration；
- protocol 文件内容与 instrument、rail、source、timestamp basis 一致；
- 每个 artifact、raw CSV、receipt 的 SHA 匹配；
- 观测 CSV sample rate 与仪器率误差不超过 5%；
- active CSV 首末 UTC epoch 与 receipt 的 `active_started_utc/active_finished_utc` 在采样率容忍范围内对齐；
- receipt UTC/monotonic duration 一致，每 active interval ≥60 秒、repeat count ≥1,000，programming/UART upload 不在测量区间；
- receipt 中 bitstream/payload/parity SHA 与候选 manifest 一致。

`.txt` 后缀用于保持当时活动 G1 code fingerprint；最终 T8 验收前，必须在新的 source freeze 下把逻辑迁入受测试 canonical source。

## 合同测试

临时三候选 campaign：每候选 3 个 idle + 3 个 active 60 秒 block，1 Hz；idle 5 W、active 7 W、每 active block 1,000 次推理。Base calculator 与 v2 audit 均 PASS，三候选得到预期 `120 mJ/inference` dynamic energy。

负例均被拒绝：receipt UTC 平移 10 秒而 CSV 不变；把 `resource_min` 重标为第二个 `accuracy_first`；改变一个 instrument serial。

## Artifact SHA256

| Artifact | SHA256 |
|---|---|
| `runtime/audit_power_campaign_v2.txt` | `5a798f0524eb78d92c5769f2f606a9d2faeadb86533b919c6ce55c88ea448556` |
| `runtime/power_measurement_manifest_template.json.txt` | `f1d3f56c1344c3c826d62b7844e74f2555af19c6d7f805fb1e3154a689210774` |
| `runtime/power_measurement_protocol_template.json.txt` | `da4dc9f898b5ffa449786c7c95255077e7dc19c7ef8cf738966603aa7843571c` |
| `runtime/power_timeseries_template.csv` | `e490184df593f6959e58692811d4c8506f16c8275f02bc1901ca9bf637746433` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |

## 结论边界

这是已测试证据合同，不是实测功耗。当前没有外部仪器命令或 raw meter trace，功耗仍为 `NOT_MEASURED`，T8/F11/F12 PENDING。GPU power 与 Vivado estimated power 只能作 diagnostic/proxy，不能填入该 schema。
