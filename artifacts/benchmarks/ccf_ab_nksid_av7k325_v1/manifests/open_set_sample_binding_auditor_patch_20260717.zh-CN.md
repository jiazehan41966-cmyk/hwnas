# 开放集逐样本绑定结果审计补丁（中文伴随档案，2026-07-17）

- 英文原件：`open_set_sample_binding_auditor_patch_20260717.md`；SHA256：`e98a9621f930eb09f83dd30e64a9274727b904d1362100e5eb922ae5e8bacec1`。

## 已闭合缺口

旧结果审计器会检查 5-known/3-unknown 类划分、输出数、prediction/checkpoint SHA 和指标，但不能证明每条 prediction 属于预声明 sample-level manifest；即使 15 单元齐全，也可能在未绑定冻结 2,617 样本对象时通过。

Staged runtime auditor 现在要求 `immutable_config.open_set_sample_manifest` 绑定 manifest path/SHA、独立 input-audit path/SHA、`FROZEN_INPUT_NOT_RESULT` 身份、2,617 样本与 15 个单元、class protocol SHA、source-freeze SHA，以及 input audit PASS/2,617 重哈希/15 单元/0 错误/0 结果增量。

对每个完成单元，独立核验 prediction `sample_id` 顺序等于 manifest `outer_val_indices`、target 等于冻结 target、class holdout/base split SHA 相同、record/provenance 完整绑定、每个 prediction row 含精确 manifest SHA。

## 合同证据

- Auditor `runtime/audit_open_run.txt`：`a179bb8880ddf82ed44a8b777ce5b5dece6bb957c51b5a3014bf9675ddf2f773`。
- Test generator：`2fdcef445fae1d98c4776e4b034d2fe06e2a11df792f6b65648acb777fe7e68e`。
- 已绑定 one-unit fixture result：`319ff249354dd8001834eb19227d0a211057ea70648a5db99c11e2b4feb3bbde`，PASS。
- Fixture audit：`a12d4c87b9e16b459227d469423bb206ab758012be39fe1ad532a8f3dbbcdc89`，1 单元、0 错误。
- 未绑定 CE+MSP rejection：`19a418a97df52197600170f4cd7b7761f2379fda7e92596efabcbaa0d5fb7f46`，预期 exit 2、FAIL。

全部 fixture 都是 `NON_SCIENTIFIC_SMOKE_ONLY_FORMAL_RESULT_INCREMENT_0`；R4/T3/F5 和方法排名仍不可用。

## 剩余 canonical integration

新 source freeze 下，`run_eval_protocol.py` 必须显式接收 sample manifest，训练前验证，并在 immutable config、record、checkpoint meta、prediction row 持久化完整绑定。此前 CE+MSP、DMCL、PLUD 均不能通过升级审计。
