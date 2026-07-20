# 开放集样本冻结补丁（中文伴随档案，2026-07-17）

- 英文原件：`open_set_sample_freeze_patch_20260717.md`；SHA256：`6621fde217b9553b95daa09ffbe6ec1c6c7d08d4e43ff1b86c64048fa7163c55`。

## 目的与冻结证据

在 CE+MSP、DMCL、PLUD 正式训练前冻结 5-known/3-unknown 实验的精确 NKSID 样本与成员关系，闭合 class-level protocol 与逐样本 provenance 的缺口，不产生性能结果。

- Dataset：NKSID，2,617 图像，8 类。
- 单元：5 outer folds × seeds 42/43/44 = 15；inner-validation fraction 0.15。
- Manifest：`open_long_tail_sample_manifest_v1.json.txt`，SHA `59878d48786129c983e976b1cf8f4fc03bda79bd9e05ec5671ab42dedc1f7a3e`。
- Builder SHA：`097c6e27e34253d66c111a6d4b43e3ae8b3cf88e961cd38833ea896c0996b087`。
- Audit：`open_long_tail_sample_manifest_v1_audit.json.txt`，SHA `63043398d989e10da319ab1a70bafa8204651e279bbf9cce28269ef1eb5f759f`；auditor SHA `c14a21d702099593138d361cc69001c500f729187a9ef78b11d22837f88305c9`。

## 独立检查

- 2,617 个 image path、label、size、SHA 全部重核验；
- 从 canonical split 重建 15 个 train/inner/outer membership；
- unknown class 不进入 known-only training/threshold calibration；
- outer known 与 unknown 互斥且精确覆盖 outer fold；
- 同 fold outer membership 跨 seed 不变，五 outer fold 互斥且并集为 2,617；
- class/evaluation protocol、split file、source-freeze manifest 与 source archive 全部 hash-bound。

Manifest 为 `FROZEN_INPUT_NOT_RESULT`，正式结果增量 0；R4/T3/F5 PENDING。Canonical 入口必须在新 source freeze 下对 CE+MSP/DMCL/PLUD 每条正式记录持久化该 SHA，结果审计器拒绝任何缺失或不匹配。

`audit_open_run.txt` 现还要求 immutable config 同时绑定 manifest 与独立审计，并逐单元检查 outer 顺序、target、class protocol、base split SHA、record/provenance 与 row-level manifest SHA。Bound fixture PASS；未绑定 smoke 被正确拒绝；两者正式增量均为 0。
