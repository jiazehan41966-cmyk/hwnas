# T7 板级延迟收集合同修复（中文伴随档案，2026-07-17）

- 英文原件：`t7_board_latency_contract_patch_20260717.md`；SHA256：`826cf476fa9c0a4e429ef2dc9f678bd8b71fa706b5024b3133967e0590b793bd`。

## 原问题

旧 board-latency CSV 只记录 bitstream、payload、parity 的 SHA 字符串；审计器只检查格式，没有绑定实际文件，也未证明每个 deployment role 固定一个候选、三候选使用相同配对验证样本或满足最小推理数。因此不能支持 T7/F9/F10 provenance 或三候选配对比较。

## 修复内容

每条 board row 现在绑定：paper/method、candidate ID/SHA、共同 candidate-selection manifest；candidate manifest 与 checkpoint；source freeze、project commit/code-state SHA；data/split/validation manifest；route report、bitstream、逐推理 payload 与 parity summary 的 path/SHA；toolchain fingerprint、board target/serial、COM 与 baud；host timestamps、latency、board cycles/clock、CRC/numeric status、temperature、target、prediction、correctness、claimability。

审计器对每个引用文件缓存计算 SHA，并强制：

- `accuracy_first`、`knee_point`、`resource_min` 各且仅各有一个不同候选；
- role 内 provenance 恒定；
- 三角色 `sample_id/target` 映射完全相同；
- latency 与 host timestamps 一致；
- 每角色至少 1,000 条推理行。

Transport 或 numeric failure 保留为合法行并进入错误率；provenance/schema failure 使整个 collection 无效。

## 合同测试

- 更新后的审计器可编译；只有 header 的 template 以 0 行正确 fail-closed，`schema_and_provenance_pass=false`。
- 临时 3,000 行 campaign（3 角色 × 1,000 配对样本）PASS：三个候选 ID 不同、paired sample set 相同、15 个 bound file、errors `[]`。
- 注入失败均被拒绝：`accuracy_first` 内出现第二候选；只改一个角色的 target；每角色只有 999 行。

## Artifact SHA256

| Artifact | SHA256 |
|---|---|
| `runtime/audit_hardware_collection.txt` | `1a48570a4d2dd016e5e7cb14ae77019bb6e398793cbabbacc198eb522137ccf4` |
| `runtime/board_latency_sample_template.csv` | `cb9bd9ba58de2b8fdeedf84d5725c9b3c0985a0a503c1c76443fa2e41f4ddab4` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |

## 结论边界

这只修复收集合同。尚未测量任何 AV7K325 推理行、未烧写候选，T7/F9/F10 PENDING。即使 latency CSV 审计 PASS，也不能替代完整验证 accuracy audit、route/parity acceptance 或外部功耗协议。
