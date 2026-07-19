# G1 NAS 低性能暂停记录（中文伴随档案）

## 来源

- 英文历史原件：`g1_nas_underperformance_pause_20260717.md`。
- 原件 SHA256：`f17a1883379cdaeff35bcd0f4002ef1db333a37a7bd110345260d8db1412e5a5`。
- 本文件忠实保留原暂停事件，并用后续状态卡更新计划任务状态；不改写原件。

## 触发原因

fold 4/seed 43 完成原子写入后，NAS 候选已有 14 个通过独立审计的 fold-seed 单元。与相同 14 个 ImageNet 预训练 MobileNetV2 单元相比，NAS 的中期 macro-F1 均值为 `0.6904529534`，参照方法为 `0.9300808161`；配对均值差为 `-0.2396278627`。14 个差值全部为负，范围为 `[-0.3250370668, -0.1621433112]`。使用 seed `20260717` 的 10,000 次配对 bootstrap 得到中期 95% CI `[-0.2612163648, -0.2169519822]`。

这是实质性低性能，但 NAS 尚未完成。本组数字只支持用户决策，不作为 T2/F6 正式证据。

## 安全中断

- fold 4/seed 43 产生 537 条预测，并通过 14 单元独立 partial audit（部分审计）。
- 部分审计 SHA256：`0e8da5427b19fb8ddf2aefb871ee4965839c21f17e294887b2668b50dc6e507c`。
- 两个 Python 进程只在该原子单元完整出现后停止。
- fold 4/seed 44 不存在 checkpoint、运行记录或预测文件。
- 原暂停时计划任务处于 Ready、最后结果为 2；随后为加强门禁已改为 Disabled。
- 当前匹配训练进程为 0，未进入 SURE 或 scratch-v2。

## 必须由用户决定的事项

1. 是否花费最后一个 fold-seed 单元完成 NAS 15/15，并在完成后再次暂停审计。
2. NAS 结果闭合后，是否继续 SURE 与 scratch-v2。
3. 是否在冻结协议下只诊断该弱候选，或通过显式协议修订和新的 source freeze 更换候选。

在这些决定被独立记录之前，不允许自动继续。补齐最后单元不等于认可候选有竞争力；更换候选不能混入当前冻结运行。

## Fail-closed 门禁验证

持久化 wrapper 在 `resume_nas_to_15` 和 `continue_downstream_closed_set_chain` 前均检查绑定暂停 SHA 的用户决定文件。无批准文件的真实计划任务调用已复核 source freeze，记录 `PAUSED_PENDING_USER_DECISION resume_nas_to_15`，以结果 2 退出，保持 14 条单元级预测文件并未启动 NAS 训练。测试记录为 `g1_nas_decision_gate_test_20260717.json.txt`。

相邻的用户决定模板明确标为不可执行，不能授权工作。当前计划任务 Disabled、批准文件不存在；恢复必须得到用户明确选择并重新核验 source freeze。

## 证据边界

本中文伴随档案不增加正式 T2/F6 证据，不授权继续训练，不改变候选、协议或 source freeze。
