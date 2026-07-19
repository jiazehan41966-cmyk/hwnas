# T6 硬件收集合同修复（中文伴随档案，2026-07-17）

- 英文原件：`t6_hardware_collection_contract_patch_20260717.md`；SHA256：`6deef335ffdd4c0f5a91a1e9bda824ce22472de804cfc60f3226e12a005eba33`。

## 发现的问题

旧 pre-collection schema 不能机械证明一行是完整网络 HLS top，semantic-equivalence status 也未绑定 report SHA；同时它对所有行都强制要求 csynth/route report，即使某阶段在 report 产生前合法失败。由于 grouped readiness 又要求 audit errors=0，一个诚实失败可能永久阻断正式 readiness，与保留失败做 failure-rate/false-feasibility 分析的要求矛盾。

## 修复

CSV 与 auditor 新增或跟踪：`paper_id/method_id`、candidate-pool manifest SHA、显式 `network_scope` 与完整网络 HLS top 名、semantic-equivalence report path/SHA、source freeze/project commit/code-state、既有 candidate/source/toolchain/config/command/checkpoint，以及 stage-aware csynth/route report、metrics、failure category。

只有 `network_scope` 为 `COMPLETE_NETWORK` 或 `FULL_NETWORK` 的行可进入 `complete_hls_route_rows`；operator-only 行可保留为合法 inventory，但不能增加 T6 denominator。

- csynth PASS：必须绑定 csynth report 与完整 HLS metrics；
- route PASS：还必须绑定 route report 与 route metrics；
- csynth failure：可将 csynth/route report 与 metrics 留空；
- route failure：保留成功 csynth/HLS metrics，缺失 route report/metrics 可为空；
- 每个 failure 必须有 `failure_stage/category`；
- route PASS 但 csynth 非 PASS 为不一致并拒绝。

## 合同测试

隔离临时目录覆盖五种情况并全部符合预期：完整网络 csynth+route PASS 计 1 条 complete；operator-only PASS 计 0 complete 且无 schema error；合法 csynth failure 与 route failure 均作为 failure 保留且不产生虚假缺 report 错误；缺 failure stage/category 被拒绝。

阈值测试生成 100 条 complete row（五架构族均分）和 1 条合法 csynth failure，独立审计返回：`total_rows=101`、`complete_hls_route_rows=100`、`architecture_family_count=5`、`failure_rows=1`、`FORMAL_COUNT_REACHED`、`grouped_5fold_ready=true`、errors `[]`。Header-only template 保持 fail-closed。合并 T7 board-latency 审计后再次回归，结果不变。

## Artifact SHA256 与陈旧交叉引用

| Artifact | 英文原件记录的 SHA256 |
|---|---|
| `runtime/audit_hardware_collection.txt` | `1a48570a4d2dd016e5e7cb14ae77019bb6e398793cbabbacc198eb522137ccf4` |
| `runtime/hls_route_sample_template.csv` | `51eee0037a8178adc79685a101515ddf8f8d97a6265523d49992d5cf0ef92a18` |
| `runtime/hardware_collection_runbook_20260716.md` | `3c72d57aad423f1e96c8e01290cb2b82057c48aca0505169f5910c43559ba45b` |
| `t6_complete_network_collection_design_20260717.md` | `cd76632dcbb0151f513dd5bb3d171b4a111fb379c6ce406cfbb647f1db5c32a2` |

最后一项已陈旧：当前只读 collection-design 原件实际 SHA256 为 `fb567a57a40a189703f66991b86ca8b081002f705eca834aafb9126b449c1f2f`。这表明历史补丁的交叉引用未随设计档案后续更新而刷新。由于 T6 仍为 0/100，本问题不污染实测行；但在新 source freeze 下正式接入前必须由生成器重建并重新绑定，禁止把旧 SHA 手工宣称为当前值。

## 结论与执行边界

本补丁只修复未来合同，不创建任何 HLS/route measurement；T6/F4 PENDING，正式样本增量 0。完整网络 generator 与 125-candidate manifest 仍需新 source freeze。HLS/route collection 不需要 AV7K325 板卡。
