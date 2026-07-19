# T1 可复现性覆盖缺口审计（中文伴随档案，2026-07-17）

- 英文原件：`t1_reproducibility_coverage_gap_20260717.md`；SHA256：`2f0b6d51dd3285884022a81f93ba052c43f8ef13670bf727dce2cd4b858f4269`。

## 范围与边界

本卡只读核对现有 T1 bundle 与冻结的 benchmark 验收条件，不修改 source、不重建 canonical T1，也不声明论文复现。权威输入包括 `source_audit.json`、`source_smoke.json`、六张环境卡、`tables/t1.csv` 及 canonical row builder。

## 当前证据

| Paper ID | Commit pin | License | 论文—代码对应 | Source smoke | 独立环境 | 本地正式结果 |
|---|---|---|---|---|---|---|
| `hw_pr_nas_2023` | PASS | MIT verified | partial | `BLOCKED_OFFICIAL_CODE_INCOMPLETE` | PASS | 未完成 |
| `sure_2024` | PASS | 缺失/不可确认再分发 | verified | PASS | PASS | 未完成 |
| `harp_2023` | PASS | BSD-3-Clause verified | verified | PASS | PASS | 未完成 |
| `esda_2024` | PASS | MIT verified | verified | PASS | PASS | C 类设计参考 |
| `dmcl_sonar_oltr_2025` | PASS | 缺失/不可确认再分发 | partial | `PASS_SOURCE_PRESENT` | PASS | 未完成 |
| `plud_sonar_oltr_2024` | PASS | 缺失/不可确认再分发 | verified | `PASS_SOURCE_PRESENT` | PASS | 未完成 |

Source smoke 或环境 PASS 只表示固定源码/adapter 在受测边界存在并可执行，不等于复现论文结果，也不使行正式 eligible。

## T1 覆盖缺陷

旧 row builder 只复制 source-audit 字段，缺少 source-smoke 状态和 HW-PR-NAS 缤失文件、独立环境状态/解释器/lock SHA、本地统一协议状态、当其他 blocker 为空时的 `local_unified_protocol_not_completed`，以及 author-code reproduction、paper-spec adapter、C-class design reference 的区分。结果是 HARP/ESDA 可显示 `formal_eligible=False` 却没有可读 blocker，虽 fail-closed 但对读者含糊。

## 新 T1 最低 schema 与规则

至少联结：`paper_id, registry_role, direction, venue, comparability_class, repo_url, observed_commit, pin_match, license_state, redistribution_allowed, paper_code_correspondence, source_smoke_status, environment_probe_status, environment_lock_sha256, local_protocol_status, execution_role, numerical_comparison_rule, formal_eligible, blockers`。

- HW-PR-NAS 必须标为 `paper_spec_adapter`，不得描述为固定 commit 下可执行的作者方法复现。
- SURE、DMCL、PLUD 在许可证未核验时继续隔离且不得再分发。
- ESDA 保持 C 类，不参加跨平台 AV7K325 数值排名。
- source/environment PASS 不能在本地统一协议及独立审计缺失时清除 `formal_eligible`。
- CSV、Markdown、LaTeX 必须从同一 joined dataset 生成，禁止只手改一种格式。

## Staged v2 结果

不改变冻结 source 的 runtime-only staged replacement 已生成 `t1_v2.csv/.md/.tex`：6 行 × 26 列，联结 source、smoke、环境和本地状态。独立审计重算 4 个顶层 source、6 张环境卡、6 个 lock 与 3 个输出，并从 CSV 重建 Markdown/LaTeX。

审计 PASS：5 篇主论文 + 1 篇补充论文；6 行均有显式 blocker；正式数值 eligible 为 0。HW-PR-NAS 明确因官方发布不完整被阻断；HARP 仍需项目完整网络图和 grouped evaluation；ESDA 保持 C 类；无许可证仓库继续隔离。

这解决 staged artifact 的表设计缺陷，但未完成 canonical-source integration。后续必须在新的 source freeze 下把同一逻辑迁入受测试 builder、将 v2 schema 提升到正式 `t1.*` 并重建 readiness。此前旧 `t1.*` 仍是 canonical snapshot，`t1_v2.*` 只是独立验证的替换候选。

本卡只证明审计与表设计边界，不证明论文数值复现。
