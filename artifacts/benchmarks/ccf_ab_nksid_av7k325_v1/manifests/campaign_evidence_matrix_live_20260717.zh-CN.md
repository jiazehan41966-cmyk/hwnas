# Campaign 实时证据矩阵（中文校正版，2026-07-17）

## 来源与校正说明

- 英文历史原件：`campaign_evidence_matrix_live_20260717.md`。
- 原件 SHA256：`2d7feb91a13cd4d0fc899945c36854e40b10466ad700932e4fc193d17a8f696c`。
- 英文原件保留了 NAS 运行期间的历史快照，其“闭集实时单元统计”和“下一自动边界”已经过时，不得作为当前状态来源。
- 本中文伴随档案根据 2026-07-17 17:53 完成的 15/15 独立审计校正为 NAS 15/15、总计 30/60，并明确禁止自动继续。英文原件保持只读以保留原 SHA。

## 逐项证据状态

| 要求 | 当前已核验证据 | 状态 | 尚需证据 |
|---|---|---|---|
| R1 论文与代码来源 | 五篇主论文均已固定；commit/remote 一致；增强版 T1 v2 已联结来源、smoke、环境与本地状态，并通过 6×26 跨格式审计 | 来源审计 PASS；canonical 接入待完成 | 在新的 source freeze 下把 v2 join 迁入受测试 builder，提升 `t1_v2.*` 并登记 HW-PR-NAS/HARP/ESDA gitlink |
| R2 独立环境 | 六个论文专用环境与 lock 文件均存在，全部探测 PASS | PASS | 只有环境或 adapter 变化时才需复测 |
| R3 闭集 | 预训练 MobileNetV2 与冻结 NAS 候选均独立接纳 15/15；NAS 平均 macro-F1 为 0.693619，预训练为 0.931632；15 个配对单元 NAS 全部更低，均值差 -0.238013，按 fold 分层的 10,000 次 bootstrap 95% CI 为 [-0.263968, -0.212667] | 方案 1 完成后再次暂停；30/60 已接纳 | scratch-v2 与 SURE 仍各缺 15 单元；完整 T2/F6 仍 PENDING；是否继续或建立新协议由用户决定 |
| R4 开放集/长尾 | 2,617 个样本、15 个单元的 5-known/3-unknown 输入 manifest 已通过哈希、成员关系和 unknown 训练泄漏审计；结果审计器能通过绑定 fixture 并拒绝未绑定 smoke | 输入与审计合同 FROZEN；结果 PENDING | 新 source freeze 下把 manifest 绑定接入 canonical 入口，运行 CE+MSP、DMCL、PLUD 各 15 单元并独立审计 |
| R5 正式 NAS | exact HV 与统计合同已测试；四方法 runner 尚不完整 | PENDING；G3 FROZEN | Random、RL、Aging、HW-PR 各 300 次 evaluator 调用 × 10 个配对 seed |
| R6 HLS/route 代理 | 工具链和 fail-closed 收集/预测合同通过合成测试；100 个唯一候选 DOE 与五个架构族 pilot 已独立冻结；planning-only 复现五个 pilot 均缺 component mapping，并拒绝旧 bitstream 复用 | 输入 PASS；生成器缺口确认；真实样本 0 | 在软件 source-freeze 边界后实现候选专用 HLS 映射和语义等价；五个 pilot 全通过后再收集剩余 95 个，并保留全部失败样本 |
| R7 板级与功耗 | 已观察到 JTAG target 与 COM5；板级延迟和三候选 UTC 对齐功耗合同仅通过合成测试；没有固定候选真实 campaign 或外部功率 trace | PENDING；NOT_MEASURED | 冻结三类 route-feasible 候选后完成真实板级延迟与同一外部仪器的 idle/active block |
| R8 measurement-first Gate | G0 PASS；G1/G2/G4 PENDING；G3 FROZEN；G5 PAUSED；功耗 NOT_MEASURED | PENDING | 每个 Gate 必须由本层证据闭合，禁止 proxy-to-board 提升 |
| R9 归档 | T1/F1 文件存在；T2–T9 与 F2–F12 按计划尚未生成 | 2/21 个图表 bundle 存在 | 所有剩余产物必须由已审计正式来源生成并通过一键重建审计 |

## 闭集单元实时统计

| 方法 | 要求单元 | 当前独立接纳 | 边界 |
|---|---:|---:|---|
| ImageNet 预训练 MobileNetV2 | 15 | 15 | 允许方法级摘要；尚不允许完整跨方法主表结论 |
| 冻结 NAS 候选 `rl_arch_135` | 15 | 15 | 方法级冻结架构摘要可用；历史 fold0 选择使其不能证明无偏 NAS 方法泛化 |
| 从零训练 MobileNetV2 v2 | 15 | 0 | 历史 scratch 仅作诊断，不能替代 clean re-audit cohort |
| 同 backbone SURE | 15 | 0 | source/environment smoke 不是正式 NKSID 结果 |
| **合计** | **60** | **30** | 60/60 和完整配对审计完成前，T2/F6 不可用 |

## 档案完整性与修复边界

1. 中文标题与图表必须显式按 UTF-8 读取；PowerShell 5.1 的 ANSI 默认显示乱码不能直接判为文件损坏。
2. T1 的来源 smoke、独立环境和 gitlink 接入缺口只能在新的 source freeze 下修复并重建，不能手工改正式表。
3. `results/.../smoke/` 下图片始终是非科学证据，禁止复制到正式 `figures/`。
4. T6 完整网络准入从 0 个真实样本开始；重复 encoding、算子级求和或旧 bitstream 不能增加样本数。
5. 历史英文档案保持只读，逐项补充绑定 SHA 的中文伴随档案；所有新建或重建产物直接使用中文。

## 当前停止边界

1. NAS 已完成 15/15 并通过独立审计；计划任务 Disabled，匹配训练进程为 0。限定批准只允许补齐最后单元，完成后已改名为 consumed 归档并从 wrapper 固定批准路径移除；下游许可始终为 false。
2. 未经用户新的明确选择，不启动 scratch-v2、SURE、corruption、开放集、NAS 四方法、HLS、板级或功耗实验，也不改变候选与协议。
3. 当前软件诊断不需要 AV7K325；只有三类最终 route-feasible 候选冻结并通过前置 Gate 后，才进入真实板级与外部功率阶段。

本矩阵是中文进度账本，不替代 canonical machine-readable readiness audit（规范机器审计），也不增加正式证据。
