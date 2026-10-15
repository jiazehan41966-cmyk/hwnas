# Sonar Detection V0.1 / FPGA 当前进度

> 2026-09-22 更新入口：[声呐协同设计当前执行状态](SONAR_WORKFLOW_CURRENT.md)。已重算最新 H4 代价覆盖、核验正在运行的联合搜索，并验收局部 DMA 板测证据。以下保留 2026-09-20 的历史审查快照，其 NEEDS_FIX 与未运行状态不代表最新状态。

核查日期：2026-09-20。仅适用于声呐检测 V0.1 分支，不替代分类项目或历史 measurement-first 状态。

| 阶段 | 当前对齐状态 | 证据与边界 |
|---|---|---|
| Search Space V0.1 | FROZEN，144 architectures | 冻结配置未修改 |
| H2 数值与接口合同 | PASS（沿用已完成合同） | 本轮未重跑整网整数/RTL验证；不代表部署完成 |
| H3 measured cost DB | H3_DB_READY，限定已测网格 | 92 次尝试、78 有效、14 历史失败、0 待运行；51/51 critical shapes；本轮 78 份报告哈希全部匹配 |
| H4 固定候选 mapping 结果 | 已生成，表内算术通过核查 | baseline/B1/C/D 共 40 个命名策略结果；仍是 CSYNTH 算子串行和及条件资源估计 |
| H4 通用 cost API/tooling | NEEDS_FIX，当前审查不放行 | 历史 manifest 写 READY；本轮发现节点映射覆盖、完整 key/合同校验和图身份绑定漏洞 |
| 整网 FPGA 部署 | 未建立 | shell reservation UNKNOWN；无本轮整网 P&R、DDR runtime 或板测证据 |

H3 的来源覆盖为 baseline 37/37、B1 41/41、C 37/37、D 37/37，各来源有重复 shape，不能相加当作 unique shape 数量。

H4 当前固定表中的代表 min-latency 串行计算估计：B1 10,779,201.640 us，C 4,762,930.385 us，D/baseline 4,509,321.045 us。它们不是板端端到端延迟。B1 两个 P2 tower 占其计算估计约 53.25%。软件精度与整数硬件成本配对不等于量化后检测精度已验证；baseline 精度沿用 D，非独立训练证据。

当前修复顺序：保留每个节点独立 mapping → 完整 measurement key 和双合同校验 → 图身份与精度绑定 → 数据准入与 timing eligibility → 修正文案 → 重新生成并绑定 H4 manifest 后复核。现有 H3 报告没有因这些 API 问题被判无效，不需要据此重跑全部 HLS。

本次仅审查及文档对齐，未开始新搜索、训练或综合。服务器 SSH 已恢复；2026-09-20 12:04 的用户进程快照未见 HLS 任务。

- [本轮完整审查、复现与修复要求](../reports/fpga_progress_review_20260920/REVIEW.md)
- [H3 manifest](../reports/fpga_h3_cost_db/H3_DATABASE_MANIFEST.json)
- [H4 历史 mapping manifest](../reports/fpga_h4_mapping_search/H4_DATABASE_MANIFEST.json)
- [H4 历史 tooling manifest](../reports/fpga_h4_tooling/H4_TOOLING_MANIFEST.json)

阅读优先级：本页与本轮审查说明当前审查结论；历史 manifest 保留生成时状态，不能单凭其中 READY 覆盖后续发现。
