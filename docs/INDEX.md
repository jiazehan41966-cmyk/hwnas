# 项目文件与文档导航

> 2026-09-20 范围内更新：[声呐检测 V0.1 / FPGA 当前进度](SONAR_DETECTION_FPGA_PROGRESS.md)，含 H3/H4 最新证据核查与待修复项。其余导航维护日期保留如下。

维护日期：2026-09-09。本页按用途组织已有文件；状态快照只代表对应日期和实验范围，不代表整个项目已完成。

## 整理文件与查找实现

- **[文件夹整理执行入口与关键实现路径](FILE_CLEANUP_PLAN.md)**：唯一的文件整理顺序；方法入口、实现目录、保留范围和四个清理批次。
- [仓库布局](REPO_LAYOUT.md)：源码、配置、产物目录的职责。
- [快速开始](QUICKSTART.md)：既有通用/AV7K325入口说明；具体实验配置及准入条件需另行核对。

## 协议与方法说明：按适用范围使用

| 范围 | 文档 | 阅读边界 |
|---|---|---|
| 当前方法与源码 | [method_design](method_design.md)、[architecture](architecture.md) | 八阶段分类方法、DMDC/CFEM、硬件聚合与独立检测入口 |
| 评估协议 | [EVAL_PROTOCOL](EVAL_PROTOCOL.md)、[配置版本](../configs/evaluation) | 区分V1/V2与所绑定的来源清单；示例命令不等于当前运行已获准 |
| 候选身份、预算与证据 | [FORMAL_EXPERIMENT_CONTRACT](FORMAL_EXPERIMENT_CONTRACT.md) | C1—C6是契约章节名，不与measurement-first的G阶段编号混用 |
| measurement-first | [MEASUREMENT_FIRST_REBUILD](MEASUREMENT_FIRST_REBUILD.md)、[FULL_EXPERIMENT_CYCLE_V2](FULL_EXPERIMENT_CYCLE_V2.md) | 该实验系列的门禁和冻结周期；不是所有后续分支的统一状态 |
| 搜索配置版本与对照 | [SEARCH_CONFIG_CANONICAL](SEARCH_CONFIG_CANONICAL.md)、[MULTIOBJECTIVE_AGING_EVOLUTION](MULTIOBJECTIVE_AGING_EVOLUTION.md) | ZCU104的20/200轮分类与640检测配置；AV7K325及RL/aging为各自历史范围 |
| 四阶段成熟算子 | [SONAR_FOURSTAGE_OPERATOR_CONCLUSION](SONAR_FOURSTAGE_OPERATOR_CONCLUSION.md) | 软件准入与完整网络硬件结论分层 |
| ZCU104接口 | [DMA shell接口](ZCU104_DMA_SHELL_V1.md)、[MBConv pilot](ZCU104_MBConv_K3_E3_DMA_PILOT.md) | 接口定义可复用，环境与构建结果是各版本快照 |
| 后续ZCU104分类记录 | [NKSID_RL_CLASSIFICATION_RECORDING](zcu104_full_experiment/NKSID_RL_CLASSIFICATION_RECORDING.md) | 分类任务的记录协议；不替代检测分支或整个仓库的文件保留规则 |
| 分类算子证据与最小改动 | [分类证据表](zcu104_full_experiment/CLASSIFICATION_OPERATOR_EVIDENCE.md)、[NAGR 提案](zcu104_full_experiment/MINIMAL_OPERATOR_CHANGE_PROPOSAL.md) | 2026-09-14 汇总的 Skip/MBConv/DMDC/CFEM 证据分层与单槽位假设；分类 F1 不得当作检测定位证据 |
| 板端与HLS | [HLS工具链](../hls_lut_builder/README.md)、[板端工具](../hls_lut_builder/board_harness/README.md) | 源码与模板需要保留；同目录下的生成工程另行清理 |
| 外部数据与对标 | [EXTERNAL_SONAR_DATASETS](EXTERNAL_SONAR_DATASETS.md)、[benchmark campaign](benchmark_campaigns/ccf_ab_nksid_av7k325_v1.md) | 保留任务转换、环境和来源记录；不同数据集指标不混用 |

## 历史记录：保留原路径，退出日常执行入口

| 分类 | 文档 | 用途 |
|---|---|---|
| 早期设计草稿 | [修改前备份](../outputs/project_cleanup_20260909/method_docs_before.zip)、[legacy](legacy/README.md) | 旧architecture/method_design正文保存在ZIP同路径条目；现有页面已更新为当前实现 |
| 旧总览与状态快照 | [PROGRESS](PROGRESS.md)、[REVIEW](REVIEW.md)、[repository_complete_introduction](repository_complete_introduction.md) | 记录历史能力、审计和目录映射，不能据“当前”二字推断今日状态 |
| 2026-07-03审计 | [FIRST_PRINCIPLES_AUDIT](FIRST_PRINCIPLES_AUDIT_20260703.md)、[结构](audit_structure_discovery.md)、[入口](audit_entry_config.md)、[候选与硬件](audit_candidate_hardware.md)、[训练与指标](audit_search_training_metrics.md)、[部署与结果](audit_deploy_result_consumption.md) | 保留原审计结论与证据；后续实现变化需要单独核对 |
| Phase0 v3 | [板测](PHASE0_V3_BOARD_RESULTS.md)、[搜索](PHASE0_V3_DEPLOYABLE_SEARCH_EVIDENCE.md)、[权重重注入](PHASE0_V3_RETRAINED_BOARD_REINJECTION.md) | 历史协议、候选、板测证据 |
| Phase0 v4 | [算子预检](PHASE0_V4_SONAR_OP_PRECHECK.md)、[结果](PHASE0_V4_SONAR_RESULTS.md) | 历史fold-0、route和固定输入板测 |
| 严格40项LUT诊断 | [基线](strict40_plumbing_baseline.md)、[原始RL/Random诊断](strict40_eval5_rl_random_findings.md)、[探索修复](strict40_rl_exploration_validation_output.md)、[冻结索引](strict40_frozen_baseline_manifest.md)、[扩展计划](strict40_next_rl_expansion.md)、[周报](strict40_weekly_group_report.md)、[汇报稿](strict40_group_meeting_talk_script.md) | 2026年5月的不同证据和表述视角，不是7份可直接删除的重复全文 |
| 代理可靠性与归因 | [v1](PROXY_RELIABILITY_AUDIT.md)、[v2](PROXY_RELIABILITY_AUDIT_V2.md)、[G1归因订正](G1_ATTRIBUTION_CORRECTION.md) | 版本与阴性证据保留；无需因整理文件而启动这些实验 |
| 历史存储决策 | [RUN_STORAGE_AUDIT](RUN_STORAGE_AUDIT.md)、[RESULT_RETENTION_DECISIONS](RESULT_RETENTION_DECISIONS.md) | 2026年4月范围；不作为今日删除授权 |
| 分版本规划 | [refine-logs/MANIFEST](../refine-logs/MANIFEST.md) | 不同规划范围及短跳转入口，先保留再按引用归档 |

其他文档继续保留；未进入本导航不代表可以删除。完整逐文件检查线索见[9月8日审计附件](../outputs/project_cleanup_audit_20260908/DOCUMENTS.md)，新增文件须按当前工作区判断。

本轮整理先保留旧路径，避免破坏脚本引用和冻结证据；历史文件的实际移动或删除在完成引用核对后单独列项。
