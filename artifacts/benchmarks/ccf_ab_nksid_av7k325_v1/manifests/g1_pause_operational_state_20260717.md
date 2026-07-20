# G1 暂停运行状态卡

## 当前状态

- 记录时间：2026-07-17T15:35:52.2260795+08:00。
- 决策状态：等待用户决定。
- NAS 完成度：14/15；唯一缺失单元为 fold 4、seed 44。
- Windows 计划任务 `Codex_HWNAS_G1_20260716`：`Disabled`。
- 匹配 `run_eval_protocol`、`g1_rl_arch_135_legacy_selected` 或 `run_g1_persistent_20260716` 的 Python 训练进程：0。
- 用户批准文件 `g1_nas_underperformance_user_decision_20260717.json.txt`：不存在。
- source freeze：PASS 556/556，无缺失、无意外文件、无变更文件。

## 恢复条件

未经用户明确选择，不得重新启用计划任务、创建批准文件、运行剩余 NAS 单元、启动 SURE/scratch-v2，或改变候选与协议。恢复时必须同时重新核验 source freeze、批准文件与计划任务动作。

## 证据绑定

- 暂停 manifest SHA256：`f4815a65aa48b723abe300dd618d6f8ac8606ba5237ce9215f7e1fabd159db86`。
- 中文归档规则 SHA256：`7fe9b2d34052e2e451cd3db64db04be937369ac583bd0cf53c83333433c272ac`。
- source freeze manifest SHA256：`8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc`。

本状态卡不增加正式 T2/F6 证据，不改变 G1、G2、G3、G4、G5 或功耗状态。
