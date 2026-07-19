# G1 NAS 方案 1 完成与独立审计报告

## 完成结论

方案 1 已完成。冻结候选 `rl_arch_135` 的唯一缺失单元 fold 4、seed 44 已完成原子落盘，NAS cohort（实验单元集合）从 14/15 补齐到 15/15。独立审计状态为 `PASS`，随后 wrapper 按授权边界自动停止；没有启动 SURE、scratch-v2、corruption、开放集、NAS 新搜索、HLS、route、AV7K325 板级或功耗实验。

## 最后单元结果

| 项目 | 数值 |
|---|---:|
| fold / seed | 4 / 44 |
| outer 样本数 | 537 |
| best epoch | 133 |
| macro-F1 | 0.7379394998 |
| top1 | 0.8603351955 |
| weighted-F1 | 0.8810828501 |
| top5 | 0.9981378026 |
| NLL | 0.5527984840 |
| Brier | 0.2601159669 |
| ECE | 0.1992532937 |
| AURC | 0.0347131381 |
| failure AUROC | 0.8481673882 |
| failure FPR95 | 0.3896103896 |

## 完整 15 单元摘要

| 指标 | 冻结 NAS 候选均值 | 标准差 | ImageNet 预训练 MobileNetV2 均值 |
|---|---:|---:|---:|
| macro-F1 | 0.6936187231 | 0.0292365606 | 0.9316318801 |
| top1 | 0.8132039345 | 0.0281462632 | 0.9778262427 |
| weighted-F1 | 0.8496698583 | 0.0210160539 | 0.9785838642 |
| ECE | 0.1767009518 | 0.0248244248 | 0.3924215776 |
| AURC | 0.0694637491 | 0.0188169079 | 0.0200319452 |
| failure AUROC | 0.7960984954 | 0.0304514661 | 0.5738570054 |
| failure FPR95 | 0.6475708589 | 0.1109373082 | 0.9052270144 |

校准和失败预测指标必须逐项解释，不能由 ECE 或 AUROC 单项推出整体模型优劣。主分类结论仍只依据预先声明的 macro-F1。

## 配对统计

- 比较方向：冻结 NAS 候选减去 ImageNet 预训练 MobileNetV2。
- 配对单位：5 个 outer fold × 3 个 seed，共 15 个单位。
- macro-F1 均值差：`-0.2380131570`。
- 10,000 次按 fold 分层的配对 bootstrap 95% CI：`[-0.2639684924, -0.2126668049]`，bootstrap seed 为 `20260717`。
- NAS 胜出：`0/15`；平局：`0/15`。
- 精确双侧配对符号翻转置换检验：`p=0.00006103515625`。
- 配对效应量 Cohen's dz：`-5.4963062324`。
- top1 配对均值差：`-0.1646223082`，95% CI `[-0.1835696116, -0.1468349210]`。
- weighted-F1 配对均值差：`-0.1289140059`，95% CI `[-0.1432623489, -0.1149640407]`。

完整结果确认此前的中期判断：补齐最后一个单元提高了证据完整性，但没有改变该候选相对预训练基线明显落后的方向。

## 独立审计与哈希

- 独立审计：`g1_nas_independent_audit_20260716.json`，状态 `PASS`，15 条 record，错误数 0。
- source freeze：`PASS 556/556`；manifest SHA256 `8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc`。
- 最后 checkpoint SHA256：`2b52ef8a8390e623dcc1dad1a9cea0e145734dafb6f7e2ce45d5a68abeb14f9b`。
- 最后 prediction SHA256：`4e2f34c0d87baa890c22622f1d4219cf3e4c6f5e8f754475477b5d63c50a826f`。
- 最后 split SHA256：`fa39042e7ca23ad9a88ccfe98e5864e0cf8215d4bda86c9f0863c709cd194385`。
- protocol summary SHA256：`59a1d4af1ed131a4a21f4b76b451b99d0d369817365eaf42d5c7d7e0e4b26aa5`。
- 最后 run record SHA256：`f20fbae72770af02023e4e3a2dfa321872c1cedf8097c5f58b558481c62b39d5`。
- 独立审计文件 SHA256：`11c0d8d6ae4fb6189074df6efa001b4c1645624c82e6b94097f8c30868abdcfe`。

## 启动故障说明

首次限定启动在训练前因 Windows PowerShell 5.1 误读无 BOM UTF-8 中文批准 JSON 而失败。该次启动没有生成 checkpoint、record 或 prediction，也没有启动下游实验。修复方式仅为把机器 JSON 的中文值改成 Unicode 转义；候选、指标、数据、划分、协议、source freeze 和最大授权范围均未改变。完整中文故障卡见 `g1_nas_resume_incident_20260717_1726.md`。

## 科学声明边界

- 该 15/15 cohort 可用于报告冻结架构的 NKSID 协议结果。
- 架构来自历史 fold0 选择流程，因此 `nas_generalization_claimable=false`；不能用本结果证明 NAS 搜索方法的无偏泛化能力。
- acquisition/mission group 元数据仍缺失，不能声称跨任务组泛化。
- 原 scratch cohort 的 provenance 存在独立审计缺口；与 scratch 的数值只能作为诊断，不能替代 scratch-v2。
- SURE 和 scratch-v2 均为 0/15，因此完整 T2/F6 仍为 `PENDING`。
- 当前不需要板卡；功耗继续为 `NOT_MEASURED`。

## 停止状态

- wrapper 已在 `continue_downstream_closed_set_chain` 门禁处停止。
- 匹配训练进程：0。
- Windows 计划任务 `Codex_HWNAS_G1_20260716`：Disabled。
- 下游授权：false。
- 已消费的方案 1 批准文件已改名为 `g1_nas_underperformance_user_decision_20260717.consumed.json.txt`；wrapper 固定可执行批准路径当前不存在。
- 未经用户新的明确决定，不得启动后续实验或改变协议。
