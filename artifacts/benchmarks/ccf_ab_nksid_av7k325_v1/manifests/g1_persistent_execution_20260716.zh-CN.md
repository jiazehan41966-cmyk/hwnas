# G1 持久化执行记录（中文伴随档案，2026-07-16 至 2026-07-17）

## 来源

- 英文历史原件：`g1_persistent_execution_20260716.md`。
- 原件 SHA256：`fd7e3b740594bb87f082c2c666df870fb04494be31b72ccef1a3d2c0837685a0`。
- 本文件将原件的执行设计、关键时间线、逐单元结果和暂停边界整理为中文；原件保持只读。

## 持久化执行设计

- Windows 计划任务：`Codex_HWNAS_G1_20260716`；原设计含登录触发和每 30 分钟重复触发，`MultipleInstances=IgnoreNew` 防止重复训练进程。
- 运行 wrapper：`artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/run_g1_persistent_20260716.ps1`。
- 预训练正式运行于 2026-07-16 14:34:12 CST 启动，fingerprint 为 `e9cc0c17cb17773396673c06cfa15157b7f8d8559454134b2da9a7ab76a47f01`。
- 启动时 source freeze 为 PASS 556/556。
- 重启后只允许使用同一 canonical `run_eval_protocol.py` 和 `--resume`；只有 fingerprint 兼容时才跳过已完整写入的原子单元，禁止 `--force`。
- 每种闭集方法达到 15/15 后，必须由 `audit_g1_run.py` 独立核验 checkpoint、预测、记录 SHA、指标复算、fold 不相交和 2,617 样本并集，才允许进入下一方法。
- 原 scratch 科学结果仍存在，但其生成的 `code_patch.diff` 曾被失败 resume 覆盖，因此只能诊断使用；正式链计划新建 `g1_mobilenet_v2_scratch_v2`。
- 四种闭集方法全部达到 60 个审计单元后，才可生成 T2/F6 并执行冻结 corruption 协议；开放集 CE+MSP、DMCL、PLUD 同样必须逐方法完成 15 单元及独立审计。
- 上述原自动链已被后续用户决策门禁覆盖；当前不得自动继续。

## 预训练 MobileNetV2 时间线与结果

- 2026-07-16 16:14 首批原子单元出现：fold0/seed42 macro-F1 `0.9108403453`、top1 `0.9730769231`、best epoch 80；fold0/seed43 macro-F1 `0.9188535215`、top1 `0.975`、best epoch 62。
- 17:26 前又完成 fold0/seed44 与 fold1/seed42、43，运行达到 5/15；18:12 的只读部分审计重新核验文件哈希、逐行 provenance（来源）、划分与指标。
- fold1/seed44 完成后达到 6/15；合成数据仅用于验证 F5/F7 的 SVG、PDF、300 dpi PNG、source CSV 和 meta JSON 生成合同，不是科学结果。
- 2026-07-17 08:19 已有 14 个完整原子单元；显式合盖造成的休眠时间不计入训练吞吐量或 GPU-hours，GPU 功率读数只作运行健康诊断，不是实验功耗。
- 08:56 fold4/seed44 完成，包含 537 个唯一 outer 样本：macro-F1 `0.9533467761`、top1 `0.9851024209`、weighted-F1 `0.9854032483`、best epoch 52。
- 预训练 15 单元汇总：macro-F1 `0.9316318801 ± 0.0241884368`，top1 `0.9778262427 ± 0.0081714897`，weighted-F1 `0.9785838642 ± 0.0080011456`，统计单位为 5 folds × 3 seeds。
- 独立审计 `g1_pretrained_independent_audit_20260716.json` PASS，SHA256 为 `f7e5c6c1acb020fa3eaf38449ba93c2ea66df4e8fcf34fa1638e0538fcca5b70`。
- 审计复算分类指标和文件哈希，核验逐行来源、同 fold 三个 seed 的 outer 样本一致性、fold 间不相交及五 fold 合计 2,617 个样本。
- source freeze 保持 PASS 556/556；manifest SHA256 为 `8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc`，保留 archive SHA256 为 `376e9e162240a2e15dd8456882c9f209ad16bfc8b8da5244fac51342c8d4410d`。

## NAS 候选 14 个原子单元

候选 `frozen_nas_champion` 对应 `g1_rl_arch_135_legacy_selected`，于 2026-07-17 08:56:13 通过 canonical 入口启动。下表保留英文原件的 14 个单元核心指标；fold0–3 各为 520 个 outer 样本，fold4 各为 537 个。

| Fold | Seed | macro-F1 | top1 | weighted-F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 0.6847830911 | 0.8019230769 | 0.8455041710 | 150 |
| 0 | 43 | 0.6741714543 | 0.7807692308 | 0.8291172437 | 143 |
| 0 | 44 | 0.6437996521 | 0.7769230769 | 0.8177937310 | 78 |
| 1 | 42 | 0.7049382928 | 0.8442307692 | 0.8727932731 | 87 |
| 1 | 43 | 0.6551299125 | 0.8096153846 | 0.8328275656 | 99 |
| 1 | 44 | 0.6734461530 | 0.7692307692 | 0.8233374491 | 138 |
| 2 | 42 | 0.6896732794 | 0.8134615385 | 0.8552309117 | 84 |
| 2 | 43 | 0.7169423734 | 0.8519230769 | 0.8752139839 | 75 |
| 2 | 44 | 0.6767176192 | 0.8076923077 | 0.8456112892 | 93 |
| 3 | 42 | 0.7307331919 | 0.8192307692 | 0.8566289503 | 97 |
| 3 | 43 | 0.7420635546 | 0.8423076923 | 0.8788305336 | 123 |
| 3 | 44 | 0.6995334498 | 0.8096153846 | 0.8514615757 | 94 |
| 4 | 42 | 0.6737691725 | 0.7839851024 | 0.8250994038 | 86 |
| 4 | 43 | 0.7006401504 | 0.8268156425 | 0.8545149424 | 109 |

## 原子审计要点

- 每个已接纳单元均核验 run record、prediction JSONL 与 checkpoint 的 SHA256 绑定。
- 逐样本 ID 与 outer position 唯一；target 与数据标签一致；prediction 与 logit argmax 一致；confidence 与 softmax 最大值误差在冻结容忍范围内。
- 每行均绑定 fold、seed、split、method、checkpoint、run fingerprint、split/data/code state 和 `PENDING` claimability。
- 分类指标、混淆矩阵、八类 F1、ECE、AURC、失败 AUROC/FPR95 均由持久化预测独立复算；浮点差异低于协议容忍范围。
- 同 fold 三个 seed 的 outer 样本 ID 与 target 相同，seed-specific inner train/validation 划分不同；fold 间 outer 集合互斥。
- 可复用部分审计只说明当前已观察单元有效，不授予方法级正式可声明性。

## 第 14 个单元与低性能暂停

- fold4/seed43 于 2026-07-17 14:24 完成，537 个样本；split SHA256 `56e38b9fb478593a16b3a21b04e58bf27622fb31c023e0018683d475c33f1fb4`，checkpoint SHA256 `f2493868d56a7680186c0525501f4c0c97ab1c0b224c8a197ca61a82a4fbaa7d`，prediction SHA256 `0de0e1ce5ff3fe7dc6e5e34af835d9c6d69821bc28c0135d76ba07ea69776632`。
- 14/14 部分审计零错误，保留审计 SHA256 `0e8da5427b19fb8ddf2aefb871ee4965839c21f17e294887b2668b50dc6e507c`。
- 14 个配对单元上，NAS macro-F1 均值 `0.6904529534`，预训练为 `0.9300808161`；配对均值差 `-0.2396278627`，14 个差值全部为负。10,000 次、seed `20260717` 的配对 bootstrap 95% CI 为 `[-0.2612163648, -0.2169519822]`。
- 根据用户要求，运行在原子写入完成后停止；fold4/seed44 没有 checkpoint、record 或 prediction。
- 历史原件记录暂停当时计划任务为 Ready；后续状态卡已将其更新为 Disabled。当前 Python 训练进程为 0，SURE/scratch-v2 未启动，批准文件不存在。

## 当前恢复边界

状态为 `PAUSED_PENDING_USER_DECISION`。用户必须决定是否仅补齐最后一个 NAS 单元、是否继续下游闭集方法，或是否在新协议和新 source freeze 下更换候选。没有独立、绑定暂停 SHA 的批准文件时，不得恢复。

本中文伴随档案不增加正式 T2/F6 证据，也不把中期统计提升为完整 15 单元结论。

## 方案 1 完成补记（2026-07-17 17:53）

- 用户明确批准“先完成方案 1”，机器批准文件只开放 `resume_nas_to_15`，并保持 `continue_downstream_closed_set_chain=false`、`allow_protocol_or_candidate_change=false`。
- 首次启动在训练前因 Windows PowerShell 5.1 误读无 BOM UTF-8 中文 JSON 而停止；未生成任何科学数据。机器 JSON 改用 Unicode 转义后，重试前重新通过 source freeze 556/556，并保持唯一缺失单元为 fold4/seed44。
- fold4/seed44 于 17:53 完成：537 个样本，best epoch 133，macro-F1 `0.7379394998`，top1 `0.8603351955`，weighted-F1 `0.8810828501`，ECE `0.1992532937`，AURC `0.0347131381`。
- checkpoint SHA256：`2b52ef8a8390e623dcc1dad1a9cea0e145734dafb6f7e2ce45d5a68abeb14f9b`；prediction SHA256：`4e2f34c0d87baa890c22622f1d4219cf3e4c6f5e8f754475477b5d63c50a826f`；split SHA256：`fa39042e7ca23ad9a88ccfe98e5864e0cf8215d4bda86c9f0863c709cd194385`。
- 15/15 独立审计 `PASS`，record count 15，source freeze `PASS 556/556`。完整 NAS 平均 macro-F1 为 `0.6936187231`；相对预训练 MobileNetV2 的配对均值差为 `-0.2380131570`，10,000 次按 fold 分层 bootstrap 95% CI 为 `[-0.2639684924, -0.2126668049]`，NAS 胜出 `0/15`。
- wrapper 在独立审计后记录 `G1_DYNAMIC_RUNS_ACCEPTED_30_OF_30`，随后因下游许可为 false 记录 `PAUSED_USER_DECISION_NOT_APPROVED` 并退出。已消费批准文件随后改名为 `g1_nas_underperformance_user_decision_20260717.consumed.json.txt`，wrapper 固定批准路径重新为空。计划任务保持 Disabled，训练进程为 0；SURE、scratch-v2 及所有后续实验均未启动。
- 当前结论仅适用于冻结的历史 fold0 选择架构；`nas_generalization_claimable=false`，不得写成 NAS 搜索方法的无偏泛化结论。完整 T2/F6 仍等待 scratch-v2 与 SURE。
