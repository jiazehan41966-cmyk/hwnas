# 声呐网络—量化—FPGA 当前执行状态

核查日期：2026-09-22。适用范围是声呐检测 V0.1 及其当前联合搜索，分类项目和外部数据迁移的历史结果保持各自协议。本页的运行数字是带时间戳的观测，不是持续监控。

## 已核实且正在执行

- 17:02:59（北京时间）的只读远端观测：`sonar_joint_v1` 正常运行，40 epoch 阶段完成 **23/144**，下一架构 epoch 9。GPU1 执行联合搜索；GPU0 执行 P0 算子 seeds43/44 确认。没有重复启动任务或中断其他进程。
- 远端隔离树的 **190 个冻结文件及完整 verify 通过**。本地有后续 H4 修复及新增文件，不能据此覆盖远端或重写 freeze。冻结时 `training_started=false` 不再代表当前运行状态。
- 独立多保真历史证据为 144→108→72，六架构三种子共18条；新 P0 算子池是20/20单种子筛查后再确认，与此前失败的 V0.2-P0 单个原型属于不同实验。
- 重新调用现有精确查表接口确认 **2592/2592** 组合有算子级 HLS 代价；旧冻结 B1/C/D 条件域仍是 **30** 个合法候选、**4** 个精度—延迟前沿点。该结果不等于正在训练的新联合搜索完成。

## 本轮同步完成的工作

| 工作 | 交付物 | 验证范围 |
|---|---|---|
| 运行与冻结绑定审计 | [runtime audit](../results/sonar_runtime_audit_20260922/audit.json) | 远端只读实际进程、epoch、GPU归属、冻结校验；本地漂移单独记录 |
| 全流程证据重算 | [workflow audit](../results/sonar_workflow_audit_20260922_v1/README.md) | 23项一致性检查；旧报告的过时字段显式区分；保留代理搜索否决 |
| 局部 FPGA 证据验收 | [DMA evidence audit](../results/detector_evidence_audit_20260922_v2/README.md) | PW 614400 bytes、FPN 2457600 bytes板端输出均零差异；10项检查通过 |
| 新搜索结果汇总范围修复 | [独立收尾器](../tools/sonar_search_finalize/README.md) | 新架构不再被旧BCD白名单排除；按本轮收据、冻结门槛计算条件域和Pareto |

PW/FPN 见证实际路由时钟为 **187.512 MHz**，WNS（最差建立时间裕量）**+1.118 ns**。这是合成参数的局部DMA见证，未绑定最终网络和训练权重。该见证的 top-minus-core 资源差不能写成整网 shell reservation；单次事务时间也不能写成完整检测器端到端延迟。

新的收尾器位于被冻结训练源码集合之外。原训练入口及其旧汇总函数保持冻结；本轮确认和量化完成后，需使用独立收尾器验收真实收据并重建最终结果，不能直接采信旧 `joint_pareto.json`。当前本地预检正确返回等待完整实验产物，不宣称已完成最终选择。

## 未完成项与依赖

| 未完成项 | 当前状态 | 下一项有效工作 |
|---|---|---|
| 新联合搜索、12架构三种子与量化 | 正在执行 | 由现有worker继续；完成后用独立收尾器校验，保留弃选抽查门槛 |
| P0新算子多种子确认 | 正在执行 | 配对确认完成后判断是否纳入后续讨论，不用单种子筛查宣布收益 |
| 最终checkpoint绑定的整数整网导出 | 等待最终候选/权重 | 对实际模型导出并验证数值一致性；混合精度需匹配参数包、帧宽和loader合同 |
| 完整图的硬件执行器 | 局部通路已验证，整网未完成 | 接通逐节点执行、双输入Add、DDR生命周期、多头输出和错误恢复 |
| 整网资源与时序 | 未建立 | 对实际完整设计综合与布局布线，记录真实时钟、资源和系统预留 |
| 板端精度、重复端到端延迟和功耗 | 未建立 | 在候选/权重/bitstream身份一致后测量，保留样本、计时及功率记录 |
| 自动代理搜索反馈 | `SURROGATE_NOT_JUSTIFIED` | 保留失败结论；使用已有多保真、验证和硬件证据进行约束筛选 |

校准和结构数据统计使用 train，冻结协议下的选择使用 valid；test 不回流到结构、量化或映射选择。历史外部迁移的 test 评估遵循独立协议，不能笼统表述为整个仓库从未访问 test。

## 复算入口

运行和硬件审计的每次输出必须使用新目录；不要覆盖历史快照。以下脚本均沿用当前仓库工具，不启动训练或硬件：

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_runtime_audit/run.py --remote --output results/NEW_RUNTIME_AUDIT
.venv_cuda/Scripts/python.exe -B tools/detector_preflight/audit_evidence.py --output results/NEW_DMA_AUDIT
.venv_cuda/Scripts/python.exe -B tools/sonar_workflow_audit/run.py --runtime-audit results/NEW_RUNTIME_AUDIT/audit.json --hardware-audit results/NEW_DMA_AUDIT/audit.json --recompute-cost --output results/NEW_WORKFLOW_AUDIT
```

检查通过是对应范围的证据完整性/一致性结果，不是完整研究结论或整网部署准入。
