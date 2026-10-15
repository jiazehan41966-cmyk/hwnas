# 当前轻量主线：并行补齐任务

日期：2026-10-07。用户要求新开并行任务补齐从 NAS 到部署的缺口。当前环境没有创建侧边栏独立任务的工具，因此实际使用本任务内三个子代理；本页不将它们表示为独立侧边栏对话。

## 固定范围

- 主线为 `elastic_neck72_v1`，固定完整 B1 骨干、48 通道轻量融合、四尺度直接检测头；不是旧 144 架构空间。
- 旧 `sonar_joint_v2_elastic_v1` 的独立训练、三档量化、六映射需求、Pareto 与三种子确认接口作为复用基础。
- 引入 RL 属于新的搜索策略版本，不能重命名历史 V2 结果冒充 RL 实验。
- 不覆盖已有冻结、代码快照、训练或硬件结果，保留工作区无关修改。
- 本批先完成本地软件实现和对应测试；未知硬件成本、缺少正式计算预算或部署证据时继续拒绝正式执行。测试和诊断数据不能成为研究结果。

## 并行责任

| 子任务 | 负责内容 | 新增代码边界 | 状态 |
|---|---|---|---|
| light_rl_search | RL 适配、奖励、Pareto、预算与恢复、搜索调度 | `tools/sonar_light_rl/` 及专属测试 | 本地诊断/回放实现，正式评估驱动未接通 |
| light_hardware_cost | 新空间成本后端、来源验收、缺测队列 | `tools/modnas_comparison/joint_cost_backend.py` 及专属测试 | 本地实现，真实测量待补 |
| light_deployment_handoff | 最终候选身份、整数工件验收、部署阶段交接 | `tools/sonar_light_deploy/` 及专属测试 | 软件工件检查实现，确认与硬件阶段待完成 |
| 当前任务 | 接口协调、配置及文档、联合验证 | 本文和独立配置/验证目录 | 联合验证结果见下文 |

共享验证输出根：`results/sonar_light_chain_20261007/`；各子任务使用 `rl/`、`hardware/`、`deployment/` 子目录。不得将不存在的测量注册为有效硬件成本。

## 汇合验收

1. 控制器遵守 72 架构合法编码与 PAN 条件变量；联合配置保留量化和映射身份。
2. 未知成本进入待测列表，不作为负奖励；已知超预算与数据/身份错误分开处理。
3. Pareto、奖励、控制器更新各有独立职责，搜索日志与恢复可追溯。
4. 真实训练/量化与硬件成本绑定同一配置和来源；正式运行在缺项时保持阻断。
5. 部署交接绑定具体 checkpoint、量化、校准、映射和整数图；无路由或板返回不能宣称通过。
6. 子任务测试通过后，执行跨模块集成检查；记录通过范围和仍缺的实验/硬件证据。

## 实现边界

### RL 搜索软件

独立策略版本 `sonar_light_rl_v1` 复用现有 `Controller` 与 `RewardFunction`。原控制器的三个输出头分别适配为架构、量化和映射，乘积分布在联合可选集合上重新归一化；使用实际采样分布的对数概率更新策略。奖励以 `mAP50_95` 和固定预算归一化的硬件代价计算。未知功耗不进入奖励，默认权重与回合数仅供诊断，尚未冻结为正式实验协议。

诊断/回放已包含采样、筛查、分训练预算的 Pareto 档案、奖励、策略更新、晋升和断点恢复。40/60/80 在合成诊断中是评价记录的预算标签，**没有实际训练这些 epoch**。恢复绑定策略、Adam 状态、基线、随机状态、配置、来源代码及证据文件；不能把更换来源后的结果续接到旧状态。

`run` 当前执行真实预检并返回阻断，不是已具备正式实验调度能力。正式 GPU 时数计费评估驱动、拒绝候选审计调度、三种子确认与最终前沿选择仍需接通。已有 `train_candidate`、`quant_candidate` 等函数本轮未被绕过或启动。

### 成本查询与补测

新增后端满足旧预检要求的 `load_verified_joint_cost_backend(...).query(architecture, profile, mapping_policy)` 接口。只接收与具体算子、精度、数值/接口合同、映射和工具报告绑定的证据；混精度检测头包含四个显式位宽转换节点。提供 `init`、`queue`、`import` 本地入口，导入失败不覆盖原表。

空表表示缺测，不能将 1296 个联合配置判为全部实测失败。算子 HLS 延迟串行求和与单核资源峰值仅用于前置筛查，不表示全网布局布线资源、板端延迟或功耗。六种映射需求仍须各自绑定具体实现，不能只改名称复用数值。

### 部署交接

`prepare_handoff` 接收具体候选身份、checkpoint 与训练回执、PTQ 导出及校准身份。`inspect_ptq_directory` 检查保存工件的哈希、图拓扑、权重 dtype、混精度边界和逐节点一致性见证。检查现有诊断权重导出不代表获得最终训练权重，也不是重新执行推理或硬件一致性验证。

三种子确认仍须重放底层准入证据。硬件参数包、HLS 数值一致性、布局布线、上板一致性、完整验证集精度、端到端延迟及功耗均单独保持未验证。凭一个 `PASS` 标记不能升级状态。

## 集成配置与复现入口

- `configs/search/sonar_joint_v2_elastic_light_cost_v1.yaml`：保留原 V2 协议，只将硬件查询接到新后端和本次空证据表。
- `configs/search/sonar_light_rl_v1.yaml`：独立 RL 诊断协议；正式协议冻结为 false，GPU 型号和总 GPU 时数上限为空。
- `results/sonar_light_chain_20261007/integration/`：联合检查、诊断和正式执行阻断证据。

在仓库根目录运行以下 PowerShell 命令；输出目录必须未存在，避免覆盖既有证据：

```powershell
& '.venv_cuda/Scripts/python.exe' -m tools.sonar_light_rl.run plan --protocol configs/search/sonar_light_rl_v1.yaml --output results/sonar_light_chain_manual/plan
& '.venv_cuda/Scripts/python.exe' -m tools.sonar_light_rl.run diagnostic --protocol configs/search/sonar_light_rl_v1.yaml --output results/sonar_light_chain_manual/diagnostic
& '.venv_cuda/Scripts/python.exe' -m tools.modnas_comparison.joint_cost_backend queue --manifest results/sonar_light_chain_20261007/hardware/joint_cost_registry.json --output results/sonar_light_chain_manual/measurement_queue.json
```

`plan` 应输出明确阻断；`diagnostic` 只执行合成软件验证；`queue` 只生成补测需求，不启动 HLS。使用 `run` 时，当前预期返回码为 2，并保存阻断报告。

## 本轮验证结果

联合验证记录位于 `results/sonar_light_chain_20261007/integration/verification_summary.json`，内含执行源文件和配置的 SHA256。

| 检查 | 实际结果 | 证据边界 |
|---|---|---|
| RL、部署及原接口回归 | 61 项通过 | 本地软件检查 |
| 成本后端及跨模块边界 | 10 项通过 | 合计 71 项；初次 W8A8 合同错误已修复，初次失败报告保留 |
| 独立配置的合成诊断 | 14 次采样、12 次更新、3 档预算标签 | 2 次未知成本没有奖励或更新；不可作为研究结果 |
| 真实空成本表接 RL | 0 次控制器更新、空 Pareto | 缺测未被误罚，未训练 |
| 诊断候选接部署交接 | 1 份交接保持未就绪 | 无最终 checkpoint、确认或硬件结论 |
| 正式 `run` | 返回码 2，`BLOCKED_NOT_LAUNCHABLE` | 新后端已接通，完整成本/预算/共同协议/正式驱动仍阻断 |
| 实际成本需求队列 | 1296 联合配置、77 类算子形状/精度/合同需求、462 个任务 | 任务仍需冻结具体映射；完整合格成本记录 0 |

三个子任务的详细说明分别见：

- `results/sonar_light_chain_20261007/rl/IMPLEMENTATION_REVIEW.md`
- `results/sonar_light_chain_20261007/hardware/README.md`
- `results/sonar_light_chain_20261007/deployment/IMPLEMENTATION_REPORT.md`

只读复核的六个历史 stem 记录不能替代完整成本表。最佳 stem 为 49469.33 us；仅替换该 stem 且其余旧映射不变时，已知骨干部分串行延迟和为 1587740.610 us，仍超过 120000 us。此为特定部分映射下的负结果，不能外推为新空间全部架构/映射都已测失败。

## 后续依赖顺序

1. 冻结具体硬件映射和完整数值/接口合同，补齐测量并验证导入；先判断轻量主线在预算下有无可行候选。
2. 冻结 RL 回合数、奖励、GPU 型号/总时数以及与 MODNAS 的共同决策和计费协议。
3. 实现并验证真实评估计费与最终确认调度，再运行已满足准入条件的搜索；保持独立搜索种子与内部确认种子区分。
4. 用完整训练、多种子和量化准入数据重新计算最终 Pareto 前沿，选择并冻结真实部署权重。
5. 依次完成硬件参数导出、HLS、布局布线、上板和实测验收；各层证据单独记录。

本轮补齐的是可检查的软件适配与证据交接基础，不是完整正式 NAS 实验或 FPGA 部署成果。
