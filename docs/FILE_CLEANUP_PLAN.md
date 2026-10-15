# 文件夹整理执行入口与关键实现路径

核对日期：2026-09-09。用途：整理整个仓库、减少冗余、保留方法实现及其复现依赖。本文不安排新实验。

**整理时只按本页的四个批次推进。** 9月8日的全量审计作为查路径的附件，不要求逐项完成706个目录的处理。已按获准清单删除308个未变化缓存文件（4,453,399字节）；169项跳过。方法正文、配置版本和证据口径已核对修正，旧文档完整备份；未移动源码或实验证据。当前方法读[method_design](method_design.md)，源码关系读[architecture](architecture.md)。

## 1. 先保留这些实现路径

以下是方法定位索引，不是“没列出来就删除”的白名单。共享依赖、测试、未提交工作和既有实验输入一并保留。

| 能力 | 使用入口 | 关键实现及依赖 |
|---|---|---|
| 配置装配与运行记录 | [run_search.py](../run_search.py) | [runtime.py](../src/hwnas_fpga/runtime.py)、[interfaces.py](../src/hwnas_fpga/interfaces.py)、[experiment.py](../src/hwnas_fpga/experiment.py) |
| 搜索空间与架构表示 | [run_search_space_probe.py](../run_search_space_probe.py) | [search_space/space.py](../src/hwnas_fpga/search_space/space.py)：SearchSpace、ArchitectureSpec、BlockSpec |
| RL（强化学习）搜索 | [run_search.py](../run_search.py) | [search/rl_searcher.py](../src/hwnas_fpga/search/rl_searcher.py)：Controller、RewardFunction、RLSearcher；[search/searcher.py](../src/hwnas_fpga/search/searcher.py) |
| Pareto（多目标非支配）筛选 | 搜索过程及结果汇总 | [search/pareto.py](../src/hwnas_fpga/search/pareto.py)、[search/efficiency.py](../src/hwnas_fpga/search/efficiency.py) |
| 网络与声呐算子 | [models/builder.py](../src/hwnas_fpga/models/builder.py) | MBConvBlock、DMDCConvSonarBlock、CFEMSonarBlock、build_block、build_model；保留同文件内其他算子及其历史实验 |
| 数据读取与划分 | [data/dataset.py](../src/hwnas_fpga/data/dataset.py) | [data/protocol.py](../src/hwnas_fpga/data/protocol.py)、[data/external_sources.py](../src/hwnas_fpga/data/external_sources.py)、[configs/datasets](../configs/datasets) |
| 训练与独立评估 | [run_retrain.py](../run_retrain.py)、[run_eval_protocol.py](../run_eval_protocol.py) | [training](../src/hwnas_fpga/training)、[configs/evaluation](../configs/evaluation)、[experiment_contract.py](../src/hwnas_fpga/experiment_contract.py)、[research_gates.py](../src/hwnas_fpga/research_gates.py) |
| 硬件代价模型 | [runtime.py](../src/hwnas_fpga/runtime.py) | [hardware/mobilenetA_operator_cost.py](../src/hwnas_fpga/hardware/mobilenetA_operator_cost.py)、[hardware/cost.py](../src/hwnas_fpga/hardware/cost.py)、[hardware/lookup_table.py](../src/hwnas_fpga/hardware/lookup_table.py)、[hardware/report_parser.py](../src/hwnas_fpga/hardware/report_parser.py) |
| INT8（8位整数）与导出 | [run_eval_int8.py](../run_eval_int8.py)、[run_export.py](../run_export.py)、[run_infer.py](../run_infer.py) | [deploy](../src/hwnas_fpga/deploy)：int8_reference、fixed_point、ptq_eval、qat、parity_gate、export等共享实现 |
| HLS（高层次综合）与硬件模板 | [hls_lut_builder/scripts](../hls_lut_builder/scripts) | [templates](../hls_lut_builder/templates)、[include](../hls_lut_builder/include)、[configs](../hls_lut_builder/configs)；尤其保留 dmdc_conv_sonar.cpp.tmpl、cfem_sonar.cpp.tmpl 及其依赖 |
| 布局布线与板端接口 | [hardware/zcu104_dma_shell_v1](../hardware/zcu104_dma_shell_v1) | rtl、hls、vivado、software、remote；[hls_lut_builder/board_harness](../hls_lut_builder/board_harness) 的 scripts、modules、templates、configs |
| 板级验证与功率记录 | [run_dynamic_validation.py](../hls_lut_builder/board_harness/scripts/run_dynamic_validation.py)、[capture_zcu104_onboard_power.py](../scripts/capture_zcu104_onboard_power.py) | [hardware/deployment_binding.py](../src/hwnas_fpga/hardware/deployment_binding.py)、[hardware/power_measurement.py](../src/hwnas_fpga/hardware/power_measurement.py)、板端运行脚本 |
| 方法对照与历史实现 | [search](../src/hwnas_fpga/search)、[benchmarks](../src/hwnas_fpga/benchmarks) | Random、Aging、Proxyless、SPOS、backbones及参考库；暂时不用不代表代码冗余 |
| 新增检测分支 | [scripts/run_detection_experiment.py](../scripts/run_detection_experiment.py) | [models/detection.py](../src/hwnas_fpga/models/detection.py)、[data/detection.py](../src/hwnas_fpga/data/detection.py)、[training/detection_trainer.py](../src/hwnas_fpga/training/detection_trainer.py)、[training/detection_metrics.py](../src/hwnas_fpga/training/detection_metrics.py)、[hardware/detection_cost.py](../src/hwnas_fpga/hardware/detection_cost.py) |
| 复现与回归检查 | [tests](../tests)、[scripts/freeze_experiment_source.py](../scripts/freeze_experiment_source.py) | [pyproject.toml](../pyproject.toml)、[configs](../configs)、[patches](../patches)、源代码冻结清单、数据划分、环境锁定记录 |

当前已不存在的旧 `src/hwnas_fpga/hardware/zcu104_operator_cost.py` 不作为新的导航目标；其删除是本轮整理前已有的工作区变动，本页没有恢复或进一步删除它。

## 2. 整个文件夹分成四类

| 类别 | 路径范围 | 整理动作 |
|---|---|---|
| 方法实现 | 根目录入口、src、configs、tests、patches；HLS/板端源码目录 | 保留原路径，先理清导航；不要为了美观打断 import、脚本和配置引用 |
| 关键输入与证据 | data、artifacts、results中不可再生的原始记录、checkpoint、源码快照、bitstream与报告 | 建保留清单；确保候选、代码、输入、输出及哈希能相互追溯 |
| 历史资料 | strict40/Phase0记录、旧设计、旧计划、旧可视化、对话导出 | 从主入口退出；确认引用后集中归档，保留原始日期和适用范围 |
| 可再生产物 | Python/pytest缓存、HLS/Vivado工程中间文件、确认可再生的预览图与日志副本 | 精确列项后清理；工程名称含cache、failed或smoke不足以证明可删 |

注意：`hls_lut_builder` 同时包含方法实现和大体积产物，不能整目录删除。`artifacts` 也混有工程缓存，不能只凭顶层目录名决定保留或删除。

## 3. 只按四个批次整理

### 第一批：建立清楚的入口

- [x] 建立本页的关键实现路径索引，并从根 README 链接到这里。
- [x] 建立[按用途分类的文档导航](INDEX.md)，为早期设计与旧状态页标明适用范围，保留原路径。
- [x] 修正主入口、方法正文、配置版本、搜索硬件/功率口径与评估契约；历史计划退出日常执行入口。逐项结果见[方法冲突处理记录](../outputs/project_cleanup_20260909/METHOD_CONFLICT_RESOLUTION.md)。

完成标准：从根 README 能找到方法实现；历史计划不会被误当成当前操作要求。**这一批不重构代码、不启动实验。**

### 第二批：清理明确冗余与根目录杂物

- [x] 已核对并执行获准缓存清单，结果见[删除回执](../outputs/project_cleanup_20260909/cache_deletion_receipt.json)；变化文件跳过，不追加缓存扫描。
- [ ] 根目录 `vivado*.log/jou`、`vitis_hls.log`、`clockInfo.txt`、`tight_setup_hold_pins.txt` 先确认是否保存唯一故障/时序信息；有价值的归档，无保留价值的列入删除清单。
- [ ] 根目录对话导出、外部调研、参考PDF归入资料档案；保留来源，不当作当前方法说明。
- [ ] `run_full_nas.py`、`retrain_best.py` 是很小的退役提示入口，`run_rl_search.py`、`visualize_results.py` 仍有转发用途：先保留，不作为优先删除对象。

完成标准：根目录的活跃入口容易识别，清理对象是已确认的缓存或冗余副本。删除前提供具体路径及影响，按项目规则取得确认。

### 第三批：处理真正占空间的构建产物

只先处理以下三处，**按单个工程推进，不扫描全部历史运行作为前置任务**：

1. [board_harness/_vivado_builds](../hls_lut_builder/board_harness/_vivado_builds)
2. [hls_lut_builder/results](../hls_lut_builder/results)
3. [board_harness/projects](../hls_lut_builder/board_harness/projects)

每个工程先保留源代码与配置、生成命令、工具版本、必要IP、综合/时序/资源报告、最终bitstream/XSA/HWH/DCP、验证回执及哈希；失败工程保留失败原因和必要复现输入。验证归档可读且引用可追溯后，再清理获准的中间产物。

`.Xil`、`.autopilot`、仿真目录和重复build可以是候选，但不能跨所有工程盲目通配删除。冻结清单不能直接改路径重写；需要保留兼容路径或独立迁移索引。

完成标准：已有结果还能复验，关键方法还能从源码重建，每次处理一个确定范围。

### 第四批：最后处理环境、旧配置和工作树

- [ ] 优先核对旧 `.venv_benchmarks/sure_2024` 与现行 `sure_2024_cuda` 的依赖和环境记录；其余6个CUDA环境在配置中各自登记，不按DLL重复判删。
- [ ] 旧配置和脚本只有在引用、历史复现及替代实现都核对后才归档。名称带旧板卡、日期或legacy不等于无价值。
- [ ] 工作树逐个核对分支、未提交变动和未合并实现；使用Git工作树管理方式处理，不能直接删除文件夹。
- [ ] `.git`、原始数据集、reference子模块和唯一训练权重不进入常规清理批次。

完成标准：环境可重建，分支工作未丢失，历史结果有明确归宿。

## 4. 已有清单只作附件

- [9月8日总审计报告](../outputs/project_cleanup_audit_20260908/REPORT.md)：查具体冲突及判断依据。
- [文档逐项清单](../outputs/project_cleanup_audit_20260908/DOCUMENTS.md)：查历史文档建议。
- [中间文件与删除候选](../outputs/project_cleanup_audit_20260908/CLEANUP_CANDIDATES.md)：查大型目录及逐路径线索。

上述容量、数量和状态是9月8日快照；9月9日已经发现新增检测代码、文档修改和旧适配器删除，执行前必须重新核对具体对象。无需为更新统计再次扫描50万个文件。

## 5. 整理完成的验收条件

- 关键方法入口和实现可定位，既有import、配置加载和源码生成路径保持可用。
- 未提交的独有实现、源数据及实验证据没有丢失。
- 冻结证据的哈希可核验；移动对象有旧路径到新路径的映射。
- 根目录和文档导航清楚；历史材料不再混作当前说明。
- 只在代码、配置或导入路径变动时运行相应回归测试；缓存和导航调整用路径、链接、差异检查验证。
- 每个执行批次记录实际保留、归档、删除的路径和实际释放空间；候选目录总大小不能当作已释放空间。
