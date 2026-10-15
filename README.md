# HW-NAS FPGA Sonar

面向声呐图像的硬件感知神经架构搜索项目。当前工作区包含 ZCU104 / MobileNetA 分类方法、独立检测分支，以及 AV7K325 等历史实现。文档核对日期：2026-09-09。

**整理整个文件夹，按 [FILE_CLEANUP_PLAN](docs/FILE_CLEANUP_PLAN.md) 执行。理解方法，先读 [method_design](docs/method_design.md)。**

| 要找的内容 | 入口 |
|---|---|
| 当前方法、算子与指标口径 | [方法说明](docs/method_design.md) |
| 方法对应哪些代码 | [实现架构与调用路径](docs/architecture.md) |
| 哪份配置属于哪个版本 | [配置版本索引](docs/SEARCH_CONFIG_CANONICAL.md) |
| 分类 RL 搜索 | [run_search.py](run_search.py)，配置由版本索引选择 |
| 固定候选评估 | [run_eval_protocol.py](run_eval_protocol.py)、[评估协议](docs/EVAL_PROTOCOL.md) |
| 检测基线训练与评估 | [run_detection_experiment.py](scripts/run_detection_experiment.py)；当前入口固定取 baseline 架构 |
| ZCU104 文档与历史协议 | [ZCU104 索引](docs/zcu104_full_experiment/README.md) |
| 全部文档与历史证据 | [文档导航](docs/INDEX.md) |

分类主路径是八阶段 MobileNetA、RL（强化学习）控制器、候选独立训练和 Pareto（非支配解）筛选。Random、Aging、Proxyless 和 SPOS 的实现仍保留，各自的历史协议不能当作本轮分类任务的默认设置。

当前分类算子表来自 ZCU104 算子级 HLS（高层次综合）与布局布线报告；表中的功率来自 Vivado 无向量估计。搜索估计、固定候选分类、完整网络硬件实现、板级运行和实测功率分别记录。配置可加载或脚本存在，均不表示这些环节已经完成。

源码位于 [src](src)、[scripts](scripts)、[hardware](hardware) 与 [hls_lut_builder](hls_lut_builder) 的源码/模板子目录；配置在 [configs](configs)，验证在 [tests](tests)。[results](results)、[artifacts](artifacts) 和 [outputs](outputs) 同时包含必要证据与中间产物，按整理清单逐项处理。

本次已将旧 README 和被改写文档的完整原文保存在[修改前备份](outputs/project_cleanup_20260909/method_docs_before.zip)，历史结果与冻结配置仍保留原路径。
