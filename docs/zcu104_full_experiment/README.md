# ZCU104 方法与版本文档索引

核对日期：2026-09-09。本目录含不同日期、候选和空间版本的协议。旧四候选计划、RAW20/CFEM20迁移与MobileNetA分类不能共同称为一个“当前主线”。

## 日常方法入口

- [当前方法说明](../method_design.md)：八阶段MobileNetA、DMDC/CFEM、原生RL和硬件代价口径。
- [实现路径](../architecture.md)：保留哪些源码以及入口如何连接。
- [配置版本索引](../SEARCH_CONFIG_CANONICAL.md)：20轮分类、200轮预算变体与640检测分支。
- [NKSID分类记录协议](NKSID_RL_CLASSIFICATION_RECORDING.md)：RL-only分类记录、候选评估和板载功率证据。
- [文件整理入口](../FILE_CLEANUP_PLAN.md)：缓存回执、关键方法保留范围和后续清理批次。

分类代价估计器已接入88行MobileNetA表；它提供算子级HLS/route聚合和Vivado功率代理。它的覆盖状态不替代完整网络布局布线、板级全样本分类或实测功率。检测训练入口目前固定取基线架构，单独记录其指标和硬件范围。

## 按版本保留的历史协议

| 文档 | 原始范围与保留原因 |
|---|---|
| [MODIFICATION_PLAN](MODIFICATION_PLAN.md)、[EXPERIMENT_PLAN](EXPERIMENT_PLAN.md) | 2026-08四阶段部署集成的修改/实验计划；不作为当前清理或实验指令 |
| [METRICS_AND_ACCEPTANCE](METRICS_AND_ACCEPTANCE.md) | 原系列指标和验收口径；历史外部仪表协议与后续板载输入功率协议分开 |
| [SERIAL_EXECUTION_PROMPTS](SERIAL_EXECUTION_PROMPTS.md)、[PARALLEL_WORK_PROMPTS](PARALLEL_WORK_PROMPTS.md) | 历史任务编排模板，不代表现在必须建工作树、并行任务或启动实验 |
| [SEARCH_TRAIN_EVAL_MIGRATION](SEARCH_TRAIN_EVAL_MIGRATION.md) | 已退役RAW20 / 9行CFEM20迁移说明 |
| [ZCU104_OPERATOR_COST_LUT_CONTRACT_V1](ZCU104_OPERATOR_COST_LUT_CONTRACT_V1.md) | 旧fourstage_cfem20_v1算子合同，非当前88行MobileNetA合同 |
| [SONAR_AND_THERMAL_METRICS](SONAR_AND_THERMAL_METRICS.md) | 声呐退化、图像质量和温度的专项定义；不等同于分类、检测搜索状态 |

原README的四个部署候选、60份checkpoint待登记、服务器/分支状态和执行顺序已保存至[修改前备份](../../outputs/project_cleanup_20260909/method_docs_before.zip)。对应原始产物仍保留原路径；本文不修改任何冻结结果、门禁或源代码快照。
