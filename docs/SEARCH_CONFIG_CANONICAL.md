# 搜索配置版本索引

核对日期：2026-09-11。根据 YAML 内容与当前运行时区分配置；文件名、默认参数和旧实验 ID 不决定当前方法。历史运行内的冻结配置保持原样。

## ZCU104 分类

| 配置 | 空间 / 数据 | RL 预算与搜索约束 | 定位 |
|---|---|---|---|
| [zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml](../configs/search/zcu104_fourstage_cfem20_v2_rl_entropy_pareto.yaml) | 八阶段 `mobilenetA_full_dmdc_cfem_v1`；NKSID，1×224×224，8类 | episodes=20，eval_epochs=10；latency≤25ms，BRAM聚合预算218 | 保留旧文件名的20轮分类配置；内容已不是四阶段RAW20 |
| [zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml](../configs/search/zcu104_mobilenetA_v2_rl_entropy_pareto_feasible_budget.yaml) | 同一八阶段分类空间 | episodes=200，eval_epochs=10；latency≤120ms，BRAM≤312 tile，BRAM18压力≤1；额外物理风险筛选 | 独立预算变体；BRAM已按物理容量硬限制 |

两份配置均为 RL-only，选择 macro_f1，使用 `zcu104_mobilenetA_full_operator_cost_lut_v1` 估计器。当前200轮配置指向 `mobilenetA_full_dmdc_cfem_v3_gap1280` 表；20轮配置当前仍指向旧v2表，会被当前估计器拒绝，不能当作可直接运行的新主配置。新表包含 GAP1280，88 行完整性及哈希预检已通过，原始 GAP HLS/route 报告亦已复核；旧 v2_recovered 表因使用 GAP320 被拒绝。搜索准入仍需完整运行时门检查，表预检通过不等于已启动新实验或完整网络可部署。功率上限15W、能耗上限375mJ是搜索代理约束，不是已经实测的指标。两份预算不同，不能拼成一个公平比较或同一冻结协议。

`search.proxy_training` 已显式记录轻量代理配方；`training` 保留完整重训练配方。AMP 与梯度累积等设置不得跨层推断。当前控制器保持 MLP。实际评估数、合法空间、终止原因、候选身份和解析后的代理设置仍需从运行产物读取。

运行入口是 [run_search.py](../run_search.py)。完整空间、Skip合法性、算子与聚合公式统一见[方法说明](method_design.md)，分类记录字段见[记录协议](zcu104_full_experiment/NKSID_RL_CLASSIFICATION_RECORDING.md)。

## ZCU104 检测

[zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml](../configs/search/zcu104_mobilenetA_v2_detection_640_multiscale_fixed_neck_head.yaml) 使用单通道640输入、三类框标注、多尺度特征和固定neck/head。空间版本为 `mobilenetA_full_dmdc_cfem_v1_detection640_multiscale`，与分类的头和算子形状不同。

实际入口 [run_detection_experiment.py](../scripts/run_detection_experiment.py) 当前固定使用 `baseline_architecture()`。YAML中的RL和200 episodes尚不对应这个入口中的搜索循环。检测表的固定neck/head覆盖不代表完整分类表可直接用于640输入。

## 历史配置保留范围

| 系列 | 保留入口 | 边界 |
|---|---|---|
| AV7K325 semantic-safe MobileNetV2 | [nksid_fpga_search_mobile_anchor_av7k325.yaml](../configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml) | 原板卡和算子策略下的历史/通用入口，不是ZCU104默认配置 |
| AV7K325 RL / Aging比较 | [MULTIOBJECTIVE_AGING_EVOLUTION](MULTIOBJECTIVE_AGING_EVOLUTION.md) | 保留独立公平比较协议；不要求RL-only任务追加其他方法 |
| 旧四阶段CFEM20 / RAW20 | [迁移说明](zcu104_full_experiment/SEARCH_TRAIN_EVAL_MIGRATION.md)、[旧合同](zcu104_full_experiment/ZCU104_OPERATOR_COST_LUT_CONTRACT_V1.md) | 9行适配器已退役，历史冻结文件保留复验；不能替代88行MobileNetA表 |
| 其他早期配置 | [configs/search/legacy](../configs/search/legacy) | 保留引用与复现依赖；目录名不构成删除依据 |

旧配置索引中的AV7K325宏骨干溯源表完整保存在[修改前备份](../outputs/project_cleanup_20260909/method_docs_before.zip)中的同路径文件。历史源结果仍在原目录，未修改其哈希或实验身份。
