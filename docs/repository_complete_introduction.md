# HW-NAS FPGA Sonar 仓库完整介绍

本文整理自当前已归档的结构发现与静态审计文档，目标是给新人、后续审计执行者和实验维护者提供一份可以直接使用的仓库总览。本文不是新增代码审计，不替代各 `audit_*.md` 的路径行号证据。

## 1. 信息来源

当前参与整理的归档文件如下：

| 归档文件 | 主题 | 在本介绍中的用途 |
| --- | --- | --- |
| `docs/FIRST_PRINCIPLES_AUDIT_20260703.md` | 第一性原理重审 | 数据协议、算子语义、证据强度与板卡资源纠偏 |
| `docs/audit_structure_discovery.md` | 仓库结构发现 | 真实功能分层、入口类别、目录职责、异常边界 |
| `docs/audit_entry_config.md` | 入口与配置闭环审计 | 主入口、配置流向、输出目录、resume 语义、legacy 入口 |
| `docs/audit_candidate_hardware.md` | 候选表示与硬件代价审计 | ArchitectureSpec 到模型、硬件估计、CandidateMetrics 的映射 |
| `docs/audit_search_training_metrics.md` | 搜索训练指标链路审计 | random/RL/proxyless 搜索、训练评估、macro_f1/top1/feasibility 传播 |
| `docs/audit_deploy_result_consumption.md` | 部署与结果消费审计 | checkpoint、ONNX/INT8/HLS stub、推理和 reporting schema |

当前没有发现独立的 `docs/audit_hls_lut*.md` 或 `docs/audit_hls_lut_production*.md`。因此本文中关于 HLS/LUT 生产链的描述来自结构发现、入口配置审计、候选硬件审计和部署消费审计中已经覆盖的部分；operator manifest、模板、Vitis/Vivado/board harness 的完整生产链仍应以独立审计补齐。

解释优先级：2026-07-03 第一性原理重审晚于下列静态归档。若旧文档把
fold-0 指标解释为最终泛化结果，或把简化 HLS `denoise`/`edge` 解释为
训练期 PyTorch 算子的部署证据，以第一性原理重审的边界为准。

## 2. 仓库一句话定位

这是一个面向水下声呐图像分类/识别任务的 HW-NAS 仓库。它的核心目标是在 FPGA 资源约束下，完成从搜索空间定义、候选架构采样、训练评估、硬件代价估计、候选选择、重训练、导出、推理到结果报告的闭环。

项目的核心指标不是单一 accuracy，而是：

- 分类指标：`macro_f1`、`top1`、`top5`、`weighted_f1`
- 硬件指标：`latency_ms`、`LUT`、`DSP`、`BRAM`、`power_w`、`energy_mj`
- 流程指标：`feasibility`、可复现配置、候选与结果 artifact schema

## 3. 真实功能分层

结构发现的结论是：这个仓库不是简单的目录树，也不是标准三层架构。它更像“两条生产链 + 一个共享 NAS 内核 + 一圈结果消费层”。

```text
设计/依据层
  docs/
  reference/
  README.md

静态输入层
  data/NKSID/
  configs/
  hls_lut_builder/configs/

主 NAS 链
  configs/data
    -> run_*.py + src/hwnas_fpga/runtime.py
      -> src/hwnas_fpga/data
      -> src/hwnas_fpga/search_space
      -> src/hwnas_fpga/models
      -> src/hwnas_fpga/hardware
      -> src/hwnas_fpga/search + src/hwnas_fpga/training
        -> src/hwnas_fpga/experiment.py
          -> results/ + checkpoints/ + artifacts/

HLS/LUT 链
  hls_lut_builder/
    -> HLS/Vivado/board measurement outputs
      -> hls_lut_builder/results/formal_lut*.json
        -> configs/search/*.yaml hardware.lut_path
          -> src/hwnas_fpga/hardware cost estimator

结果消费层
  src/hwnas_fpga/deploy/
  scripts/
  run_export.py
  run_infer.py
  visualize_results.py
  results_archive/
  outputs/
```

主 NAS 链基本是单向流水线：`configs/data -> runtime/entrypoint -> src core -> results/checkpoints -> retrain/export/reporting`。

HLS/LUT 链不是纯下游工具链。结构发现指出它存在功能性双向依赖：`hls_lut_builder` 生成 formal LUT/status，供 NAS runtime 和 hardware estimator 消费；同时 HLS/LUT 脚本又会复用 `src/hwnas_fpga/hardware` 的 report parser 和 LUT 数据结构。

## 4. 顶层入口

入口可以分为当前主链路、前置/旁路入口、后续入口、HLS/LUT 入口和兼容/退休入口。

| 入口文件 | 类型 | 当前定位 |
| --- | --- | --- |
| `run_search.py` | 当前主入口 | random、RL、Proxyless 搜索的统一入口 |
| `run_search_space_probe.py` | 前置探测入口 | 不训练，只做搜索空间硬件可行性探测 |
| `run_backbone_baseline.py` | 旁路实验入口 | backbone screening 与 anchor export |
| `run_operator_ablation.py` | 旁路实验入口 | operator ablation，当前会复用 `run_backbone_baseline.py` 中的工具函数 |
| `run_retrain.py` | 后续入口 | 使用搜索得到的 best candidate 做最终重训练 |
| `run_export.py` | 后续入口 | 从 checkpoint 导出 ONNX、INT8 package 和 HLS project stub |
| `run_infer.py` | 后续入口 | 从 checkpoint 恢复模型并对图片或目录推理 |
| `run_build_lut.py` | HLS/LUT 入口 | 从 HLS profiling/report 构建 LUT |
| `run_rl_search.py` | 兼容入口 | 转发到 `run_search.py --search-method rl` |
| `visualize_results.py` | 兼容入口 | 转发到 `scripts/visualize_results.py` |
| `run_full_nas.py` | 退休入口 | 明确 retired，不是活跃执行链路 |
| `retrain_best.py` | 退休入口 | 明确 retired，不是活跃执行链路 |

对于新用户，推荐按下面的顺序理解入口：

```text
run_search.py
  -> run_retrain.py
    -> run_export.py
    -> run_infer.py
  -> scripts/generate_paper_search_table.py
  -> scripts/plot_hw_nas_publication_figures.py
```

`run_search_space_probe.py` 是搜索前的 feasibility 探测，不应被解释为训练搜索结果。`run_full_nas.py` 和 `retrain_best.py` 是兼容/退休入口，不应作为新流程起点。

## 5. 核心源码职责

`src/hwnas_fpga` 是共享 NAS 内核层。各子模块的职责如下：

| 模块 | 主要职责 | 关键输出或消费对象 |
| --- | --- | --- |
| `interfaces.py` | 定义搜索约束、候选、指标等共享数据结构 | `SearchConstraints`、`SearchCandidate`、`CandidateMetrics` |
| `runtime.py` | 组装配置、搜索空间、硬件估计器、数据管线等运行时对象 | config -> runtime objects |
| `experiment.py` | 统一创建 run 目录并落盘配置、候选、summary、checkpoint | `config.yaml`、`cli_args.json`、`run_info.json`、`candidates.*`、`best_model.pt` |
| `data/` | NKSID 数据加载、transform、split、k-fold、class weights | train/val/test loaders |
| `search_space/` | 搜索空间配置、采样、baseline architecture、resolve_blocks、pre-prune、probe | `ArchitectureSpec`、`BlockSpec`、`ResolvedBlockSpec` |
| `models/` | PyTorch 模型构建、backbone、Proxyless supernet | searchable model、supernet |
| `hardware/` | FPGA cost estimator、board profile、LUT query、report parser | `CostEstimate`、LUT hit/miss、feasibility |
| `search/` | random、RL、Proxyless 搜索器，Pareto 与约束工具 | evaluated candidates、best candidate |
| `training/` | 训练、评估、重训练工具 | metrics、history、checkpoint payload |
| `deploy/` | ONNX 导出、推理、INT8 weight package、HLS stub、report ingestion | deployment artifacts |

需要注意两个跨边界现象：

- `models/proxyless.py` 不只是模型定义，也计算 Proxyless supernet 的 expected hardware metrics。
- 根目录实验脚本不是纯薄包装，`run_backbone_baseline.py` 和 `run_operator_ablation.py` 含实验协议逻辑，且后者复用前者的工具函数。

## 6. 候选表示与硬件代价链路

候选表示的主流转关系如下：

```text
config/search_space
  -> SearchSpaceConfig
    -> ArchitectureSpec / StageSpec / BlockSpec
      -> models.builder.build_model()
      -> SearchSpace.resolve_blocks()
        -> ResolvedBlockSpec
          -> FPGACostEstimator.estimate()
            -> CostEstimate
              -> CandidateMetrics
                -> SearchCandidate
                  -> ExperimentTracker artifacts
```

语义上，候选架构必须同时能被三个系统解释：

- `search_space`：能采样、反序列化、validate、resolve block。
- `models`：能把每个 block 构造成 PyTorch 模型。
- `hardware`：能把同一个 block 映射到 analytical cost 或 LUT query key。

已经归档的候选/硬件审计给出的重要边界包括：

- `conv`、`dw_pw_conv`、`mbconv`、`fused_mbconv`、`skip`、`denoise`、`edge` 大体有模型和代价实现对应关系。
- `mixconv` 的 `kernel_size` 在 search/LUT key 中有区分，但模型和分析代价固定使用 `(3,5,7)`，存在语义不一致。
- `FPGACostEstimator` 与 `BackboneCostEstimator` 是两套并行估计体系，NAS candidate 与 backbone baseline 的 DSP/BRAM/LUT 口径不能直接横向比较。
- LUT entry 中存在 `power_w/energy_mj`，但 estimator 命中 LUT 时主要消费 latency/DSP/BRAM/LUT；因此 `CandidateMetrics.power_w/energy_mj` 不等同于 formal LUT 实测功耗/能耗。
- `CandidateMetrics` 同时承载分类指标和硬件指标，便于 artifact 落盘，但也要求下游明确区分 `macro_f1/top1/accuracy/selection_score`。

## 7. 搜索、训练与指标链路

搜索训练链路可以概括为：

```text
config
  -> runtime creates data/search_space/estimator/searcher
    -> searcher samples candidate
      -> hardware feasibility check
      -> train/evaluate or proxyless supernet evaluation
        -> CandidateMetrics
          -> ExperimentTracker
            -> candidates.jsonl / candidates.json / candidates.csv
            -> best_candidate.json / best_model.pt
            -> summary.json / run_info.json
```

三类搜索器的行为差异：

| 搜索器 | 候选来源 | 训练/评估方式 | 记录特点 |
| --- | --- | --- | --- |
| random | search space sampling | 调用 `train_model()` | 可行候选进入训练和候选记录 |
| RL | controller 采样 | 调用 `train_model()`，含 reward/baseline/controller 状态 | resume 依赖 controller 与 search state artifacts |
| proxyless | supernet warmup/search 后提取 argmax 架构 | 不走普通 `train_model()`，内部评估 argmax summary | expected hardware metrics 与最终 estimator 口径不同 |

当前归档中最关键的指标结论是：

- `macro_f1` 是当前应优先跟踪的搜索与论文指标。
- `top1` 和 `accuracy` 不应混用；审计发现某些路径会把 selection score 写进 `accuracy` 字段。
- `feasible/infeasible` 的处理会影响候选是否进入 best、Pareto 和 summary。
- `retrain_architecture()` 的输出 schema 与搜索评估 schema 不完全一致，尤其是 `macro_f1/top5/best_eval` 闭环不足。
- `search_space_probe` 明确是可行性抽样，不是训练搜索结果；artifact 需要在命名或 schema 上进一步显式区分。

## 8. 配置与运行产物

主配置流向是：

```text
config YAML + CLI args
  -> runtime.pick(cli, config, default)
  -> run_search.py resolves dataset/search/hardware/project
  -> ExperimentTracker
    -> config.yaml
    -> cli_args.json
    -> run_info.json
    -> logs/
    -> results/
    -> checkpoints/
```

典型 run 目录结构：

```text
results/<run_name>/
  config.yaml
  cli_args.json
  run_info.json
  logs/
  results/
    baseline.json
    dataset_summary.json
    search_space_summary.json
    candidates.jsonl
    candidates.json
    candidates.csv
    pareto_front.json
    pareto_selection.json
    summary.json
    best_candidate.json
    candidates/
      <arch_id>.json
  checkpoints/
    search_state.json
    best_candidate.json
    best_model.pt
    final_best_model.pt
    controller_latest.pt
    controller_best.pt
```

配置层已确认的关键点：

- `project.seed`、`output_dir`、`run_name`、`dataset.*`、`search_space.*`、`constraints.*`、`hardware.lut_path` 都进入运行链路。
- `hardware.operator_manifest_path` 在主配置中可能未显式声明，但 runtime 默认读取 `hls_lut_builder/configs/operator_manifest.yaml`，这是配置层到 HLS/LUT 层的隐式依赖。
- `--resume` 语义不是全局恢复语义，而是主要服务 RL 流程；它依赖 `candidates.jsonl`、`search_state.json`、`controller_latest.pt` 等 artifacts。

## 9. 部署、推理与结果消费

部署链路是：

```text
checkpoint
  -> run_export.py
    -> model.onnx + model.json
    -> optional quantized_weights_int8.pt + quantized_weights_int8.json
    -> optional hls_project/ stub

checkpoint
  -> run_infer.py
    -> load model
    -> recover image_size/input_channels/class_names
    -> prediction JSON
```

需要明确区分：

- `best_model.pt`：搜索阶段 best checkpoint，schema 以 `candidate + model_state_dict` 为核心。
- `final_best_model.pt`：重训练阶段 checkpoint，schema 以 `architecture + metrics + model_state_dict` 为核心。
- `model.json`：ONNX sidecar metadata，保存 input shape、class names、architecture、opset、model info。
- `quantized_weights_int8.pt`：完整 INT8 package，包含 architecture、candidate、class names、weights、scales。
- `quantized_weights_int8.json`：当前只是量化摘要，不是完整部署 manifest。
- `hls_project/`：HLS project stub，不是完整 HLS/Vivado/formal LUT 生产链。

结果消费层分成三类：

| 消费脚本 | 当前口径 |
| --- | --- |
| `scripts/visualize_results.py` | 旧口径，仍大量使用 `accuracy` 作为主排序和图表字段 |
| `scripts/generate_paper_search_table.py` | 论文表格口径，使用 `macro_f1/top1/top5/latency/DSP/BRAM/LUT/power` |
| `scripts/plot_hw_nas_publication_figures.py` | 论文图口径，要求 `macro_f1/top1/latency_ms/lut/dsp/bram/power_w` |
| `scripts/generate_backbone_baseline_table.py` | backbone baseline 口径，使用 `macro_f1/top1/fpga_latency/peak resources/power/feasible` |
| `scripts/audit_run_storage.py` | inventory only，不是删除策略 |

部署审计中最重要的两个可复现性问题：

- search `best_model.pt` 的 `candidate.metrics` 可能不会被 `export_checkpoint_to_onnx` 写入 `checkpoint_metrics`，导致 ONNX metadata 缺失 `macro_f1/top1/latency/LUT/DSP/BRAM/power`。
- `run_infer.py` 的输入尺寸恢复依赖 checkpoint 所在 run 目录，如果 checkpoint 被单独复制到部署目录，可能退回默认 `image_size=224` 和 `input_channels=1`。

## 10. HLS/LUT 链路的当前理解

HLS/LUT 链路在当前归档中的定位是“独立生产链 + NAS runtime 消费者”：

```text
hls_lut_builder/configs
  -> HLS case generation
  -> Vitis HLS / Vivado / board harness
  -> formal_lut*.json + formal_lut_status*.json
  -> configs/search/*.yaml hardware.lut_path / formal_lut_status_path
  -> src/hwnas_fpga/hardware lookup_table + cost estimator
```

目前可以确认：

- `run_build_lut.py` 是顶层 HLS/LUT 前置生产入口之一。
- `hls_lut_builder/results/formal_lut*.json` 会被 search config 的 `hardware.lut_path` 消费。
- runtime 会默认读取 `hls_lut_builder/configs/operator_manifest.yaml`。
- `run_export.py --prepare-hls` 生成的是 HLS stub，不是 `hls_lut_builder` 的 formal LUT 生产链。

仍待独立审计确认的部分：

- operator manifest 与 NAS op 的完整映射。
- packed stream contract 是否被每个模板满足。
- HLS estimate、Vivado downstream、board harness 三种指标来源的单位和字段一致性。
- formal LUT status 中 `status/defer_reason/board_cycles/board_latency_ms` 的消费闭环。

## 11. 已知边界和高优先级风险

这些不是泛泛重构建议，而是归档审计已经定位出的边界或风险点。

| 优先级 | 问题 | 影响面 |
| --- | --- | --- |
| P1 | `CandidateMetrics.accuracy` 在部分路径可能保存 selection score，不稳定等于 top1 | 搜索排序、结果消费、论文图表 |
| P1 | Proxyless expected hardware metrics 与 `FPGACostEstimator.estimate()` 口径不同 | Proxyless 搜索期硬件惩罚与最终候选落盘指标 |
| P1 | `mixconv.kernel_size` 在 search/LUT key 与模型/分析代价中语义不一致 | 同构候选可能查不同 LUT 条目 |
| P1 | NKSID 当前 image-index fold 有高 filename-adjacency 泄漏风险，且无 acquisition group 元数据 | 泛化结论与显著性 |
| P1 | PyTorch 与 HLS `denoise`/`edge` 算法不一致 | 算子部署声明与硬件代价 |
| P1 | ONNX metadata 对 search checkpoint 可能丢失 best metrics | 部署 artifact 可复现性 |
| P1 | 旧 visualization 仍以 accuracy 为主 | 与 macro_f1/top1 优先口径冲突 |
| P2 | `num_classes/head` 存在配置、数据加载、模型构建、cost head 多来源 | 模型 head 与 cost head 一致性 |
| P2 | NAS estimator 与 backbone estimator 的 peak/total 资源口径不同 | backbone 与 NAS 候选横向比较 |
| P2 | LUT 中 power/energy 字段未被 estimator 作为 formal 实测功耗闭环消费 | power/energy 解释 |
| P2 | `run_operator_ablation.py` 复用 `run_backbone_baseline.py` 的顶层脚本函数 | 入口层边界 |
| P2 | `hardware.operator_manifest_path` 隐式默认到 HLS/LUT 配置 | 配置可复现性 |
| P2 | `retrain_architecture()` 输出的评估 schema 与搜索评估不完全一致 | 搜索-重训指标闭环 |
| P2 | INT8 sidecar JSON 不是完整 manifest | 外部部署或 HLS 消费者 |
| P2 | HLS stub 容易被误读为完整部署链 | 部署边界认知 |

## 12. 新人最小可用学习路径

如果目标是跑通最核心的 NAS 流程，建议按下面顺序熟悉文件：

1. `README.md`
2. `docs/REPO_LAYOUT.md`
3. `docs/audit_structure_discovery.md`
4. `configs/search/*.yaml` 中当前 formal/mainline 配置
5. `run_search.py`
6. `src/hwnas_fpga/runtime.py`
7. `src/hwnas_fpga/experiment.py`
8. `src/hwnas_fpga/interfaces.py`
9. `src/hwnas_fpga/search_space/space.py`
10. `src/hwnas_fpga/models/builder.py`
11. `src/hwnas_fpga/hardware/cost.py`
12. `src/hwnas_fpga/training/trainer.py`
13. `src/hwnas_fpga/search/searcher.py`
14. 需要 RL 时再读 `src/hwnas_fpga/search/rl_searcher.py`
15. 需要 Proxyless 时再读 `src/hwnas_fpga/search/proxyless_searcher.py`
16. 搜索后读 `run_retrain.py`
17. 部署时读 `run_export.py`、`run_infer.py`、`src/hwnas_fpga/deploy/*`
18. 论文表格和图读 `scripts/generate_paper_search_table.py`、`scripts/plot_hw_nas_publication_figures.py`

可以暂缓阅读或只按需阅读：

- `reference/`：设计来源和参考实现，不进入主运行链。
- `results_archive/`、`results-launch/`：历史结果和归档结果。
- `outputs/presentations/solar-system`：非 HW-NAS 运行链产物。
- `run_full_nas.py`、`retrain_best.py`：退休兼容入口。
- `scripts/legacy/`、`docs/legacy/`、`configs/search/legacy/`：历史保留区域。

## 13. 推荐的实际执行顺序

面向一次完整实验，推荐按以下顺序执行：

```text
1. 搜索空间探测，可选
   python run_search_space_probe.py --config <search_config> --num-samples <N>

2. 主搜索
   python run_search.py --config <search_config>

3. 重训练
   python run_retrain.py --run-dir results/<search_run_name>

4. 导出
   python run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt

5. 推理
   python run_infer.py --checkpoint results/<run_name>/checkpoints/best_model.pt --input <image_or_dir>

6. 结果表格和论文图
   python scripts/generate_paper_search_table.py ...
   python scripts/plot_hw_nas_publication_figures.py ...
```

如果实验依赖 formal LUT，需要先确认：

- `hardware.lut_path` 指向的 `formal_lut*.json` 存在。
- `hardware.formal_lut_status_path` 指向的 status 文件存在。
- `hardware.operator_manifest_path` 是显式配置，或明确接受 runtime 默认路径。
- 当前 operator vocabulary 与 search config 中的 op choices 一致。

## 14. 读这个仓库时最容易误解的地方

1. `accuracy` 不一定是 top1。当前审计显示它在某些路径可能承载 selection score。
2. `run_search_space_probe.py` 不训练，不代表搜索结果。
3. `run_export.py --prepare-hls` 只生成 HLS stub，不是完整 HLS/Vivado 流程。
4. `hls_lut_builder` 不是简单外部工具目录，它和 `src/hwnas_fpga/hardware` 有双向功能依赖。
5. `models/` 不只是模型定义，`models/proxyless.py` 还承担硬件感知搜索相关计算。
6. `scripts/visualize_results.py` 是旧口径结果消费脚本，和 paper table/figure 脚本的指标优先级不一致。
7. `results/` 不能随意删除；`scripts/audit_run_storage.py` 是 inventory，不是删除工具。
8. `reference/` 是设计参考，不进入 active runtime，但保留用于 provenance。

## 15. 当前文档体系中的位置

本文应当作为审计归档后的“总览入口”。建议阅读顺序：

```text
docs/repository_complete_introduction.md
  -> docs/audit_structure_discovery.md
  -> docs/audit_entry_config.md
  -> docs/audit_candidate_hardware.md
  -> docs/audit_search_training_metrics.md
  -> docs/audit_deploy_result_consumption.md
```

Dynamic Phase0 evidence is maintained separately:

- `docs/PHASE0_V3_BOARD_RESULTS.md`
- `docs/PHASE0_V3_RETRAINED_BOARD_REINJECTION.md`
- `docs/PHASE0_V4_SONAR_RESULTS.md`

Current measurement and research-gate archives are maintained separately:

- `docs/EVAL_PROTOCOL.md`
- `docs/MEASUREMENT_FIRST_REBUILD.md`
- `docs/PROXY_RELIABILITY_AUDIT.md`
- `docs/PROXY_RELIABILITY_AUDIT_V2.md`
- `docs/EXTERNAL_SONAR_DATASETS.md`

Proxy Reliability Gate 0 v2 reduces formal execution to 1,200 reusable
150-epoch trajectories, but the current formal count is still `0/1200`.
The completed CPU one-epoch benchmarks are explicitly ineligible as formal
evidence, so Gate 0 remains `not_ready`. External detection/mixed-task datasets
also remain separate from NKSID classification and board evidence.

后续如果补做 HLS/LUT 生产链审计，应新增：

```text
docs/audit_hls_lut_production.md
```

并回填本文第 10 节中的待确认项。
