# 仓库结构发现归档

归档时间：2026-05-17

说明：本文件归档的是基于实际抽样读取后的结构发现，不是代码审计。

## 第一部分：客观事实

### 顶层目录与重要文件

顶层一级目录：

- `.claude`
- `.comate`
- `.deepeval`
- `.git`
- `.idea`
- `.pytest_cache`
- `.Xil`
- `artifacts`
- `configs`
- `data`
- `docs`
- `hls_lut_builder`
- `outputs`
- `reference`
- `results`
- `results-launch`
- `results_archive`
- `scripts`
- `src`
- `tests`
- `__pycache__`

重要顶层文件：

- `E:\1\hwnas\hwnas\README.md`
- `E:\1\hwnas\hwnas\run_search.py`
- `E:\1\hwnas\hwnas\run_retrain.py`
- `E:\1\hwnas\hwnas\run_export.py`
- `E:\1\hwnas\hwnas\run_infer.py`
- `E:\1\hwnas\hwnas\run_build_lut.py`
- `E:\1\hwnas\hwnas\run_backbone_baseline.py`
- `E:\1\hwnas\hwnas\run_operator_ablation.py`
- `E:\1\hwnas\hwnas\run_search_space_probe.py`
- `E:\1\hwnas\hwnas\generate_estimator_lut.py`
- `E:\1\hwnas\hwnas\test_full_pipeline.py`

### 文件数量级

- `src`：73，几十个。
- `configs`：58，几十个。
- `tests`：58，几十个。
- `scripts`：26，几十个。
- `docs`：20，几十个。
- `artifacts`：110，上百个。
- `outputs`：185，上百个。
- `results_archive`：389，上百个。
- `data`：2620，千级。
- `results`：5600，千级。
- `reference`：48824，万级。
- `hls_lut_builder`：87636，万级，主要因为参考库和生成的 HLS/Vivado 工程产物。
- `git ls-files` 中 `hls_lut_builder` 只有 61 个跟踪文件，说明该目录混有大量生成物。

### 入口点

当前主入口是：

- `E:\1\hwnas\hwnas\run_search.py`

前置/旁路入口包括：

- `E:\1\hwnas\hwnas\run_backbone_baseline.py`
- `E:\1\hwnas\hwnas\run_search_space_probe.py`
- `E:\1\hwnas\hwnas\run_operator_ablation.py`

后续入口包括：

- `E:\1\hwnas\hwnas\run_retrain.py`
- `E:\1\hwnas\hwnas\run_export.py`
- `E:\1\hwnas\hwnas\run_infer.py`

HLS/LUT 入口主要包括：

- `E:\1\hwnas\hwnas\hls_lut_builder\scripts\gen_project.py`
- `E:\1\hwnas\hwnas\hls_lut_builder\scripts\run_synthesis.py`
- `E:\1\hwnas\hwnas\hls_lut_builder\scripts\parse_reports.py`
- `E:\1\hwnas\hwnas\hls_lut_builder\scripts\run_pilot_pipeline.py`
- `E:\1\hwnas\hwnas\run_build_lut.py`

兼容/退休入口：

- `E:\1\hwnas\hwnas\run_full_nas.py`：退休兼容入口。
- `E:\1\hwnas\hwnas\retrain_best.py`：退休兼容入口。
- `E:\1\hwnas\hwnas\run_rl_search.py`：转发到 `run_search.py --search-method rl` 的兼容入口。

## 第二部分：层次判断

识别出的真实结构更像“两条生产链 + 一个共享 NAS 内核 + 一圈结果消费层”：

```text
E:\1\hwnas\hwnas\reference, docs
  -> 提供设计依据和说明，不直接进入主运行链

E:\1\hwnas\hwnas\data + configs
  -> run_*.py / src\hwnas_fpga\runtime.py
     -> src\hwnas_fpga\data
     -> src\hwnas_fpga\search_space
     -> src\hwnas_fpga\models
     -> src\hwnas_fpga\hardware
     -> src\hwnas_fpga\search + training
        -> src\hwnas_fpga\experiment.py
           -> results / artifacts / checkpoints

E:\1\hwnas\hwnas\hls_lut_builder
  -> HLS/Vivado/board measurement outputs
     -> hls_lut_builder\results\formal_lut*.json
        -> configs\search\*.yaml hardware.lut_path
           -> src\hwnas_fpga\hardware cost estimator
```

### 层成员清单

设计/依据层：

- `E:\1\hwnas\hwnas\docs`
- `E:\1\hwnas\hwnas\reference`
- `E:\1\hwnas\hwnas\README.md`

静态输入层：

- `E:\1\hwnas\hwnas\data\NKSID`
- `E:\1\hwnas\hwnas\configs`
- `E:\1\hwnas\hwnas\hls_lut_builder\configs`

HLS/LUT 生产层：

- `E:\1\hwnas\hwnas\hls_lut_builder\templates`
- `E:\1\hwnas\hwnas\hls_lut_builder\include`
- `E:\1\hwnas\hwnas\hls_lut_builder\scripts`
- `E:\1\hwnas\hwnas\hls_lut_builder\board_harness`
- `E:\1\hwnas\hwnas\hls_lut_builder\results`
- `E:\1\hwnas\hwnas\run_build_lut.py`

共享 NAS 内核层：

- `E:\1\hwnas\hwnas\src\hwnas_fpga\interfaces.py`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\data`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\models`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\hardware`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\training`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\search`

应用编排层：

- `E:\1\hwnas\hwnas\run_search.py`
- `E:\1\hwnas\hwnas\run_retrain.py`
- `E:\1\hwnas\hwnas\run_export.py`
- `E:\1\hwnas\hwnas\run_infer.py`
- `E:\1\hwnas\hwnas\run_backbone_baseline.py`
- `E:\1\hwnas\hwnas\run_operator_ablation.py`
- `E:\1\hwnas\hwnas\run_search_space_probe.py`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py`
- `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py`

结果消费/报告层：

- `E:\1\hwnas\hwnas\scripts`
- `E:\1\hwnas\hwnas\results`
- `E:\1\hwnas\hwnas\artifacts`
- `E:\1\hwnas\hwnas\results_archive`
- `E:\1\hwnas\hwnas\results-launch`
- `E:\1\hwnas\hwnas\outputs`

### 层关系

层关系不是纯单向流水线。

主 NAS 链基本是：

```text
configs/data
  -> run_*.py/runtime
     -> src core
        -> results/checkpoints
           -> retrain/export/reporting
```

HLS/LUT 链存在功能性双向依赖：

- `hls_lut_builder\scripts\parse_reports.py`、`generate_formal_lut.py` 会导入 `src\hwnas_fpga\hardware` 的解析器/LUT 结构。
- `src\hwnas_fpga\runtime.py` 默认读取 `hls_lut_builder\configs\operator_manifest.yaml`。
- 搜索配置消费 `hls_lut_builder\results\formal_lut*.json`。

## 第三部分：异常和模糊点

- 现象：`hls_lut_builder` 既是底层 HLS 生产者，又复用上层 Python 包的硬件解析/LUT 数据结构。
  文件路径：`E:\1\hwnas\hwnas\hls_lut_builder\scripts\parse_reports.py`、`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py`。
  影响：这是功能循环依赖，不是简单下游工具目录。

- 现象：模型目录含硬件感知搜索实现。
  文件路径：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\proxyless.py`。
  影响：`models` 不只是模型定义，也承担 Proxyless 超网和硬件指标期望计算。

- 现象：根目录入口脚本不是薄包装，包含大量实验协议逻辑。
  文件路径：`E:\1\hwnas\hwnas\run_backbone_baseline.py`、`E:\1\hwnas\hwnas\run_operator_ablation.py`。
  影响：应用层逻辑分散在根目录，`run_operator_ablation.py` 还直接导入 `run_backbone_baseline.py`。

- 现象：报告解析职责重复出现。
  文件路径：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\report_parser.py`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\report_parser.py`、`E:\1\hwnas\hwnas\hls_lut_builder\scripts\parse_reports.py`。
  影响：需要后续单独确认谁是权威解析入口。

- 现象：存在看起来不属于 HW-NAS 运行链的输出。
  文件路径：`E:\1\hwnas\hwnas\outputs\presentations\solar-system`。
  影响：应视为孤立产物或其他任务残留，不能纳入 NAS 分层。

- 现象：入口脚本数量多，且有退休/兼容入口。
  文件路径：`E:\1\hwnas\hwnas\run_full_nas.py`、`E:\1\hwnas\hwnas\retrain_best.py`、`E:\1\hwnas\hwnas\run_rl_search.py`、`E:\1\hwnas\hwnas\docs\REPO_LAYOUT.md`。
  影响：入口总体可辨认，但首次接触时容易误选旧入口。

## 第四部分：后续审计建议

### 任务名：入口与配置闭环审计

- 范围：`E:\1\hwnas\hwnas\run_*.py`、`retrain_best.py`、`visualize_results.py`、`src\hwnas_fpga\runtime.py`、`src\hwnas_fpga\experiment.py`、`configs`。
- 重点问题：CLI/config 优先级、seed、输出目录、当前配置与 legacy 配置边界。
- 预估文件数：约 70。
- 依赖关系：必须先做。

### 任务名：候选表示与硬件代价审计

- 范围：`src\hwnas_fpga\interfaces.py`、`search_space`、`models\builder.py`、`models\proxyless.py`、`hardware`。
- 重点问题：`ArchitectureSpec` 到模型/代价的映射，latency/LUT/DSP/BRAM/power 的一致性。
- 预估文件数：约 18。
- 依赖关系：依赖入口与配置审计。

### 任务名：搜索训练指标链路审计

- 范围：`src\hwnas_fpga\search`、`training`、`data`、`run_search.py`、`run_retrain.py`。
- 重点问题：macro_f1/top1 选择逻辑、可行性过滤、训练/验证划分、class weight、候选记录。
- 预估文件数：约 18。
- 依赖关系：依赖任务 1 和 2。

### 任务名：HLS/LUT 生产链审计

- 范围：`hls_lut_builder`、`run_build_lut.py`、`generate_estimator_lut.py`、`src\hwnas_fpga\hardware\report_parser.py`、`lut_pipeline.py`。
- 重点问题：算子词汇对齐、formal LUT/status 文件、HLS/Vivado/board 三层指标回填。
- 预估文件数：跟踪文件约 70，生成物只抽样。
- 依赖关系：依赖任务 2，可与任务 3 并行。

### 任务名：部署与结果消费审计

- 范围：`src\hwnas_fpga\deploy`、`run_export.py`、`run_infer.py`、`scripts`、代表性 `results/artifacts`。
- 重点问题：checkpoint 元数据、ONNX/INT8 导出、paper table/figure 脚本读取的结果 schema。
- 预估文件数：约 40，加结果目录抽样。
- 依赖关系：依赖任务 1、3；硬件结论依赖任务 4。
