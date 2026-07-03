# 部署与结果消费审计

范围：`run_export.py`、`run_infer.py`、根目录 `visualize_results.py`、`src/hwnas_fpga/deploy/*`、`scripts/visualize_results.py`、`scripts/generate_paper_search_table.py`、`scripts/generate_backbone_baseline_table.py`、`scripts/plot_hw_nas_publication_figures.py`、`scripts/audit_run_storage.py`，以及代表性 `results/artifacts` schema 抽样。

执行方式：静态审计。未运行真实 ONNX 导出、INT8 导出、HLS/Vivado、板卡流程或长耗时推理。`.pt` 二进制 checkpoint 未做 `torch.load` 抽样验证，本审计只引用 producer/consumer 代码和可读 JSON/YAML/CSV schema。

前置审计：依赖 `docs/audit_entry_config.md` 与 `docs/audit_search_training_metrics.md` 的入口、tracker 和指标口径结论。硬件指标解释只引用当前范围内的 consumer 行为，不重新审计 HLS/LUT 生产链。

## 1. artifact schema 表

结论：下游链路支持两类 checkpoint schema、三类导出 artifact、两类 results schema。`final_best_model.pt` 与 `best_model.pt` 均可被 `load_checkpoint_model` 静态识别；ONNX metadata 足够保存输入 shape、class names、architecture 和 opset，但 search checkpoint 的 `checkpoint_metrics` 会丢失；INT8 `.pt` package 比 sidecar `.json` 完整。

| artifact | 产生位置 | 消费位置 | 必需字段 | 可选字段 | 证据 |
| --- | --- | --- | --- | --- | --- |
| `checkpoints/best_model.pt` | `ExperimentTracker.save_best_candidate` 在保存 best candidate 时写入 checkpoint | `deploy.inference.load_checkpoint_model` | `candidate.encoding`、`model_state_dict` | `candidate.metrics`、`history`、`extra` | `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:229`、`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:245`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:36`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:105` |
| `checkpoints/final_best_model.pt` | `run_retrain.py` 写入 retrain checkpoint payload | `deploy.inference.load_checkpoint_model` | `architecture`、`model_state_dict` | `source_candidate`、`metrics`、`history` | `E:\1\hwnas\hwnas\run_retrain.py:187`、`E:\1\hwnas\hwnas\run_retrain.py:195`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:41`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:46` |
| `model.onnx` + `model.json` | `run_export.py` 调用 `export_checkpoint_to_onnx` | 外部 ONNX runtime 或后续 HLS stub | `input_shape`、`opset_version`、`model_info`、`architecture` | `dynamic_axes`、`checkpoint_metrics`、`class_names` | `E:\1\hwnas\hwnas\run_export.py:54`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:115`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:159` |
| `quantized_weights_int8.pt` | `export_checkpoint_quantized_weights` | 外部部署或 HLS 消费者，当前仓库内未发现直接消费代码 | `format`、`architecture`、`quantization`、`weights`、`scales` | `candidate`、`class_names` | `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:111`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:147` |
| `quantized_weights_int8.json` | `export_checkpoint_quantized_weights` 写 sidecar summary | 人读或外部 inventory | `bit_width`、`scheme`、`tensors[].name`、`tensors[].shape`、`tensors[].scale`、`tensors[].dtype` | size/compression summary | `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:98`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:149` |
| `hls_project/hls_project.json`、`run_hls.tcl`、`README.txt` | `create_hls_project_stub` | 手动 HLS scaffold，不是 formal LUT flow | `onnx_path`、`config`、`generated_files` | `board`、`fpga_part`、`clock_mhz` | `E:\1\hwnas\hwnas\run_export.py:64`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:34`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:54` |
| `results/summary.json` | `ExperimentTracker.finalize` | paper table、visualization、storage review | `status`、`total_evaluated`、`feasible`、`infeasible`、`best_candidate` | `extra` | `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:282`、`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:293`、`E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\results\summary.json:72` |
| `results/candidates.json`、`results/candidates.jsonl`、`candidates.csv` | `ExperimentTracker.record_candidate` 和 `finalize` | visualization、paper table、publication figures | candidate `arch_id`、`encoding`、`metrics` | JSONL `feasible`、`cost_estimate`、`history`、`extra` | `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:199`、`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:273`、`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:304`、`E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\results\candidates.jsonl:1` |
| `run_info.json`、`config.yaml` | `ExperimentTracker` 和入口配置落盘 | inference settings、result consumers、storage review | `status`、`run_name`、`dataset.image_size`、`dataset.input_channels` | `search_method`、`hardware.*` | `E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\run_info.json:3`、`E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\config.yaml:6`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:51` |

## 2. 部署链路图

结论：部署链路是 `checkpoint -> ONNX/INT8/HLS stub -> metadata/inference`，不是 `checkpoint -> 完整 HLS/Vivado bitstream`。HLS stub 只生成占位 Tcl 和说明文件，formal LUT 生产链仍在 `hls_lut_builder`。

```text
results/<run>/
  checkpoints/best_model.pt
    producer: ExperimentTracker.save_best_candidate
    schema: candidate + model_state_dict
      |
      | run_export.py --checkpoint
      v
  model.onnx + model.json
    metadata: input_shape, opset_version, class_names, architecture, checkpoint_metrics
      |                         |
      | --quantize-int8         | --prepare-hls
      v                         v
  quantized_weights_int8.pt     hls_project/
  quantized_weights_int8.json     hls_project.json
                               run_hls.tcl with placeholder add_files

results/<retrain-run>/
  checkpoints/final_best_model.pt
    producer: run_retrain.py
    schema: architecture + metrics + model_state_dict
      |
      v
  run_infer.py
    reads checkpoint, run_dir/config.yaml, run_dir/cli_args.json, class name fallback
```

证据：

- `run_export.py` 以 `--checkpoint` 为必需参数，并支持 `--prepare-hls`、`--quantize-int8`、`--report`：`E:\1\hwnas\hwnas\run_export.py:23`、`E:\1\hwnas\hwnas\run_export.py:32`、`E:\1\hwnas\hwnas\run_export.py:36`、`E:\1\hwnas\hwnas\run_export.py:37`。
- ONNX metadata 写入 `input_shape`、`opset_version`、`dynamic_axes`、`model_info`，并追加 checkpoint metadata：`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:115`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:159`。
- 推理入口恢复 `image_size`、`input_channels` 后加载 checkpoint 并输出预测 JSON：`E:\1\hwnas\hwnas\run_infer.py:38`、`E:\1\hwnas\hwnas\run_infer.py:44`、`E:\1\hwnas\hwnas\run_infer.py:75`。
- HLS stub 的 Tcl 保留 `# add_files <generated_hls_sources.cpp>` 占位，README 要求手动放入 HLS C/C++ sources：`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:54`、`E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:60`。
- formal LUT flow 是 `hls_lut_builder` 下的独立 flow，README 明确该目录包含 formal operator-level LUT build flow：`E:\1\hwnas\hwnas\hls_lut_builder\README.md:3`。

## 3. 结果消费矩阵

结论：结果消费分成三种 schema 口径。旧 `scripts/visualize_results.py` 使用 `accuracy` 作为主排序和图表字段；paper table 与 publication figures 使用 `macro_f1/top1/latency/LUT/DSP/BRAM/power`；storage audit 只做 inventory，不是删除策略。

| 脚本 | 输入文件 | 输出文件 | 使用指标 | schema fallback | 风险 |
| --- | --- | --- | --- | --- | --- |
| `run_export.py` | `best_model.pt` 或 `final_best_model.pt` | `model.onnx`、`model.json`、可选 `hls_project/*`、可选 INT8 package | `checkpoint_metrics`、`input_shape`、`class_names`、architecture | 通过 `load_checkpoint_model` 支持 `candidate` 和 `architecture` 两类 checkpoint | search checkpoint 的 root `metrics` 为空时，ONNX metadata 的 `checkpoint_metrics` 为空 |
| `run_infer.py` | checkpoint、image file/dir、可选 class names | 可选 inference JSON | prediction confidence、topk | `config.yaml`、`cli_args.json`、NKSID class fallback、`class_{i}` fallback | checkpoint 被复制到非 run_dir 后，`image_size/input_channels` 可能退回默认值 |
| 根目录 `visualize_results.py` | CLI 参数透传 | 由 `scripts/visualize_results.py` 决定 | 同 maintained script | wrapper 直接 runpy | 兼容入口本身清晰 |
| `scripts/visualize_results.py` | `summary.json`、`candidates.json`，fallback `candidates.jsonl`，fallback `search_state.json` | png、csv、tex、html summary | `accuracy`、`latency_ms`、`dsp`、`lut`、`bram` | `candidates.json` 优先，JSONL 次之，search_state 用于 partial | 与 macro_f1/top1 优先口径不一致 |
| `scripts/generate_paper_search_table.py` | `summary.json`、`candidates.jsonl`，fallback `candidates.json` 和 `search_state.json` | markdown、latex table | `macro_f1`、`top1`、`top5`、`latency_ms`、`dsp`、`bram`、`lut`、`power_w` | search_state fallback 默认 `selection_metric=macro_f1` | 与旧 visualization 的 primary metric 不同，但与当前论文口径一致 |
| `scripts/generate_backbone_baseline_table.py` | `results/backbone_summary.json`、可选 selected pool/run_info | `fair_macro_backbone_table.md` | `macro_f1`、`top1`、`top5`、`fpga_latency_ms`、`peak_dsp`、`peak_bram`、`peak_lut`、`power_w`、`feasible` | 无 summary 文件直接报错 | 只适用于 backbone baseline schema，不适用于 search candidates schema |
| `scripts/plot_hw_nas_publication_figures.py` | backbone CSV、search `candidates.csv` | publication figures、manifest | `macro_f1`、`top1`、`latency_ms`、`lut`、`dsp`、`bram`、`power_w` | 缺字段直接报错 | 与 paper table 一致，但与旧 visualization 不一致 |
| `scripts/audit_run_storage.py` | top-level `results/`、`artifacts/` 目录 | markdown inventory | 文件数、大小、tag、suggested action | heuristic 分类 | 代码和已生成文档都声明 inventory only，不构成删除策略 |

证据：

- `scripts/visualize_results.py` 读取 `summary.json`，优先 `candidates.json`，再 fallback `candidates.jsonl`：`E:\1\hwnas\hwnas\scripts\visualize_results.py:49`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:58`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:80`。
- `scripts/visualize_results.py` 的 summary、Top-K、散点、Pareto、LaTeX table 以 `accuracy` 为主：`E:\1\hwnas\hwnas\scripts\visualize_results.py:158`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:191`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:219`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:291`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:603`。
- `scripts/generate_paper_search_table.py` 以 `macro_f1` fallback 选择 best candidate，并输出 Macro-F1、Top-1、Top-5、Latency、DSP、BRAM、LUT、Power：`E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:61`、`E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:124`、`E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:176`、`E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:276`、`E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:342`。
- `scripts/plot_hw_nas_publication_figures.py` 要求 search CSV 包含 `macro_f1/top1/latency_ms/lut/dsp/bram/power_w`，manifest 也记录同一组指标：`E:\1\hwnas\hwnas\scripts\plot_hw_nas_publication_figures.py:155`、`E:\1\hwnas\hwnas\scripts\plot_hw_nas_publication_figures.py:419`、`E:\1\hwnas\hwnas\scripts\plot_hw_nas_publication_figures.py:540`。
- `scripts/audit_run_storage.py` 生成 inventory only 文档，不执行删除：`E:\1\hwnas\hwnas\scripts\audit_run_storage.py:2`、`E:\1\hwnas\hwnas\scripts\audit_run_storage.py:75`、`E:\1\hwnas\hwnas\docs\RUN_STORAGE_AUDIT.md:3`。

## 4. 发现列表

### P1 - ONNX metadata 对 search checkpoint 丢失 best metrics

现象：`best_model.pt` 由 `ExperimentTracker.save_best_candidate` 写入 `candidate.metrics`，但 `export_checkpoint_to_onnx` 只读取 checkpoint root 层 `payload.get("metrics", {})`。因此从 search `best_model.pt` 导出的 `model.json` 可能缺失 `macro_f1/top1/latency/LUT/DSP/BRAM/power`，而 retrain `final_best_model.pt` 因 root 层有 `metrics` 不受影响。

证据：search checkpoint payload 字段在 `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:246` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:251`，候选 metrics 的 JSON schema 在 `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:344` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:350`；ONNX metadata 只取 root metrics 在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:159` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py:165`；retrain checkpoint root metrics 在 `E:\1\hwnas\hwnas\run_retrain.py:187` 到 `E:\1\hwnas\hwnas\run_retrain.py:193`。

影响：使用 search `best_model.pt` 做 ONNX 导出时，导出 metadata 不能追溯最关键的 macro_f1、top1、latency、LUT、DSP、BRAM、power，破坏部署 artifact 的可复现性说明。

具体动作：修改 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\export.py` 的 `export_checkpoint_to_onnx`，将 `checkpoint_metrics` 改为 `payload.get("metrics") or payload.get("candidate", {}).get("metrics") or payload.get("source_candidate", {}).get("metrics") or {}`，并补一个覆盖 `best_model.pt` schema 的短测试。

### P1 - 旧可视化脚本仍以 accuracy 作为主口径

现象：`scripts/visualize_results.py` 的 summary、Top-K、accuracy-latency、Pareto 和 LaTeX table 都以 `accuracy` 为主；同仓库的 paper table 和 publication figures 以 `macro_f1/top1/latency/LUT/DSP/BRAM/power` 为主。

证据：旧 visualization 的 `Best Acc` 和 accuracy 排序在 `E:\1\hwnas\hwnas\scripts\visualize_results.py:158`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:191`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:219`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:291`、`E:\1\hwnas\hwnas\scripts\visualize_results.py:603`；paper table 使用 `macro_f1/top1/top5/latency/dsp/bram/lut/power` 在 `E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:276` 到 `E:\1\hwnas\hwnas\scripts\generate_paper_search_table.py:314`；publication figures 要求字段在 `E:\1\hwnas\hwnas\scripts\plot_hw_nas_publication_figures.py:155` 到 `E:\1\hwnas\hwnas\scripts\plot_hw_nas_publication_figures.py:159`。

影响：同一 run 可能在旧 visualization 与 paper scripts 中得到不同的 best candidate 和 Pareto 展示，尤其违背当前项目对 macro_f1、top1 和硬件指标的优先级。

具体动作：在 `E:\1\hwnas\hwnas\scripts\visualize_results.py` 增加 `--metric` 参数并默认 `macro_f1`；将 Top-K、Pareto、summary table、LaTeX table 的 primary score 改为该参数；保留 `accuracy` 作为可选字段展示。

### P2 - 推理恢复依赖 checkpoint 所在目录结构

现象：`resolve_inference_settings` 假设 `run_dir = checkpoint.parent.parent`，只从该目录的 `config.yaml` 和 `cli_args.json` 恢复 `image_size/input_channels`。如果用户只复制 `best_model.pt` 或 `final_best_model.pt` 到部署目录，推理会 fallback 到 `image_size=224`、`input_channels=1` 或依赖 CLI override。

证据：run_dir 推导和 config/cli 读取在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:51` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:70`；默认值在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:124` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py:135`；CLI override 在 `E:\1\hwnas\hwnas\run_infer.py:27` 到 `E:\1\hwnas\hwnas\run_infer.py:28`；代表性 run 的 dataset shape 在 `E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\config.yaml:6` 到 `E:\1\hwnas\hwnas\results\formal_lut_compact_4stage_nksid_main_20ep_100ms\config.yaml:11`。

影响：checkpoint 脱离原 run 目录后，推理预处理 shape 可能与训练配置不一致，属于 silent bug 类型。

具体动作：在 `E:\1\hwnas\hwnas\run_infer.py` 增加 `--metadata` 参数，读取 `model.json` 或显式 metadata 文件；在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\inference.py` 中让 `resolve_inference_settings` 支持 metadata fallback，并在输出 JSON 中记录 settings 来源。

### P2 - INT8 sidecar JSON 不是完整部署 manifest

现象：INT8 `.pt` package 包含 `format`、`architecture`、`candidate`、`class_names`、`weights`、`scales`，但 sidecar `.json` 只写 quantization summary 和 tensor summaries。

证据：package 完整字段在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:111` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:119`；summary 字段在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:98` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:110`；写 sidecar JSON 在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:149` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py:153`。

影响：非 PyTorch 消费者或 HLS 侧工具若只读取 `.json`，无法从 sidecar 追溯 architecture、candidate、class names 与 package 路径。

具体动作：在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\quantization.py` 将 sidecar 改为 manifest，保留现有 `quantization` summary，并新增 `format`、`package_path`、`architecture`、`candidate.arch_id`、`class_names`、每个 tensor 的 `scale/dtype/shape`。

### P2 - HLS stub 容易被误读为完整部署链

现象：`run_export.py --prepare-hls` 创建的是 HLS project stub；生成 Tcl 仍有 `# add_files <generated_hls_sources.cpp>` 占位，README 也要求用户手动放入 HLS C/C++ sources。

证据：CLI help 写的是 HLS project stub 在 `E:\1\hwnas\hwnas\run_export.py:32`；stub manifest 与 Tcl 写入在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:34` 到 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:59`；README 文本在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py:60`；formal LUT flow 独立声明在 `E:\1\hwnas\hwnas\hls_lut_builder\README.md:3`。

影响：用户可能把 `hls_project` 当成完整 HLS/Vivado 生产链，实际它不生成 C/C++ kernel、不接入 `hls_lut_builder` formal LUT case，不保证硬件指标可比较。

具体动作：在 `E:\1\hwnas\hwnas\src\hwnas_fpga\deploy\hls_backend.py` 生成的 `hls_project.json` 中新增 `"stub_only": true`、`"requires_generated_sources": true`、`"formal_lut_flow": "hls_lut_builder"`，并在 `README.txt` 写明它不是 formal LUT builder。

### P3 - storage audit 是 inventory，不是删除策略

现象：`audit_run_storage.py` 和已生成 `RUN_STORAGE_AUDIT.md` 都声明 inventory only，分类字段是 suggested action。

证据：脚本文档字符串在 `E:\1\hwnas\hwnas\scripts\audit_run_storage.py:2`；markdown 生成文本在 `E:\1\hwnas\hwnas\scripts\audit_run_storage.py:75` 到 `E:\1\hwnas\hwnas\scripts\audit_run_storage.py:83`；已生成文档声明在 `E:\1\hwnas\hwnas\docs\RUN_STORAGE_AUDIT.md:3` 到 `E:\1\hwnas\hwnas\docs\RUN_STORAGE_AUDIT.md:10`。

影响：当前证据不支持把它视为自动清理或删除工具；它只能作为结果目录盘点输入。

具体动作：无业务修正动作。后续跨层综合中把 `scripts/audit_run_storage.py` 归入 inventory/reporting consumer，不归入部署或删除链路。

## 5. 孤立或非本项目产物列表

结论：`outputs/presentations/solar-system` 与 HW-NAS 部署链没有 producer/consumer 关系，应归为非本项目产物或其他任务输出；`artifacts/search_smoke_results` 等目录可以纳入 storage inventory，但不应作为核心部署 artifact。

| 路径 | 证据 | 为什么不纳入部署链 |
| --- | --- | --- |
| `E:\1\hwnas\hwnas\outputs\presentations\solar-system` | `package.json` 只有 presentation/package 信息：`E:\1\hwnas\hwnas\outputs\presentations\solar-system\package.json:1`；build report 输出 pptx 和 slide preview：`E:\1\hwnas\hwnas\outputs\presentations\solar-system\scratch\build-report.json:2`、`E:\1\hwnas\hwnas\outputs\presentations\solar-system\scratch\build-report.json:6` | 内容是太阳系演示文稿产物，不含 HW-NAS checkpoint、candidate、metrics、ONNX、INT8 或 HLS schema |
| `E:\1\hwnas\hwnas\outputs\presentations\solar-system\scratch\assets` | asset manifest 记录 NASA 图片标题和 URL：`E:\1\hwnas\hwnas\outputs\presentations\solar-system\scratch\asset-manifest.json:3`、`E:\1\hwnas\hwnas\outputs\presentations\solar-system\scratch\asset-manifest.json:10` | 这是 presentation 素材目录，不参与 sonar image classification 数据流 |
| `E:\1\hwnas\hwnas\artifacts\search_smoke_results` | storage audit 将其归为 `smoke-temp` 和 `archive_candidate`：`E:\1\hwnas\hwnas\docs\RUN_STORAGE_AUDIT.md:20` | 可作为 storage inventory 项，不能作为 deployment schema 权威样本 |

## 自检

- 覆盖了 export、infer、quantization、HLS stub、visualization、paper table、publication figures、storage audit。
- 列出了 checkpoint schema、results schema、paper table schema、HLS stub schema 的 producer 和 consumer。
- 明确指出旧 visualization 使用 `accuracy`，新 paper/figure 脚本使用 `macro_f1/top1/latency/LUT/DSP/BRAM/power`。
- 识别了 `outputs/presentations/solar-system` 作为孤立或非本项目产物。
- 未运行真实导出、推理、HLS/Vivado、板卡测量或长任务；`.pt` 内容仍需用 `torch.load` 抽样验证。
