# 入口与配置闭环审计归档

归档时间：2026-05-17

审计状态：已完成静态审计；未修改源码；未运行训练、HLS、Vivado。

## 入口地图

| 入口文件 | 类型 | 当前状态 | 关键参数/行为 | 证据 |
|---|---|---|---|---|
| `run_search.py` | 当前主入口 | 主链路入口 | README 最小测试、完整搜索均使用该脚本；REPO_LAYOUT 标注为 unified search entrypoint | `E:\1\hwnas\hwnas\README.md:11`, `E:\1\hwnas\hwnas\README.md:14`, `E:\1\hwnas\hwnas\docs\REPO_LAYOUT.md:10` |
| `run_search_space_probe.py` | 前置探测入口 | 不训练，只做可行性探测 | README 将其作为搜索空间可行率验证 | `E:\1\hwnas\hwnas\README.md:17`, `E:\1\hwnas\hwnas\docs\REPO_LAYOUT.md:12` |
| `run_retrain.py` | 后续重训练入口 | 搜索后入口 | README 指向 `--run-dir results/<search_run_name>` | `E:\1\hwnas\hwnas\README.md:23`, `E:\1\hwnas\hwnas\docs\REPO_LAYOUT.md:14` |
| `run_infer.py` | 后续推理入口 | 搜索/重训练后消费 checkpoint | README 指向 `checkpoints/best_model.pt` | `E:\1\hwnas\hwnas\README.md:26`, `E:\1\hwnas\hwnas\README.md:103` |
| `run_export.py` | 后续导出入口 | ONNX/HLS stub/INT8 导出 | README 给出 `--prepare-hls` 和 `--quantize-int8` | `E:\1\hwnas\hwnas\README.md:29`, `E:\1\hwnas\hwnas\README.md:32` |
| `run_build_lut.py` | HLS/LUT 前置生产入口 | 构建 LUT 表 | README 描述 HLS report -> LUT | `E:\1\hwnas\hwnas\README.md:35`, `E:\1\hwnas\hwnas\README.md:180` |
| `run_rl_search.py` | 兼容入口 | 自动转发到 `run_search.py --search-method rl` | 包装器注入 `--search-method rl` 并 import `run_search.main` | `E:\1\hwnas\hwnas\run_rl_search.py:12`, `E:\1\hwnas\hwnas\run_rl_search.py:15` |
| `run_full_nas.py` | 废弃入口 | 明确 retired，返回 1 | 提示使用 `run_search.py -> run_retrain.py -> run_export.py` | `E:\1\hwnas\hwnas\run_full_nas.py:2`, `E:\1\hwnas\hwnas\run_full_nas.py:10` |
| `retrain_best.py` | 废弃入口 | 明确 retired，返回 1 | 提示使用 `run_retrain.py` | `E:\1\hwnas\hwnas\retrain_best.py:2`, `E:\1\hwnas\hwnas\retrain_best.py:10` |
| `visualize_results.py` | 兼容入口 | 转发到 `scripts/visualize_results.py` | 使用 `runpy.run_path` 执行维护脚本 | `E:\1\hwnas\hwnas\visualize_results.py:2`, `E:\1\hwnas\hwnas\visualize_results.py:12` |

## 配置字段流向

| 配置字段 | 定义位置 | 消费位置 | 默认值来源 | 审计结论 |
|---|---|---|---|---|
| `project.seed` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:3` | `run_search.py` 设置 torch seed | `pick(cli, config, 42)` | CLI > config > default 顺序清晰：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:55`, `E:\1\hwnas\hwnas\run_search.py:227` |
| `project.output_dir` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:4` | `ExperimentTracker(output_root=...)` | `results` | 由统一 tracker 创建 run 目录：`E:\1\hwnas\hwnas\run_search.py:259`, `E:\1\hwnas\hwnas\run_search.py:276` |
| `project.run_name` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:5` | `ExperimentTracker(run_name=...)` | 时间戳 slug | 明确控制 run 目录名：`E:\1\hwnas\hwnas\run_search.py:260`, `E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:101` |
| `dataset.*` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:7` | `create_data_pipeline()` | dummy/NKSID 默认 | `num_classes/split_seed` 进入数据链路：`E:\1\hwnas\hwnas\run_search.py:432`, `E:\1\hwnas\hwnas\run_search.py:444` |
| `search_space.*` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:18` | `runtime.create_search_space()` | profile defaults | stage/op/head 字段进入 SearchSpaceConfig：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:123`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:154` |
| `constraints.*` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:45` | cost estimator/searcher | 空约束或板卡默认 | 约束覆盖 latency、LUT、DSP、BRAM、power、bandwidth、offchip：`E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:46` |
| `hardware.lut_path` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:60` | `load_lut_query_engine()` | None | 相对路径按当前工作目录解析：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:167` |
| `hardware.formal_lut_status_path` | `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:61` | `LutQueryEngine.load_formal_status_json()` | strict 模式必填 | strict formal LUT 缺 status 会直接报错：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:163`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:181` |
| `hardware.operator_manifest_path` | 主配置未设置 | 默认读取 `hls_lut_builder/configs/operator_manifest.yaml` | runtime 默认路径 | 这是跨层默认依赖：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:213`, `E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:221` |

## 数据流图

```text
config YAML + CLI args
  -> runtime.pick(cli, config, default)
     E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:55
  -> run_search.py resolves dataset/search/hardware/project
     E:\1\hwnas\hwnas\run_search.py:227
  -> ExperimentTracker
     E:\1\hwnas\hwnas\run_search.py:276
       -> config.yaml + cli_args.json
          E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:158
       -> run_info.json
          E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:153
       -> candidates.jsonl / candidates/*.json
          E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:190
       -> best_candidate.json / best_model.pt
          E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:229
       -> candidates.csv / summary.json / run_info.json
          E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:262
```

## 发现列表

### P1

现象：`--resume` 只支持 RL，且依赖 `candidates.jsonl`、`search_state.json`、`controller_latest.pt` 三件套。
证据：`E:\1\hwnas\hwnas\run_search.py:80`, `E:\1\hwnas\hwnas\run_search.py:87`, `E:\1\hwnas\hwnas\run_search.py:89`, `E:\1\hwnas\hwnas\run_search.py:91`, `E:\1\hwnas\hwnas\run_search.py:518`。
影响：非 RL 搜索不能恢复；缺任一 resume artifact 会中断。
具体审计结论：恢复语义不是全局入口语义，而是 RL 专用语义。
后续动作：在 `run_search.py` 的 `--resume` help 文案和 README 运行说明中明确 “RL only”，并列出必需 artifact。

### P2

现象：`hardware.operator_manifest_path` 在主配置中未显式声明，但 runtime 默认跨到 `hls_lut_builder/configs/operator_manifest.yaml`。
证据：主配置 hardware 段只有 `lut_path/formal_lut_status_path` 等字段：`E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:55`；默认路径在 runtime 中拼出：`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:221`。
影响：搜索配置可复现性依赖一个未写入 config 的 HLS 侧文件。
具体审计结论：这是配置层到 HLS/LUT 生产层的隐式依赖。
后续动作：在正式 search config 中显式加入 `hardware.operator_manifest_path: hls_lut_builder/configs/operator_manifest.yaml`。

### P2

现象：`run_operator_ablation.py` 直接 import 顶层脚本 `run_backbone_baseline.py` 作为工具模块。
证据：`E:\1\hwnas\hwnas\run_operator_ablation.py:18`。
影响：入口脚本之间存在横向依赖，入口边界不如 `src\hwnas_fpga` 包内 API 清晰。
具体审计结论：这是入口层反向复用另一个入口脚本的工具函数。
后续动作：把被复用的 baseline 工具函数迁移到 `src\hwnas_fpga` 包内模块，并让两个入口脚本共同 import 包内函数。

### P3

现象：废弃入口仍在根目录，但运行时会明确提示 retired 并返回 1。
证据：`E:\1\hwnas\hwnas\run_full_nas.py:2`, `E:\1\hwnas\hwnas\run_full_nas.py:10`, `E:\1\hwnas\hwnas\retrain_best.py:2`, `E:\1\hwnas\hwnas\retrain_best.py:10`。
影响：误运行不会执行旧流程，但根目录入口数量增加认知成本。
具体审计结论：这是兼容保留，不是活跃主链路。
后续动作：在 README 的入口列表中把 `run_full_nas.py`、`retrain_best.py` 标记为 retired wrapper。

## 无法确认项

- 未运行任何训练或 resume 流程，因此未动态验证 `controller_latest.pt` 与 `search_state.json` 的恢复完整性。
- 未运行 HLS/Vivado，因此未验证 `lut_path` 与 `formal_lut_status_path` 对应内容是否与当前 operator manifest 完全一致。
- 未抽样打开历史 results 目录，因此未判断旧 run artifact 是否全部符合当前 `ExperimentTracker` schema。
