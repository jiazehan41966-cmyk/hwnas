# HW-NAS FPGA Sonar

面向水下声呐图像分类/识别任务的硬件感知神经架构搜索（HW-NAS）项目。目标是在 FPGA 资源约束下，完成从搜索、训练、重训练、导出到部署准备的完整闭环。

---

## 快速开始

```bash
# 最小化测试
python3 run_search.py --search-method rl --episodes 3 --train-epochs 1 --batch-size 8

# 完整搜索
python3 run_search.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml

# A1: 搜索空间可行率验证（随机采样，不训练）
python3 run_search_space_probe.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml --num-samples 200

# 指定结果目录
python3 run_search.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml --output-dir results

# 重训练搜索得到的最优架构
python3 run_retrain.py --run-dir results/<search_run_name>

# 使用搜索得到的最优模型识别图片
python3 run_infer.py --checkpoint results/<run_name>/checkpoints/best_model.pt --input /path/to/image_or_dir

# 导出 ONNX 并生成 HLS 工程骨架
python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --prepare-hls

# 额外导出 INT8 权重量化包
python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --quantize-int8

# 从 HLS profiling 报告构建 LUT 表
python3 run_build_lut.py --manifest configs/hardware/lut_manifest_example.yaml --output artifacts/fpga_lut.pkl

# 声呐图像 PSNR/SSIM（默认 input_as_reference，仅作算子影响分析）
python3 scripts/measure_sonar_image_quality.py --data-dir data/NKSID --split val --fold 0

# 冻结评估协议（外层5折 × 多seed，唯一可对外声明的分类指标入口）
python3 run_eval_protocol.py --arch mobilenet_v2 --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44
```

详细使用说明：[docs/QUICKSTART.md](docs/QUICKSTART.md)

> Current formal search entry point: `configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml`
>
> Legacy generic `nksid_fpga_search*.yaml` configs have moved to `configs/search/legacy/`.
>
> The current MobileNetV2 mainline no longer treats `dw_pw_conv` as a default searchable operator.

---

## 结果目录

每次搜索都会在 `results/<run_name>/` 下生成完整运行目录，默认包含：

```text
results/<run_name>/
|-- config.yaml
|-- cli_args.json
|-- run_info.json
|-- logs/
|   `-- console.log
|-- results/
|   |-- baseline.json
|   |-- dataset_summary.json
|   |-- search_space_summary.json
|   |-- candidates.jsonl
|   |-- candidates.json
|   |-- candidates.csv
|   |-- pareto_front.json
|   |-- pareto_selection.json
|   |-- summary.json
|   `-- candidates/
|       `-- <arch_id>.json
`-- checkpoints/
    |-- search_state.json
    |-- best_candidate.json
    |-- best_model.pt
    |-- final_best_model.pt      # retrain 后生成
    |-- controller_latest.pt     # RL 搜索时生成
    `-- controller_best.pt       # RL 搜索时生成
```

---

## 项目文档

| 文档 | 说明 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 整体技术架构与模块边界 |
| [docs/method_design.md](docs/method_design.md) | 问题模型、方法设计与实现映射 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速开始与使用指南 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 实现进展与功能总结 |
| [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md) | 审计、证据与交接索引 |
| [docs/REVIEW.md](docs/REVIEW.md) | 当前仓库审查入口与旧快照说明 |
| [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) | 冻结评估协议（外层5折×多seed，唯一可声明入口） |
| [docs/FIRST_PRINCIPLES_AUDIT_20260703.md](docs/FIRST_PRINCIPLES_AUDIT_20260703.md) | 数据协议、算子语义与证据强度重审 |
| [docs/PHASE0_V3_BOARD_RESULTS.md](docs/PHASE0_V3_BOARD_RESULTS.md) | Phase0 v3 low-DSP full-route and COM5 board-claimable results |
| [docs/PHASE0_V4_SONAR_RESULTS.md](docs/PHASE0_V4_SONAR_RESULTS.md) | Phase0 v4 search/retrain/route/COM5/image-quality handoff |
| [docs/SEARCH_CONFIG_CANONICAL.md](docs/SEARCH_CONFIG_CANONICAL.md) | 当前规范搜索配置说明 |
| [docs/REPO_LAYOUT.md](docs/REPO_LAYOUT.md) | 仓库布局说明 |

---

## Current Evidence Status

The 2026-07-03 first-principles audit supersedes broader interpretations of the
older evidence. Reported Phase0 classification scores are legacy fold-0
validation results, not untouched-test generalization estimates. The current
`denoise` and `edge` HLS templates are also not semantically identical to their
train-time PyTorch blocks. New canonical searches load the semantic-safe
operator policy and exclude both operators until numeric parity and matching
weight export exist. See
[docs/FIRST_PRINCIPLES_AUDIT_20260703.md](docs/FIRST_PRINCIPLES_AUDIT_20260703.md).

The 2026-06-22 Phase0 v4 sonar snapshot contains 7 Pareto route-screen rows:
6 are route-clean with stable five-run COM5 evidence and 1 (`rl_arch_116`) is
full-route-fail. Five v4 candidates also completed retrain150. See
[docs/PHASE0_V4_SONAR_RESULTS.md](docs/PHASE0_V4_SONAR_RESULTS.md) for exact
metrics and artifact paths.

Evidence boundaries:

- search `macro_f1`/`top1` and hardware fields are proxy evidence;
- retrain150 metrics are PyTorch validation-set evidence;
- COM5 is deterministic harness-input latency/output sanity, not full NKSID
  board accuracy;
- PSNR/SSIM with `input_as_reference` is operator-effect analysis, not
  clean-reference restoration quality;
- measured power/energy remains `not measured`.
- compact reproducible audit evidence is tracked under
  `artifacts/first_principles_audit_20260703/`; large generated result trees
  remain local and Git-ignored.

The Phase0 v3 low-DSP baseline remains frozen as a comparison source in
[docs/PHASE0_V3_BOARD_RESULTS.md](docs/PHASE0_V3_BOARD_RESULTS.md).

---

## 项目结构

```text
.
|-- README.md                         # 项目说明
|-- run_search.py                     # 搜索入口脚本
|-- run_retrain.py                    # 重训练入口脚本
|-- run_infer.py                      # 推理入口脚本
|-- run_export.py                     # ONNX / HLS 导出入口
|-- run_build_lut.py                  # HLS report -> LUT 构建入口
|-- configs/                          # 配置文件
|   |-- search/
|   |   |-- nksid_fpga_search_mobile_anchor_av7k325.yaml
|   |   `-- legacy/
|   `-- hardware/
|       |-- zynq7020.yaml
|       |-- kintex7_xc7k325.yaml
|       `-- lut_manifest_example.yaml
|-- docs/                             # 文档
|-- hls_lut_builder/                  # HLS profiling / LUT 构建辅助工具
|-- reference/                        # 参考代码库分析
|   |-- FBNet/
|   |-- HW-NAS-Bench/
|   |-- TinyTNAS/
|   `-- ANALYSIS.md
|-- tests/                            # 测试
`-- src/hwnas_fpga/
    |-- interfaces.py                 # 接口定义
    |-- runtime.py                    # 运行时辅助逻辑
    |-- search_space/                 # 搜索空间
    |   `-- space.py
    |-- hardware/                     # 硬件估计与 LUT
    |   |-- cost.py
    |   |-- lookup_table.py
    |   |-- lut_pipeline.py
    |   `-- report_parser.py
    |-- metrics/                      # PSNR / SSIM / MSE
    |-- models/                       # 模型构建
    |   `-- builder.py
    |-- data/                         # 数据加载
    |   `-- dataset.py
    |-- training/                     # 训练与重训练
    |   |-- trainer.py
    |   `-- retrain.py
    |-- search/                       # 搜索算法
    |   |-- searcher.py
    |   |-- constrained.py
    |   `-- pareto.py
    `-- deploy/                       # ONNX / HLS / 推理
        |-- export.py
        |-- quantization.py
        |-- hls.py
        |-- hls_backend.py
        |-- report_parser.py
        `-- inference.py
```

---

## 核心功能

### 已实现

- **搜索空间**：stage-based 可搜索架构空间。
- **硬件估计**：分析模型 + LUT 查找表 + board profile。
- **模型构建**：`ArchitectureSpec` 到 PyTorch Model。
- **训练评估**：完整训练流程。
- **搜索算法**：RL 搜索、约束剪枝与 Pareto 选择。
- **重训练**：对 best architecture 进行独立最终重训练。
- **部署导出**：ONNX 导出、HLS 项目骨架、report parser。
- **INT8 量化**：导出权重量化包，供 FPGA/HLS 后端使用。
- **LUT profiling**：从 HLS report 构建 LUT table。
- **HLS/Vivado/COM5 证据链**：full-route gate、固定输入板测与稳定性产物。
- **声呐图像质量**：PSNR/SSIM/MSE dataset mode 与 paired mode。
- **对比实验**：Fused MBConv vs 标准 MBConv、有/无早期剪枝。
- **Pareto 优化**：多目标优化与前沿分析。

### 待实现

- 权重共享超网训练。
- 四路声呐消融完整闭环。
- NKSID 完整验证集的样本级板上准确率。
- 外部功率计或可读监控路径的实测功耗/能耗闭环。
- 独立的 HLS/LUT 生产链审计归档。

---

## LUT Profiling

真实 FPGA profiling 可以通过 `run_build_lut.py` 从 Vivado/Vitis HLS 报告构建查找表：

```bash
python3 run_build_lut.py \
  --manifest configs/hardware/lut_manifest_example.yaml \
  --output artifacts/fpga_lut.pkl \
  --summary-json artifacts/fpga_lut_summary.json
```

生成的 `fpga_lut.pkl` 可以直接接入搜索配置：

```yaml
hardware:
  board: zynq7020
  lut_path: artifacts/fpga_lut.pkl
  use_dummy_lut: false
```

---

## 方案参考

| 参考库 | 用途 |
|---|---|
| **FBNet** | stage-based 搜索空间、LUT 架构设计 |
| **TinyTNAS** | 约束驱动、时间限制搜索 |
| **HW-NAS-Bench** | 硬件指标设计 |
| **HW-PR-NAS** | Pareto 排名保持 |
| **DARTS** | 可微 NAS 基线（参考用） |

详见：[reference/ANALYSIS.md](reference/ANALYSIS.md)
