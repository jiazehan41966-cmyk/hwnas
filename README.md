# HW-NAS FPGA Sonar

面向水下声呐图像分类/识别任务的硬件感知神经架构搜索（HW-NAS）项目，目标是在 FPGA 资源约束下完成从搜索、训练到部署的闭环。

---

## 快速开始

```bash
# 最小化测试
python3 run_search.py --search-method rl --episodes 3 --train-epochs 1 --batch-size 8

# 完整搜索
python3 run_search.py --config configs/search/nksid_fpga_search.yaml

# A1: 搜索空间可行率验证（随机采样，不训练）
python3 run_search_space_probe.py --config configs/search/nksid_fpga_search_lightweight_sonar_av7k325.yaml --num-samples 200

# 指定结果目录
python3 run_search.py --config configs/search/nksid_fpga_search.yaml --output-dir results

# 重训搜索得到的最优架构
python3 run_retrain.py --run-dir results/<search_run_name>

# 用搜索得到的最优模型识别图片
python3 run_infer.py --checkpoint results/<run_name>/checkpoints/best_model.pt --input /path/to/image_or_dir

# 导出 ONNX 并生成 HLS 工程骨架
python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --prepare-hls

# 额外导出 INT8 权重量化包
python3 run_export.py --checkpoint results/<run_name>/checkpoints/final_best_model.pt --quantize-int8

# 从 HLS profiling 报告构建 LUT 表
python3 run_build_lut.py --manifest configs/hardware/lut_manifest_example.yaml --output artifacts/fpga_lut.pkl
```

详细使用说明：[docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## 结果落盘

每次搜索都会在 `results/<run_name>/` 下生成完整运行目录，默认包含：

```text
results/<run_name>/
├── config.yaml
├── cli_args.json
├── run_info.json
├── logs/
│   └── console.log
├── results/
│   ├── baseline.json
│   ├── dataset_summary.json
│   ├── search_space_summary.json
│   ├── candidates.jsonl
│   ├── candidates.json
│   ├── candidates.csv
│   ├── pareto_front.json
│   ├── pareto_selection.json
│   ├── summary.json
│   └── candidates/
│       └── <arch_id>.json
└── checkpoints/
    ├── search_state.json
    ├── best_candidate.json
    ├── best_model.pt
    ├── final_best_model.pt   # retrain 后生成
    ├── controller_latest.pt   # RL 搜索时生成
    └── controller_best.pt     # RL 搜索时生成
```

---

## 项目文档

| 文档 | 说明 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 整体技术架构与模块边界 |
| [docs/method_design.md](docs/method_design.md) | 问题模型、方法设计与实现映射 |
| [docs/project_overview.md](docs/project_overview.md) | 项目概览与问题定义 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速开始与使用指南 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 实现进展与功能总结 |

---

## 项目结构

```text
.
├── README.md                 # 项目说明
├── run_search.py            # 搜索入口脚本
├── run_build_lut.py         # HLS report -> LUT 构建入口
├── configs/                  # 配置文件
│   └── search/
│       └── sonar_fpga_baseline.yaml
│   └── hardware/
│       ├── zynq7020.yaml
│       ├── kintex7_xc7k325.yaml
│       └── lut_manifest_example.yaml
├── docs/                     # 文档
│   ├── architecture.md
│   ├── method_design.md
│   ├── project_overview.md
│   ├── QUICKSTART.md
│   └── PROGRESS.md
├── reference/                # 参考代码库分析
│   ├── FBNet/
│   ├── HW-NAS-Bench/
│   ├── TinyTNAS/
│   └── ANALYSIS.md
└── src/hwnas_fpga/
    ├── interfaces.py         # 接口定义
    ├── search_space/         # 搜索空间
    │   └── space.py
    ├── hardware/             # 硬件估计
    │   ├── cost.py
    │   ├── lookup_table.py   # LUT查找表
    │   ├── lut_pipeline.py   # profiling manifest -> LUT
    │   └── report_parser.py
    ├── models/               # 模型构建
    │   └── builder.py
    ├── data/                 # 数据加载
    │   └── dataset.py
    ├── training/             # 训练与重训
    │   ├── trainer.py
    │   └── retrain.py
    ├── search/               # 搜索算法
    │   ├── searcher.py
    │   ├── constrained.py    # 约束搜索器
    │   └── pareto.py         # Pareto前沿
    └── deploy/               # ONNX / HLS 导出
        ├── export.py
        ├── quantization.py
        ├── hls.py
        ├── hls_backend.py
        ├── report_parser.py
        └── inference.py
```

---

## 核心功能

### ✅ 已实现

- **搜索空间**: stage-based 可搜索架构空间
- **硬件估计**: 分析模型 + LUT 查找表 + board profile
- **模型构建**: ArchitectureSpec → PyTorch Model
- **训练评估**: 完整的训练流程
- **搜索算法**: RL 搜索 + 约束剪枝 + 真正 Pareto 选优
- **重训练**: best architecture 独立最终重训
- **部署导出**: ONNX 导出 + HLS 项目骨架 + report parser
- **INT8量化**: 权重量化包导出，供 FPGA/HLS 后端使用
- **LUT profiling**: HLS report -> LUT table 构建链
- **对比实验**: Fused MBConv vs 标准 MBConv、有/无早期剪枝
- **Pareto优化**: 多目标优化与前沿分析

### ⏳ 待实现

- 权重共享超网训练
- 真实声呐数据加载 (MARIS/UATD)
- HLS/Vivado/Vitis 实际调用与板上回填

---

## LUT Profiling

真实 FPGA profiling 可以通过 `run_build_lut.py` 从 Vivado/Vitis HLS 报告构建查找表：

```bash
python3 run_build_lut.py \
  --manifest configs/hardware/lut_manifest_example.yaml \
  --output artifacts/fpga_lut.pkl \
  --summary-json artifacts/fpga_lut_summary.json
```

生成的 `fpga_lut.pkl` 可以直接接到搜索配置：

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
