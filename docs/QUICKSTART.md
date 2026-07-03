# HW-NAS 快速开始

Last verified: 2026-07-03

本指南只覆盖当前维护的入口。历史配置与脚本保留在 `legacy/` 路径中，
不应作为新实验默认入口。

## 1. 运行边界

开始实验前先区分证据层：

- search proxy：搜索阶段的 `macro_f1`、`top1` 与硬件估计；
- retrain：独立重训练后的 PyTorch 验证集指标；
- route：Vivado full-route 的 WNS 与实际 LUT/DSP/BRAM；
- COM5：固定 harness 输入的板上延迟/输出一致性；
- power：只有外部功率计或可读监控路径的采样结果才算实测功耗。

这些层不能互相替代。COM5 不是完整 NKSID 板上验证集精度，也不是功耗
测量路径。

## 2. 环境与最小检查

项目在 Windows/PowerShell 下维护。优先使用项目已有虚拟环境；如果该
环境未安装测试依赖，可用已配置好依赖的 Python 运行测试。

```powershell
python --version
python -m pip install -e ".[dev]"
python -m pytest -q
```

根目录 `pyproject.toml` 将 pytest 收敛到本项目 `tests/`，不会再递归
收集 `reference/` 下游仓库测试。

生成文件应写入 `results/`、`outputs/` 或 `logs/`，不要提交训练输出、
checkpoint、Vivado/HLS 临时目录或本地数据集。

正式搜索前建议先执行数据协议审计：

```powershell
python scripts/audit_nksid_protocol.py `
  --data-dir data/NKSID `
  --fold 0 `
  --neighbor-radius 1 `
  --hash-files `
  --output-dir results/first_principles_audit_20260703
```

NKSID 图像读取默认 `image_error_policy=raise`。坏图会立即终止运行，避免
用空白图保留原标签并污染指标。仅为复现旧行为时才显式配置
`image_error_policy: blank`。

## 3. 搜索空间探测

探测只采样并检查可行性，不训练模型：

```powershell
python run_search_space_probe.py `
  --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml `
  --num-samples 200
```

输出中的 `feasible` 只代表当前估计器/约束下可行，不代表 route-clean
或 board-claimable。

## 4. 搜索与重训练

通用 MobileNetV2 anchor 搜索：

```powershell
python run_search.py `
  --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml
```

该默认配置显式加载
`hls_lut_builder/configs/operator_manifest_semantic_safe.yaml`。在
PyTorch/HLS 数值语义对齐前，`denoise` 和 `edge` 不进入新的可声明搜索。

快速连通性检查：

```powershell
python run_search.py `
  --search-method rl `
  --episodes 3 `
  --train-epochs 1 `
  --batch-size 8
```

对搜索运行目录中的最优架构做独立重训练：

```powershell
python run_retrain.py --run-dir results/<search_run_name>
```

每次正式实验至少保留 config、seed、dataset/fold、命令、checkpoint 和
结果目录。论文报告优先使用 `macro_f1`（宏平均 F1）与 `top1`，不要在
未核对代码路径时把 `accuracy` 直接等同于 `top1`。

## 5. 推理与导出

```powershell
python run_infer.py `
  --checkpoint results/<run_name>/checkpoints/final_best_model.pt `
  --input <image_or_dir>

python run_export.py `
  --checkpoint results/<run_name>/checkpoints/final_best_model.pt `
  --prepare-hls
```

`--prepare-hls` 生成部署准备骨架，不等同于完整 HLS/Vivado/full-route
证据。正式板级结论应引用 route/COM5 产物。

## 6. 声呐图像质量

NKSID 没有成对 clean/noisy 参考图。默认 dataset mode 使用
`input_as_reference`，仅分析算子对结构的影响：

```powershell
python scripts/measure_sonar_image_quality.py `
  --data-dir data/NKSID `
  --split val `
  --fold 0 `
  --image-size 224 `
  --transforms identity,denoise,edge,edge_enhanced `
  --output-dir results/sonar_image_quality_psnr_ssim_20260622
```

结果包括 per-image、by-class、overall CSV，以及 JSON/Markdown 摘要。
PSNR/SSIM 与分类 `macro_f1`/`top1` 必须分开报告。

如果以后具备真实成对参考图，可使用：

```powershell
python scripts/measure_sonar_image_quality.py `
  --reference-dir <reference_dir> `
  --candidate-dir <candidate_dir> `
  --output-dir results/sonar_image_quality_paired
```

## 7. Phase0 v4 证据打包

默认命令只读取已有 search/retrain/route/COM5/checkpoint 产物并刷新表格，
不会启动训练、Vivado 或 COM5：

```powershell
python scripts/phase0_v4_three_lane_closure.py
```

只有明确需要长时任务时才使用 `--run-retrain` 或 `--run-hardware`。
当前四路声呐消融尚未完成，必须检查输出字段
`comparison_ready=true` 后才能形成正式消融结论。

独立刷新 v4/v3 板级比较：

```powershell
python scripts/build_phase0_v4_vs_v3_board_comparison.py
```

## 8. 结果与交接文档

- 当前仓库与审计入口：`docs/PROJECT_MEMORY.md`
- 当前审查入口：`docs/REVIEW.md`
- 第一性原理重审：`docs/FIRST_PRINCIPLES_AUDIT_20260703.md`
- 可提交的精简证据：`artifacts/first_principles_audit_20260703/`
- Phase0 v3 板级基线：`docs/PHASE0_V3_BOARD_RESULTS.md`
- Phase0 v4 声呐结果：`docs/PHASE0_V4_SONAR_RESULTS.md`
- 仓库结构：`docs/REPO_LAYOUT.md`
- 配置主线：`docs/SEARCH_CONFIG_CANONICAL.md`

本地结果目录默认不进入 Git。发布前应提交代码、配置、测试和可复现说明，
并在文档中保留明确的证据路径与非结论边界。
