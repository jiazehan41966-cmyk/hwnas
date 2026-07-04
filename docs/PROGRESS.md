# HW-NAS 项目进展

Last updated: 2026-07-03; latest consolidated evidence snapshot: 2026-06-22.

## 当前结论

项目已具备从搜索空间、随机/RL/Proxyless 搜索、独立重训练、LUT 估计、
HLS/Vivado route gate 到 COM5 固定输入板测的闭环。结论必须按证据层
报告，不能把搜索 proxy、重训练验证、route、COM5、图像质量或功耗混为
一类。2026-07-03 第一性原理重审进一步确认：现有分类指标属于旧
fold-0 验证协议，`denoise`/`edge` 的 PyTorch 与 HLS 实现不具备语义
一致性，不能据此声明训练算子已经部署。

## 已完成

| 能力 | 状态 | 主要入口 |
|---|---|---|
| 搜索空间与可行性探测 | 已实现 | `run_search_space_probe.py` |
| Random/RL/Proxyless 搜索 | 已实现 | `run_search.py`、`run_rl_search.py` |
| 独立重训练 | 已实现 | `run_retrain.py` |
| LUT/分析混合硬件估计 | 已实现 | `src/hwnas_fpga/hardware/` |
| ONNX/INT8/HLS stub 导出 | 已实现 | `run_export.py` |
| HLS/Vivado/full-route 工具链 | 已实现并形成 Phase0 证据 | `hls_lut_builder/` |
| COM5 固定 harness 输入板测 | 已实现并保存稳定性产物 | `hls_lut_builder/board_harness/` |
| PSNR/SSIM/MSE 图像质量 | 已实现 | `src/hwnas_fpga/metrics/`、`scripts/measure_sonar_image_quality.py` |
| v4 三线证据打包与状态识别 | 已实现 | `scripts/phase0_v4_three_lane_closure.py` |

## Phase0 v3 基线

Phase0 v3 low-DSP 路线已有 4 个 full-network AV7K325 COM5
board-claimable 候选。`rl_arch_186` 与 `rl_arch_242` 还具备独立
retrain150 权重重注入证据。

权威交接：`docs/PHASE0_V3_BOARD_RESULTS.md` 与
`docs/PHASE0_V3_RETRAINED_BOARD_REINJECTION.md`。

## Phase0 v4 声呐路线

2026-06-22 证据快照包含：

- 7 个 Pareto route-screen 候选；
- 6 个 route-clean 且具备五次稳定 COM5 测量的历史协议候选；
- 1 个 full-route fail 候选 `rl_arch_116`；
- 5 个候选完成 retrain150；
- NKSID val fold 0 的 520 张图完成 PSNR/SSIM/MSE 分析。

精确指标、路径与非结论边界见
`docs/PHASE0_V4_SONAR_RESULTS.md`。

## 尚未完成

| 项目 | 当前状态 | 完成门槛 |
|---|---|---|
| 四路声呐消融 | `no_sonar` 仅 `3/300`，其余未运行 | 四个 variant 均 `comparison_ready=true` |
| 完整板上验证集精度 | 未运行 | 有可追踪的 NKSID 样本级板上推理结果 |
| 实测功耗/能耗 | `not measured` | 外部功率计或可读监控 CSV 通过验收 |
| 独立 HLS/LUT 生产链审计 | 文档缺口 | 新增 `docs/audit_hls_lut_production.md` |
| acquisition-group-safe 外层测试协议 | 元数据缺失 | 获得 mission/sequence group 并完成外层测试 |
| PyTorch/HLS 算子语义一致性 | `denoise`/`edge` 未通过 | 匹配计算、权重导出和 fixed-point 数值对齐 |

## 报告红线

- `macro_f1`、`top1` 优先；`accuracy` 不自动等于 `top1`。
- NAS LUT `latency_ms` 是估计值，不是 COM5 实测延迟。
- COM5 只证明当前 bitstream 与固定 harness 输入下的延迟/输出一致性。
- fold-0 的 search/retrain 指标不是独立测试集泛化估计。
- 简化 HLS `denoise`/`edge` 的 route/COM5 不能证明训练期 PyTorch 算子已部署。
- `input_as_reference` 的 PSNR/SSIM 只用于算子影响/结构保持分析。
- 没有外部功率数据时，power/energy 必须写 `not measured`。
