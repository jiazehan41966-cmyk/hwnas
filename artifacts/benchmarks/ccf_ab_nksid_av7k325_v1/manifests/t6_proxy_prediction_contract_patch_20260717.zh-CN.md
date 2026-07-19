# T6 grouped-proxy 预测合同补丁（中文伴随档案，2026-07-17）

- 英文原件：`t6_proxy_prediction_contract_patch_20260717.md`；SHA256：`c3050e41c0c3c2264f2b885ad2ab36f135aa441934fc477ad69b3b33afeaf67a`。

## 范围与冻结接口

本补丁只闭合 T6 缺失的预测侧证据接口，不创建任何真实 HLS/route 结果。可执行补充使用 source freeze 之外的 `.txt` runtime-evidence 约定。

- 四方法：`analytic_lut`、`linear_regression`、`gradient_boosting`、`harp_gnn`。
- 13 个 target：HLS cycles/II/LUT/DSP/BRAM/FF；route WNS/TNS/achieved clock/LUT/DSP/BRAM/FF。
- Split：5-fold leave-one-architecture-family-out；固定精确 train/test sample list，每族恰好 held out 一次。
- Prediction key：每个完整真值样本，对每个 `(sample_id, method_id, target_name)` 恰好一行。
- Provenance：predictor artifact/config、fold manifest、feasibility config、truth CSV、项目 commit/code state、source-freeze SHA 全部绑定。
- HARP ownership 显式为 `paper_id=harp_2023`；项目 baseline 用 `paper_id=project_internal`。作者论文数值不能作为本地 prediction。

## Feasibility 规则

AV7K325 物理容量：LUT 203800、FF 407600、BRAM 445、DSP 840。正式 route feasibility：LUT≤203800、FF≤407600、BRAM≤445、DSP≤700、WNS≥0、achieved clock≥200 MHz。审计器分别从 measured 和 predicted 数值重算 feasibility，不信任提交的 boolean label。

## 合成合同测试

100 个完整 truth row、5 个架构族、5,200 个 held-out prediction（100×4×13）。正例生成 52 个 method-target 指标行与 4 个方法级 feasibility summary；以下四类注入错误均被拒绝：duplicate prediction key、missing key、measured value 与 truth 不一致、predicted feasibility label 与重算限制不一致。

保留 summary：`contract_tests/hls_proxy_v1/contract_test_summary.json`，SHA256 `b314cbd876715999cf66c862cf433e4b273a57487931fd40f101af2d32f17661`。

## Runtime artifact SHA256

| Artifact | SHA256 |
|---|---|
| `hls_proxy_prediction_template.csv` | `b9974f6307dcc3af5dfb0aee965a503865a9795f28669e0df1b019e3c1916bac` |
| `hls_proxy_fold_manifest_template.json.txt` | `6cbbeb2c8fa5a09d1e15b2643c24efc837ad20b2ff846bd64dcc4814b9dfbc16` |
| `hls_proxy_feasibility_template.json.txt` | `ad56219a153183a9e1b9b2c0f6edf3c8d5050125aedec6a60d285a79f364e0b5` |
| `audit_hls_proxy_predictions.txt` | `768b8f171f0447e2ea1e25714d069873ef8deda522661aea65b1977087900e56` |
| `test_hls_proxy_contract.txt` | `6cde088dbd4702ef73136ad6eb5a47aeec8997ede04dd18f8eba01ff451d6603` |

## 边界

合同测试使真实 T6 denominator 增量严格为 0。至少 100 个 semantic-safe 完整网络项目真值行及真实 held-out prediction 通过相同审计前，T6/F4 PENDING。合同 PASS 不证明 route-feasible deployment、COM5 latency、板级 accuracy 或外部仪器 power。
