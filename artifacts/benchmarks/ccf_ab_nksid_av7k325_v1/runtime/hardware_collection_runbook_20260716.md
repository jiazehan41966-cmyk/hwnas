# AV7K325 HLS、板级延迟与外部功耗采集 runbook

本 runbook 把 R6/R7 的外部执行要求落实为可执行顺序和固定字段。当前机器已经识别并通过 Vivado/Vitis HLS 2023.2、正式器件数据库、JTAG 枚举以及最小 csynth/route 工具链 smoke；这些结果不等于完整网络、bitstream、板级推理或外部功耗实测。历史 ZCU102、Vivado 估计功耗或仅有 COM5 的事实均不得替代本项目实测。

## 0. 预检

只读预检：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/hardware_preflight_20260716.ps1
```

用已发现的显式工具路径和仪器采集命令做严格预检：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/hardware_preflight_20260716.ps1 `
  -VivadoPath "F:\vivado\Vivado\2023.2\bin\vivado.bat" `
  -HlsPath "F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat" `
  -SerialPort COM5 `
  -InstrumentCommand "C:\path\to\meter_capture.exe" `
  -ProbeVersions `
  -RequireReady
```

The 2026-07-17 discovery pass found the matched toolchain at the explicit paths above. The following read-only checks also pass:

```powershell
F:\vivado\Vivado\2023.2\bin\vivado.bat -mode batch -nolog -nojournal -notrace `
  -source artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/query_av7k325_parts_20260717.txt

# Start hw_server with an idle timeout, then execute the target query.
F:\vivado\Vivado\2023.2\bin\vivado.bat -mode batch -nolog -nojournal -notrace `
  -source artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/query_av7k325_hw_target_20260717.txt
```

Observed target: `localhost:3121/xilinx_tcf/Digilent/210512180081`; observed device: `xc7k325t_0`. These commands do not program a bitstream and cannot be cited as inference, latency, route, or power evidence.

The local 2023.2 HLS and route-license smokes are archived at `results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/hardware_toolchain_20260717/smoke_manifest.json`. Their nine SHA256 bindings pass, but their formal HLS/route sample-count increment is exactly zero because both inputs are trivial toolchain probes rather than complete networks.

四项必须同时为真：Vivado 可调用、HLS 可调用、COM5 状态 OK、外部仪器采集程序可调用。COM5 仅证明 UART bridge（串口桥）存在。

## 1. 冻结候选池与 ≥100 样本

1. 等待正式 NAS/retrain 证据后冻结候选池；保存候选 JSON、checkpoint、生成 HLS 源、配置与各自 SHA256。
2. 按 architecture family（架构家族）分层选取至少 100 个语义安全完整网络；不得用 operator-only 样本补足分母。
3. 每个候选无论成功或失败都写一行 `hls_route_sample_template.csv`。只有显式 `network_scope=COMPLETE_NETWORK`、绑定完整网络 top function、语义等价报告、候选池 manifest、source freeze 和 project code state 的行才可能进入完整样本计数。综合失败、route 失败和超时必须保留 failure stage/category。
4. HLS 与 route 使用同一器件、时钟策略、Vivado/HLS 版本和脚本 SHA。样本少于 30 只描述；30–99 标记 exploratory；达到 100 后才允许 grouped 5-fold（按架构家族分组五折）正式比较。
5. T6 的方法固定为 analytic/LUT estimator、linear regression、gradient boosting、HARP-GNN；指标固定为 MAE、RMSE、sMAPE、Spearman rho、Kendall tau、top-k recall、false-feasible rate、false-infeasible rate及校准图。

采集过程中随时运行只读 inventory audit：

```powershell
D:\software\python\python.exe artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/audit_hardware_collection.txt `
  --hls-route-csv <hls_route_samples.csv> `
  --board-latency-csv <board_latency_samples.csv> `
  --output <hardware_collection_audit.json>
```

审计器分别报告 total、semantic-safe、full-network、complete HLS/route、failure rows 和 architecture-family 数；只有 complete≥100、family≥5 且 schema/provenance 无错误时才给出 `grouped_5fold_ready=true`。合法的失败行只要求截至失败阶段实际存在的报告：csynth 失败不得伪造 csynth/route 报告，route 失败必须绑定已通过的 csynth 报告但不强求不存在的 route 报告。失败行必须保留，但不用于冒充完整回归目标；缺失 failure stage/category 仍是 schema 错误。

当 inventory 达到 `grouped_5fold_ready=true` 后，按 `hls_proxy_fold_manifest_template.json.txt` 冻结五个 leave-one-architecture-family-out fold。每个完整 truth sample 对 analytic/LUT、linear regression、gradient boosting 和 HARP-GNN 的 13 个目标各保留且只保留一条 held-out prediction，即至少 `complete_rows × 4 × 13` 行 `hls_proxy_prediction_template.csv`。不得用训练折预测、作者论文原始数字、operator-only 行或随机 row split 补齐。

代理预测审计命令：

```powershell
D:\software\python\python.exe artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/audit_hls_proxy_predictions.txt `
  --hls-route-csv <hls_route_samples.csv> `
  --prediction-csv <hls_proxy_predictions.csv> `
  --fold-manifest <hls_proxy_fold_manifest.json> `
  --feasibility-config <hls_proxy_feasibility.json> `
  --output <hls_proxy_audit.json>
```

审计器逐行重验 truth、fold、feasibility、predictor artifact/config 和 source/code 哈希；重新计算 measured value、fold membership、predicted/measured feasibility、MAE、RMSE、sMAPE、Spearman rho、Kendall tau-b 与 top-10% recall。AV7K325 物理容量和正式 route 判定分开保存：物理 DSP 为 840，正式可声明阈值仍为 DSP≤700、WNS≥0、achieved clock≥200 MHz。`formal_metrics_ready=true` 只表示 grouped proxy 证据合同闭合，不表示板级 latency 或功耗已测。

HARP 输入必须来自候选生成的 HLS C/C++ 经 LLVM 13 形成的层级程序图；禁止把 NAS architecture JSON 直接伪装成 HARP 图。

## 2. 冻结三类部署候选

仅从 route-feasible 且来源可追溯的候选中冻结：

- `accuracy_first`：macro-F1 最高；
- `knee_point`：归一化 accuracy/latency/resource Pareto 前沿的 knee；
- `resource_min`：macro-F1 距最佳不超过 0.01 时 LUT 最低。

在候选 manifest 中记录选择输入表 SHA、算法/脚本 SHA、候选 JSON/checkpoint/INT8/HLS/bitstream SHA；冻结后不得因板测结果不好而换候选。

## 3. HLS、route 与 bitstream

每个候选依次执行：语义等价检查 → ONNX/INT8 导出 → fixed-point/HLS parity → HLS csynth → Vivado synth/place/route → bitstream。进入 COM5 前必须满足：

- parity summary `overall_pass=true`；
- route status clean；WNS/TNS、实际时钟、LUT/DSP/BRAM/FF 已归档；
- bitstream SHA256 与候选 manifest 一致；
- 工具版本、命令、日志、报告和生成源 SHA 均存在。

Vivado routed power 只存为 `estimate/proxy`，不得进入 T8 的 measured power 列。

## 4. COM5 板级准确性与延迟

先用 `run_dynamic_validation.py --dry-run` 验证 manifest、payload、bitstream 与 parity 输入，再执行完整验证集。保存原始逐样本 JSONL；另按 `board_latency_sample_template.csv` 导出每次推理记录。每候选报告：

- 验证集 macro-F1、top1、per-class F1、混淆矩阵；
- latency p50/p95/p99、FPS、错误率；
- CRC、status、numeric mismatch、缺失/重复样本计数；
- bitstream、payload、parity、clock 和工具链绑定。

板级 CSV 必须为三个角色各绑定且只绑定一个不同候选，并使用相同的 `sample_id/target` 验证集合；每角色至少保留 1,000 条逐推理延迟记录。每行同时绑定候选选择 manifest、候选 manifest、checkpoint、source freeze、project commit/code state、data/split/validation manifest、route report、bitstream、payload 和 parity summary。审计器重新计算所有被引用文件的 SHA，检查 host timestamps 与 `latency_ms`、target/prediction 与 `correct`、角色内固定 provenance 以及跨角色配对样本集合。错误推理可以保留并进入错误率，但 provenance/schema 错误不得进入 T7。

不能用单一 deterministic harness 输入的 argmax/latency 代表完整 NKSID 板级准确率。

## 5. 外部功率仪器

三个候选必须使用同一仪器、校准状态、采样率、board-input-total rail、接线、量程和采集命令。每个候选：

1. 采集至少 3 个 idle block，每块 ≥60 s；
2. 先完成 UART upload，再调用 `run_power_repeat.py --repeat-count >=1000`；
3. 对齐 receipt 中的 active UTC 区间，采集至少 3 个 active block，每块 ≥60 s；
4. 原始 CSV 使用 `timestamp_s,power_w` 或 `timestamp_s,voltage_v,current_a`，时间严格递增；正式 CSV 的 `timestamp_s` 必须是 UTC Unix epoch seconds，并在 v2 容差内与 receipt 的 `active_started_utc/active_finished_utc` 对齐，不能直接积分包含前后空闲段的未裁剪窗口；
5. 从 `power_measurement_protocol_template.json.txt` 生成同一份冻结协议文件，绑定仪器序列号、校准证书、采样率、量程、接线与 block 门槛；其 SHA 同时作为三候选 `measurement_protocol_fingerprint`；
6. 填写 `power_measurement_manifest_template.json.txt`，为每份 raw CSV 和 receipt 填写 SHA，并绑定 selection/candidate/route/bitstream/payload/parity/source-freeze/code/data/split；先逐候选运行 `scripts/import_power_measurement.py`；
7. 三份候选基础 manifest 全部 PASS 后，运行 `runtime/audit_power_campaign_v2.txt` 检查三个固定角色、所有文件哈希、同仪器/同协议及 UTC 对齐。当前 v2 是不扰动 G1 指纹的 `.txt` 证据工具；正式 campaign 前必须在新 source freeze 下迁移为受测试的 canonical source，然后再运行 `scripts/audit_power_campaign.py` 与新版审计。

v2 三候选检查命令：

```powershell
D:\software\python\python.exe artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/audit_power_campaign_v2.txt `
  <accuracy_first_power_manifest.json> `
  <knee_point_power_manifest.json> `
  <resource_min_power_manifest.json> `
  --output <power_campaign_v2_audit.json>
```

主指标为 `dynamic_energy_mj_per_inference`；次指标为 idle/active/dynamic W、FPS/W、板温。只有三候选同协议 campaign audit PASS 后，能耗才可进入 F12 Pareto 图。

## 6. 固定归档位置与验收

原始输出放入：

```text
results/benchmarks/ccf_ab_nksid_av7k325_v1/formal/hardware/<campaign_run_id>/
```

正式派生表图放入：

```text
artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/
```

一键验收顺序：HLS/route inventory audit → grouped proxy audit → parity audit → board dynamic-validation audit → per-candidate power import → three-candidate power campaign audit → benchmark readiness → measurement-first ledger。任一环节失败时，T6/T7/T8、F4/F9–F12 保持 `PENDING`，功耗保持 `NOT_MEASURED`。

## 7. 模板

- `hls_route_sample_template.csv`：HLS/route 样本级字段；包含 paper/method、候选池、完整网络范围/top、语义等价报告、source freeze、project commit/code state、工具链和阶段报告哈希绑定。
- `hls_proxy_prediction_template.csv`：每个 held-out sample × method × target 的预测—实测长表，绑定 predictor artifact/config、fold、feasibility、truth 与 source/code provenance。
- `hls_proxy_fold_manifest_template.json.txt`：五个架构族 leave-one-family-out 的 train/test sample 冻结清单；每个完整样本必须恰好作为一次 held-out test sample。
- `hls_proxy_feasibility_template.json.txt`：AV7K325 物理容量与正式 route Gate 的分离配置；DSP 840 是物理容量，DSP≤700 才是本项目正式部署判定。
- `audit_hls_proxy_predictions.txt`：完整 truth admission、逐键完备性、分组泄漏、预测器/配置哈希、实测值与可行性重算以及 52 组代理指标的 fail-closed 审计器。
- `board_latency_sample_template.csv`：板级逐推理延迟/正确性字段；每角色固定一个候选、至少 1,000 行、跨角色相同样本标签映射，并绑定 selection/candidate/validation/route/bitstream/payload/parity 与代码/数据哈希。
- `power_timeseries_template.csv`：外部功率仪器原始时序字段。
- `power_measurement_protocol_template.json.txt`：三个候选共享的仪器、接线、UTC 时间基准和 block 门槛协议。
- `power_measurement_manifest_template.json.txt`：与现有功耗审计器兼容并增加 raw/receipt SHA、UTC 对齐和全链路 provenance 的候选 manifest。
- `audit_power_campaign_v2.txt`：三角色、同协议/同仪器、文件绑定、采样率和 active UTC 区间的 fail-closed 补充审计器。
- `audit_hardware_collection.txt`：样本完整性、分层门槛与板级逐推理 schema 的 fail-closed 审计器。

模板中的占位符必须替换为实值；模板本身永远不是测量证据。

## 8. Frozen T6 structural DOE added on 2026-07-17

- Manifest: `../manifests/t6_structural_candidate_pool_v1.json.txt`, SHA256 `046e1ecba4e73306beb46e381070b5c251ef9d59b08b8045c635b1a905d48f65`.
- Independent audit: `../manifests/t6_structural_candidate_pool_v1_audit.json.txt`, SHA256 `f93b129a6249e6e4c75496b26dbeccf8b10cf41ea6605624029674cbe93ec43d`, status `PASS`.
- The pool contains 100 unique encodings, five families with 20 candidates each, and five fixed 80/20 leave-one-family-out folds.
- The source is structural only and supplies no historical accuracy, HLS, route or board truth.
- Every selected row is pending complete-network export, semantic equivalence and HLS/route collection; formal T6 increment remains zero.
- Execute one source-linked pilot from each family first. Only after all five pilots pass should the remaining 95 candidates be queued; retain failed attempts with their stage and category.

## 9. Frozen five-family export pilots

- Pilot manifest: `../manifests/t6_five_family_pilot_manifest_v1.json.txt`, SHA256 `ae9aad20b0c10f5958a530f2860272d5455543970de19ba91e261b08efa7c17c`.
- Independent audit: `../manifests/t6_five_family_pilot_manifest_v1_audit.json.txt`, SHA256 `88010ac17644747aa9c7dc92ec1ddf1d4fd9a24468050a60def607dd88991117`, status `PASS`.
- Selection is deterministic within each frozen family: require the legacy analytic flag to be true, minimize complete-network block count, then break ties with the frozen salted selection key. The analytic flag is an ordering hint only and is not hardware truth.
- Frozen pilots are `probe_0000`, `probe_0060`, `probe_0048`, `probe_0136` and `probe_0188`, covering the five declared families exactly once.
- The contract regression passes the valid input and rejects a premature board requirement and duplicated family/encoding. These fixtures add zero T6 rows.
- The current stage is export plus semantic equivalence, so a physical board is not required. Board entry remains bitstream programming and COM5 dynamic validation after route-feasible selection.
- Planning-only probes of the existing isolated full-network harness confirm `not_generated_mapping_incomplete` for all five pilots. The audit `../manifests/t6_five_family_mapping_gap_audit_v1.json.txt` records at least six missing component rows per pilot, no candidate-HLS mapping and no eligible arch-84 bitstream reuse. Repair candidate-specific HLS mapping and semantic equivalence under the next source freeze before any generate/build command.
