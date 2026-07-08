# 声呐算子 G5 实验计划（折叠部署 + 四路消融 + 指标协议）

制定日期：2026-07-07。本计划是 `docs/MEASUREMENT_FIRST_REBUILD.md` G5 门禁的
执行方案，目标是让 `denoise` / `edge` 要么以完整证据链重返搜索空间，要么以
完整证据链被正式淘汰。所有分类数字必须走冻结协议
`nksid_outer5fold_inner_contiguous_v1`（`docs/EVAL_PROTOCOL.md`）。

## 已就绪的前置条件（2026-07-07 完成）

| 组件 | 位置 | 状态 |
|---|---|---|
| 推理期重参数化折叠 | `src/hwnas_fpga/deploy/reparam.py`（`fold_denoise_block` / `fold_edge_block` / `fold_sonar_blocks`） | 15 项 eval 等价测试通过 |
| 声呐指标 SNR/EPI/ENL/SSI | `src/hwnas_fpga/metrics/image_quality.py` | 8 项测试通过 |
| 测量脚本接入新指标 | `scripts/measure_sonar_image_quality.py` | paired 模式冒烟通过 |

折叠语义（G5 部署侧证据的基础）：

- `DenoiseBlock`（DW+BN 特征分支 ⊕ softmax 平滑分支 → ReLU → PW+BN）
  折叠为 **单个带 bias 的 depthwise k×k 卷积 → ReLU → 带 bias 的 1×1 卷积**，
  与 `dw_pw_conv` 部署形态相同。
- `EdgeAwareBlock`（4 方向 Sobel DW+BN → concat → 1×1 融合+BN）
  折叠为 **单个带 bias 的标准 k×k 稠密卷积**。
- 折叠仅在 eval 语义（BN running stats）下位级等价；INT8 量化必须
  作用于折叠后的权重（单一量化规范，见 E2）。

## 匹配对照的精确数字（已实测，stage-3 槽位：C_in=C_out=32、stride=1、28×28）

参数量/MACs 以**折叠后（部署形态）**计量——这正是 G5 门禁比较的语义对象；
训练期数字附注供参考。由 `fold_denoise_block`/`fold_edge_block` + PyTorch
参数计数直接测得：

| 块 | 折叠后参数 | 折叠后 MACs | 匹配对照 | 参数误差 | MACs 误差 |
|---|---:|---:|---|---:|---:|
| `denoise k3` | 1,376 | 1,028,608 | `mbconv k3 e1`（折叠后 1,376 / 1,028,608） | **0.0%** | **0.0%** |
| `edge k3` | 9,248 | 7,225,344 | `mbconv k3 e4`（折叠后 9,632 / 7,325,696） | **4.0%** | **1.4%** |

（训练期参数：denoise 1,728 / mbconv e1 1,440；edge 5,568 / mbconv e4 9,920。
差异来自 BN 与 softmax 平滑分支，折叠后消失。）

两组都满足 G5 的 `parameter_count_within_5pct` 与 `macs_within_5pct`。
`output_shape_equal`：四种块在该槽位输出均为 (N, 32, 28, 28)。

---

## E1：四路消融（G5 核心判据）

### 目的

在无泄漏协议下回答唯一问题：**在参数量/MACs 匹配的前提下，声呐先验块
相对 mbconv 是否带来 macro_f1 实际增益？**

### 固定骨干 `sonar_ablation_backbone_v1`

基于 `rl_arch_135` 编码，唯一改动：stage-3 深度从 1 改为 2，形成两个消融
槽位（A、B），使 `denoise_edge` 变体可以同时容纳两个算子，且四个变体
之间只有槽位算子不同（因子化设计）：

```text
input 1×224×224, stem conv3x3 s2 → 32ch @112
stage0: conv    k1 e1 s1, 16ch @112
stage1: mbconv  k3 e6 s2, 24ch @56
stage2: mbconv  k3 e3 s2, 32ch @28
stage3: 32ch s1 depth2 @28   ← 槽位 A + 槽位 B（本消融唯一变量）
head: GAP → FC(8)
```

### 四个变体（槽位配置）

| 变体 | 槽位 A | 槽位 B |
|---|---|---|
| `mbconv_control` | `mbconv k3 e1 s1` | `mbconv k3 e4 s1` |
| `denoise` | `denoise k3 e1 s1` | `mbconv k3 e4 s1` |
| `edge` | `mbconv k3 e1 s1` | `edge k3 e1 s1` |
| `denoise_edge` | `denoise k3 e1 s1` | `edge k3 e1 s1` |

槽位 A 的对照是 mbconv e1（与折叠 denoise 参数完全相同），槽位 B 的对照是
mbconv e4（与折叠 edge 误差 ≤4%），因此任意两个变体之间的容量差 ≤4%。

candidate JSON 放在 `configs/ablation/sonar_g5_v1/`（4 个文件，命名
`<variant>.candidate.json`），编码格式沿用
`hls_lut_builder/.../003_rl_arch_135.candidate.json` 的 `candidate.encoding`
结构。stage-3 段示例（`denoise_edge` 变体）：

```json
{
  "channels": 32, "depth": 2, "stride": 1,
  "blocks": [
    {"op": "denoise", "kernel_size": 3, "expand_ratio": 1, "stride": 1},
    {"op": "edge",    "kernel_size": 3, "expand_ratio": 1, "stride": 1}
  ]
}
```

其余 stage 四个变体逐字节相同。生成后必须先跑一次参数/MACs 校验
（PyTorch 计数 + 折叠计数），把实际数字写进 manifest 的
`matched_control` 字段。

### 训练与评估配置（全部沿用冻结协议默认值）

| 项 | 值 |
|---|---|
| 协议 | `nksid_outer5fold_inner_contiguous_v1`，外层验证集只消费一次 |
| 折 × 种子 | folds 0,1,2,3,4 × seeds 42,43,44 = 每变体 15 次运行 |
| epochs | 150 |
| 优化器 | AdamW，lr 1e-3，weight_decay 1e-4 |
| 调度 | cosine + 5 epoch warmup（min_lr_ratio 0.01） |
| label smoothing | 0.1 |
| 长尾 | logit adjustment tau=1.0 |
| batch | 8 × 梯度累积 4（等效 32，RTX 3050 Ti AMP） |
| best-epoch 选择 | 内层连续区块选择集（15%），禁止触碰外层验证集 |
| 图像 | 224×224 灰度单通道 |

命令（每变体一条，串行或按 GPU 空闲排队）：

```powershell
python run_eval_protocol.py `
  --candidate-path configs/ablation/sonar_g5_v1/mbconv_control.candidate.json `
  --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name g5_ablation_mbconv_control

python run_eval_protocol.py `
  --candidate-path configs/ablation/sonar_g5_v1/denoise.candidate.json `
  --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name g5_ablation_denoise

python run_eval_protocol.py `
  --candidate-path configs/ablation/sonar_g5_v1/edge.candidate.json `
  --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name g5_ablation_edge

python run_eval_protocol.py `
  --candidate-path configs/ablation/sonar_g5_v1/denoise_edge.candidate.json `
  --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name g5_ablation_denoise_edge
```

共 60 次运行（4 变体 × 15）。协议入口自带按 (fold, seed) 的 resume，
中断后重跑同命令即续。**排在 G1 基线三连完成之后**（同一块 GPU）。

### 统计检验（G5 `paired_stratified_bootstrap` + Holm）

- 单位：15 个 (fold, seed) 配对——每个变体与 control 在完全相同的
  (fold, seed) 上比较外层验证集预测。
- 方法：配对分层 bootstrap，按类别分层重采样外层验证集样本，
  迭代 **≥10,000 次**，统计 Δmacro_f1 的均值与 p 值（双侧）。
- 多重校正：`denoise`、`edge`、`denoise_edge` 三个对照的 p 值做
  Holm 校正（`hwnas_fpga.hardware.sonar_operator_gate.holm_adjust` 已实现）。
- 判据（与门禁一致）：Holm 校正后 p < 0.05 **且** Δmacro_f1 均值 > 0
  才算"实际增益"。
- 交付脚本：`scripts/compare_sonar_ablation_bootstrap.py`（待写），
  输入 4 个 `results/protocol/g5_ablation_*/protocol_summary.json` 及
  逐样本预测记录，输出 manifest 的 `comparisons_vs_control` 段。

### 产物与门禁回填

1. `results/protocol/g5_ablation_<variant>/protocol_summary.{json,md}` × 4；
2. manifest（按 `artifacts/sonar_operator_gate/manifest.template.json`）
   填 `ablation_variants`（folds/seeds/completed_runs=15/claimable/outer_leakage=false）、
   `comparisons_vs_control`、`operators.*.matched_control`；
3. `python scripts/audit_sonar_operator_gate.py` 重新出具
   `artifacts/sonar_operator_gate/sonar_operator_gate.json`。

### 判读规则

- 任一算子未过 Holm 或 Δ≤0 → 该算子保持 PAUSED，转 E4 设计迭代或正式淘汰；
- |Δmean| 小于配对 std → 结论写"无可分辨差异"，不得写"有增益趋势"；
- 消融结果只回答识别效果；部署可行性由 E2 独立回答，两者都过才 ADMITTED。

---

## E2：折叠导出 → INT8 → HLS 位精确 parity（G5 部署侧证据）

### 目的

用折叠后的标准卷积形态打通 `single_quantization_spec` /
`weight_export_complete` / 三类张量 parity / `bit_exact_zero_mismatch` /
`hls_evidence_complete` / `hls_feasible` 六个部署侧子门。**不再为
denoise/edge 编写专用 HLS 分支模板**——这是折叠方案的全部意义。

### 步骤与配置

1. **权重来源**：E1 的 `denoise` / `edge` 变体各取一个已完成 checkpoint
   （约定 fold1/seed42，与 G4 首链对象对齐）。E1 未完成前可先用
   30 epoch 冒烟权重打通流程，但 manifest 只认正式 checkpoint。
2. **折叠导出**（交付脚本 `scripts/export_folded_sonar_weights.py`，待写）：
   `model.eval()` → `fold_sonar_blocks(model)` → 保存折叠后 state_dict 与
   逐层规格 JSON；记录 `weight_export_sha256`（对导出包做 SHA256）。
3. **INT8 量化**：契约固定为 `per_tensor_symmetric_int8_v1`
   （`src/hwnas_fpga/deploy/quantization.py`）。**只量化折叠后的权重**；
   软件侧与 HLS 侧使用同一份量化规格文件，
   `software_spec_sha256 == hls_spec_sha256` 必须成立。
4. **整数参考模拟**：`src/hwnas_fpga/deploy/fixed_point.py` 逐层整数模拟，
   生成逐层输出记录。
5. **HLS 侧**：折叠 denoise 走现有 depthwise+pointwise conv 模板；折叠
   edge 走标准稠密 conv k3 模板（`hls_lut_builder` 现有算子库均已覆盖，
   不新增模板）。跑 csynth + OOC/route，产出
   `hls.evidence_complete=true`、`hls.route_feasible=true`
   （`src/hwnas_fpga/hardware/hls_evidence.py` 口径）。
6. **parity 记录**（JSONL），三类输入缺一不可，逐元素比较：
   - 真实样本：NKSID fold1 外层验证集图像 ≥ 32 张；
   - 边界张量：全 0、全 127、全 -128、±1 LSB 棋盘格 ≥ 8 个；
   - 随机张量：固定种子 0..127 共 128 个 INT8 随机输入；
   - 判据：`compared_element_count > 0` 且 `mismatch_count == 0`（零差异）。
7. **审计**：

```powershell
python scripts/audit_int8_hls_parity.py --records <parity_records.jsonl>
```

结果回填 manifest 的 `operators.denoise` / `operators.edge` 段。

### 风险与边界

- 折叠等价只保证 FP32 eval 语义；INT8 下"分支先量化再相加"与
  "折叠后量化"不同——契约规定**折叠后量化**，HLS 与软件模拟必须同源。
- 历史简化 HLS 模板（`denoise_serial_lowdsp_stage3_k3.cpp.tmpl` 等）
  作废，不得再作为任何证据来源。

---

## E3：声呐图像质量测量（新指标协议）

### E3a：input_as_reference 复测（算子效应，非复原质量）

对 5 个外层折各跑一次（此前只有 fold0），使用已接入的
PSNR/SSIM/MSE/SNR/EPI/SSI/ENL：

```powershell
foreach ($fold in 0..4) {
  python scripts/measure_sonar_image_quality.py `
    --data-dir data/NKSID --split val --fold $fold --image-size 224 `
    --transforms identity,denoise,edge,edge_enhanced `
    --output-dir results/sonar_image_quality_v2_fold$fold
}
```

重点读数：`denoise` 变换的 **EPI**（高斯平滑的边缘保持短板，量化它）
与 **SSI/ENL**（斑点抑制是否真的发生）。边界声明沿用脚本输出：
input_as_reference 不证明复原质量，不得与 macro_f1 合并叙述。

### E3b：合成斑点配对协议（让 PSNR/MSE/SNR 语义成立）

交付脚本 `scripts/make_synthetic_speckle_pairs.py`（待写），规格：

- 干净参考：NKSID fold0 外层验证集 520 张原图（作为相对参考，边界照注）；
- 噪声模型：乘性 speckle `I_noisy = clip(I_clean × S_L, 0, 1)`，
  `S_L ~ Gamma(shape=L, scale=1/L)`（均值 1），视数 L ∈ {1, 2, 4} 三档；
- 随机种子：全局 20260707，逐图种子 = 全局种子 + 样本索引；
- 目录结构：`data/synthetic_speckle/L{1,2,4}/{ref,noisy}/<class>/<name>.png`。

测量（以 L=1 为例，参考=干净、候选=去噪算子输出）：

```powershell
python scripts/measure_sonar_image_quality.py `
  --reference-dir data/synthetic_speckle/L1/ref `
  --candidate-dir data/synthetic_speckle/L1/denoised `
  --image-size 224 --output-dir results/sonar_iq_paired_L1
```

判读表（去噪算子在配对协议下的合格线，作为 E4 设计迭代的量化标尺）：

| 指标 | 期望方向 | 合格线（相对 noisy 基线） |
|---|---|---|
| PSNR / SNR / MSE | ↑ / ↑ / ↓ | 优于 noisy vs ref 基线 |
| SSIM | ↑ | 优于基线 |
| EPI | ↑ | ≥ 0.85 且不低于基线（防止过度平滑） |
| SSI | ↓ | < 1 |
| ENL（同质区） | ↑ | 高于 noisy |

E3 全程 CPU 可跑，不占训练 GPU，可立即启动。

---

## E4：条件触发的设计迭代（E1 出结果后才启动）

| 编号 | 触发条件 | 实验 | 配置要点 |
|---|---|---|---|
| E4a 浅层注入 | E1 中 denoise Δ≤0 | denoise 移出竞争槽位，作为 stem 后固定预处理层 | 骨干同 v1 但 stage3 恢复双 mbconv 对照；新增 `stem_denoise=true` 变体；同协议 5 折×3 seed 对比 |
| E4b log 域 | E3b 显示乘性噪声假设成立（L=1 档 SSI 改善明显） | 输入变换 `x → log(1+255x)/log(256)`（部署为一张 256 项查表） | 在 dataset 归一化层加开关；与 E4a 正交，可 2×2 因子设计 |
| E4c edge 瘦身 | E1 中 edge Δ>0 但未过 Holm，或 LUT 预算触顶 | `edge_v2`：2 方向（Gx/Gy）+ 幅值近似 `|gx|+|gy|`，融合输入 2C | 折叠后参数约为 v1 的 50%；匹配对照改为 mbconv e2（需重算 5% 匹配） |

每个 E4 实验的运行量与 E1 单变体相同（15 runs），统计方法同 E1。

---

## 执行顺序与资源

```text
现在可启动（不占训练 GPU）：
  E3a 五折 input_as_reference 复测        （CPU，~小时级）
  E3b 合成斑点配对脚本 + 测量             （CPU，~小时级）
  E2 步骤 1-4 冒烟（30ep 权重打通折叠→INT8→整数模拟）（GPU 分钟级/CPU）

G1 基线三连完成后：
  E1 四路消融 60 runs × 150ep             （GPU，数天级，resume 安全）
  E1 完成 → 统计脚本 → manifest → G5 审计
  E2 用 E1 正式 checkpoint 重出 parity + HLS 证据
  E1+E2 齐 → G5 门禁重审 → ADMITTED / 淘汰
  条件满足 → E4
```

待写交付物清单：

1. `configs/ablation/sonar_g5_v1/*.candidate.json`（4 个变体）；
2. `scripts/compare_sonar_ablation_bootstrap.py`（配对分层 bootstrap + Holm）；
3. `scripts/export_folded_sonar_weights.py`（折叠导出 + SHA256）；
4. `scripts/make_synthetic_speckle_pairs.py`（合成斑点配对数据）；
5. E4c 触发时：`edge_v2` 块实现 + 折叠函数 + 匹配重算。

## 声明边界

- E1 数字是软件验证集分类证据；E2 是部署语义证据；E3 是图像质量证据。
  三类证据分开报告，不得合并成单一"声呐效果"结论。
- 在 G5 重审为 PASS 之前，`denoise`/`edge` 不得进入任何新的正式搜索。
