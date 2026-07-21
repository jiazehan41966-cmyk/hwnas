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

### 统计检验（G5 分层配对 bootstrap + fold-cluster sign-flip + Holm）

- 单位：15 个 (fold, seed) 配对——每个变体与 control 在完全相同的
  (fold, seed) 上比较外层验证集预测。
- 方法：以 outer fold 为 cluster 的层级配对 bootstrap；每次先对 5 个
  outer fold 有放回抽样，再在抽中的 fold 内对 3 个 seed 运行有放回抽样，
  迭代 **≥10,000 次**，统计 15 个配对运行的 Δmacro_f1 均值与 95% CI。
  p 值使用 fold-cluster paired sign-flip null，避免把同一 fold 内的 seed
  当作独立样本。
- 多重校正：`denoise`、`edge`、`denoise_edge` 三个对照的 p 值做
  Holm 校正（`hwnas_fpga.hardware.sonar_operator_gate.holm_adjust` 已实现）。
- 判据（与门禁一致）：Holm 校正后 p < 0.05 **且** Δmacro_f1 均值至少为
  `0.01`（`min_meaningful_delta`）才算"实际增益"；四个变体还必须共享
  同一 `protocol_context_sha256`，并各自完整包含 5 folds × 3 seeds。
- 交付脚本：`scripts/compare_sonar_ablation_bootstrap.py`，
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
2. **折叠导出**（交付脚本 `scripts/export_folded_sonar_weights.py`）：
   `model.eval()` → `fold_sonar_blocks(model)` → 保存折叠后 state_dict 与
   逐层规格 JSON；记录 `weight_export_sha256`（对导出包做 SHA256）。
3. **INT8 量化**：正式路径固定为
   `per_tensor_symmetric_int8_v2`（`src/hwnas_fpga/deploy/quantization.py`）。
   折叠后的权重使用对称 INT8，bias 使用输入 scale × weight scale 定标的
   INT32 累加域；每层必须有明确的输入/输出 scale 和重定标记录。
4. **软件整数参考模拟**：`src/hwnas_fpga/deploy/int8_reference.py` 与
   `src/hwnas_fpga/deploy/fixed_point.py` 执行完整整数图；不支持的算子必须
   fail-closed（失败即停），不得回退到 FP32。软件参考通过后仍不能替代
   HLS parity。
5. **HLS 侧**：折叠 denoise 走现有 depthwise+pointwise conv 模板；折叠
   edge 走标准稠密 conv k3 模板（`hls_lut_builder` 现有算子库均已覆盖，
   不新增模板）。跑 csynth + OOC/route，产出
   `hls.evidence_complete=true`、`hls.route_feasible=true`，并记录 HLS
   实际消费的量化规格路径、SHA256、证据文件 SHA256 和工具版本。
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

交付脚本 `scripts/make_synthetic_speckle_pairs.py`，规格：

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
| EPI | ↑ | 不低于同档"参考高斯"（见下），且不低于 noisy 基线 |
| SSI | ↓ | < 1 |
| ENL（同质区） | ↑ | 高于 noisy |

**EPI 合格线修订（2026-07-10 实测）**：原定的绝对线 EPI≥0.85 被证伪——
在 n=58 子集上对 9 种单通道单遍滤波变体（高斯 r=1/1.5/2、经典 Lee、
k 平滑/k² Lee、聚合残差门控、μ 梯度门控、两遍门控、log 域）做了全扫描，
L=1 的 EPI 天花板约 0.52、L=4 约 0.69，任何轻量单遍滤波都到不了 0.85。
改为**相对参考线**：每档 L 以"该档最优半径的纯高斯"为参考
（L=1 参考 r≈1.5–2：EPI 0.522；L=4 参考 r≈1：EPI 0.690），候选算子的
EPI 不得低于同档参考且 SSI 不得高于参考。扫描明细见
`results/` 下的 gate sweep 记录与本文档 E4d。

E3 全程 CPU 可跑，不占训练 GPU，可立即启动。

---

## E4：条件触发的设计迭代（E1 出结果后才启动）

| 编号 | 触发条件 | 实验 | 配置要点 |
|---|---|---|---|
| E4a 浅层注入 | E1 中 denoise Δ≤0 | denoise 移出竞争槽位，作为 stem 后固定预处理层 | 骨干同 v1 但 stage3 恢复双 mbconv 对照；新增 `stem_denoise=true` 变体；同协议 5 折×3 seed 对比 |
| E4b log 域 | E3b 显示乘性噪声假设成立（L=1 档 SSI 改善明显） | 输入变换 `x → log(1+255x)/log(256)`（部署为一张 256 项查表） | 在 dataset 归一化层加开关；与 E4a 正交，可 2×2 因子设计。注意 2026-07-10 扫描中 log 域纯图像空间指标为负收益（PSNR 大跌），其价值只可能在特征空间训练中体现，指标扫描不足以否决但也不支持 |
| E4c edge 瘦身 | E1 中 edge Δ>0 但未过 Holm，或 LUT 预算触顶 | `edge_v2`：2 方向（Gx/Gy）+ 幅值近似 `|gx|+|gy|`，融合输入 2C | 折叠后参数约为 v1 的 50%；匹配对照改为 mbconv e2（需重算 5% 匹配） |
| E4d 自适应门控 denoise_v2 | E1 中 denoise Δ≤0，或 EPI 相对参考线不达标 | `adaptive_denoise`（已实现，`AdaptiveDenoiseBlock`）替换 denoise 槽位，同协议 15 runs vs mbconv_control 与 denoise v1 | 见下文设计说明 |

每个 E4 实验的运行量与 E1 单变体相同（15 runs），统计方法同 E1。

### E5：v2 算子重新设计（2026-07-21，基于 E1 判决）

E1（60 runs，冻结协议）判决 v1 算子：`denoise` Δ=-0.012（无效果／与 mbconv
冗余）、`edge` Δ=-0.092（明显有害）。两者病因不同，v2 分别对症：

| v2 算子 | op 名 | 修复的病因 | 关键设计 |
|---|---|---|---|
| `AdaptiveDenoiseBlock` | `adaptive_denoise` | v1 固定高斯平滑 → 冗余 | Lee 式可学习门控 + 池化聚合边缘证据；门可关死退化回 v1，行为空间含 v1 |
| `EdgeAugmentBlock` | `edge_v2` | v1 丢弃强度/DC → 有害 | 强度主路 + **加性**边缘旁支，`gamma` 初始 0（起步即普通 dw_pw，**保底不劣**）；2 方向替代 4 方向降 LUT |

实现：`src/hwnas_fpga/models/builder.py`；测试 `tests/test_edge_augment.py`、
`tests/test_adaptive_denoise.py`（21 项通过，含 gamma=0 等价性、信息保留、
梯度通路）。配置生成器：`scripts/make_sonar_g5_v2_candidates.py` →
`configs/ablation/sonar_g5_v2/`（2×2 因子设计，骨干与槽位同 v1）。

容量匹配（v2 算子不可折叠，按可训练参数 + stage-3 MACs 直接匹配）：
`adaptive_denoise` 参数差 1.98%、MACs 差 0%（**满足 G5 ±5%**）；
`edge_v2` 比其 mbconv e2 对照**更便宜**（参数 -11%、MACs -29%），
故其正向结果是保守的，负向结果也不能归因于对照容量劣势。

判据与 E1 一致：5 折 × 3 seed × 4 变体 = 60 runs，配对分层 bootstrap
+ Holm 校正。**在 v2 通过之前，`adaptive_denoise`/`edge_v2` 同样不得进入
正式搜索空间。**

### E4d：AdaptiveDenoiseBlock（denoise_v2）设计与证据

实现：`src/hwnas_fpga/models/builder.py` 的 `AdaptiveDenoiseBlock`，
op 名 `adaptive_denoise`（未准入前不得进入正式搜索）。结构：

```text
mu  = smooth(x)                  # softmax 归一化可学习核，高斯初始化（同 v1）
d   = x - mu
e   = avgpool3(|d|)              # 空间聚合的边缘证据
g   = sigmoid(alpha*e + beta)    # 逐通道可学习门控（Lee 的 k 系数的可学习化）
lee = mu + g*d                   # 均匀区→局部均值，结构区→保留
out = PW(ReLU(feat(x) + lee))    # feat 分支与 PW/残差同 v1
```

设计由 2026-07-10 合成斑点扫描直接约束
（[results/sonar_gate_sweep_20260710/gate_sweep_summary.md](../results/sonar_gate_sweep_20260710/gate_sweep_summary.md)）：

1. 门控证据必须空间聚合（`avgpool3(|d|)` 而非逐像素 `|d|`）——经典 Lee
   的逐像素门控在 L=1 下 EPI 0.39，反而低于纯高斯 0.49；
2. `beta` 可学习使训练能把门关死退化回 v1 纯平滑，v2 行为空间包含 v1；
3. 平滑强度与噪声水平的匹配是 EPI 第一决定因素，可学习核按训练分布
   自动校准有效半径——这是固定代理（r=1 高斯）不具备的；
4. 单通道手工滤波的 EPI 天花板（L1≈0.52）说明经典代理只能定下界，
   v2 的最终判定必须走 E4 消融（分类 Δmacro_f1），不走图像指标。

部署与准入前置：门控依赖输入，**不可折叠**为单一静态卷积；INT8 部署
sigmoid 用 256 项查表，parity 契约需为查表语义单独定规格。进入搜索前
还需补 `cost.py` 成本条目与 strict LUT 证据，匹配对照重新按折叠后
参数计算（门控本身无新增卷积，参数量与 v1 基本一致）。

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

当前交付物与执行边界：

1. `configs/ablation/sonar_g5_v1/*.candidate.json`（4 个变体）；
2. `configs/ablation/sonar_g5_v1/matching_report.json`（折叠部署形态的
   参数量/MACs 匹配报告）；
3. `scripts/compare_sonar_ablation_bootstrap.py`（fold-cluster 配对 bootstrap
   + sign-flip + Holm）；
4. `scripts/export_folded_sonar_weights.py`（折叠导出 + SHA256 + INT8 v2 规格）；
5. `scripts/make_synthetic_speckle_pairs.py`（合成斑点配对数据）；
6. `scripts/run_g5_ablation_queue.ps1`、`scripts/finish_g1_missing.ps1` 和
   `scripts/finalize_protocol_summary.py`（可恢复的 G1/G5 执行与离线汇总）；
7. E4c 触发时：`edge_v2` 块实现 + 折叠函数 + 匹配重算。

当前没有可提交为实验结论的 E1 四路 60-run 结果、E2 零差异 parity/HLS
证据或 route-feasible 证据；实现和候选配置的存在不等于 G5 通过。

## 声明边界

- E1 数字是软件验证集分类证据；E2 是软件整数/HLS 部署语义证据；E3 是图像质量证据。
  三类证据分开报告，不得合并成单一"声呐效果"结论。
- 在 G5 重审为 PASS 之前，`denoise`/`edge` 不得进入任何新的正式搜索。
