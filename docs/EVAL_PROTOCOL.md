# 冻结评估协议 (nksid_outer5fold_inner_contiguous_v1)

生效日期：2026-07-04。本协议取代 legacy fold-0 单折协议；此后所有可对外声明的
分类指标必须由本协议产出。fold-0 单折、单 seed、best-epoch 选在同一验证集上的
历史数字一律标注为 legacy proxy。

## 动机（对应 2026-07-03 第一性原理审计）

1. legacy 协议在 fold record 0 上同时做架构筛选、best-epoch 选择和最终报告，
   且 95.8% 的验证图像在训练集中有同类相邻编号图像（邻接泄漏）。
2. 验证集仅 520 张、fishing_net 全集仅 20 张，单折单 seed 的 macro_f1 波动
   可达数个百分点；v4 最优 (0.8607) 与 v3 基线 (0.8545) 的差异在噪声带内。

## 协议定义

- **外层**：官方 `kfold_train.txt` / `kfold_val.txt` 的第一个重复
  （记录 `p0-k0` .. `p0-k4`）构成 5 个外层折。加载时强制校验：五折验证集两两
  不相交、并集覆盖全部 2617 样本、train 记录恰为 val 记录的补集；任何偏差
  直接抛 `ProtocolError`，不允许静默回退随机划分。
- **外层验证集只允许消费一次**：训练结束后用 best checkpoint 做一次评估，
  产出报告数字。禁止参与架构筛选、epoch 选择、早停或任何超参决策。
- **内层选择集**：从外层训练索引中，按类别以文件名编号排序后取一个连续
  （可环绕）区块（默认 15%）作为内层验证集，用于 best-epoch 选择。连续区块
  把邻接泄漏限制在每类至多两个交界对，而随机交错会最大化泄漏。
- **多 seed**：正式数字至少 3 个 seed（默认 42/43/44）× 5 折，报告
  mean ± std 与 per-class F1（fishing_net、small_propeller 单列说明）。

实现：[src/hwnas_fpga/data/protocol.py](../src/hwnas_fpga/data/protocol.py)、
[run_eval_protocol.py](../run_eval_protocol.py)。
测试：`tests/test_protocol.py`。

## 训练配方（协议默认）

`run_eval_protocol.py` 默认使用现代配方
（[src/hwnas_fpga/training/recipe.py](../src/hwnas_fpga/training/recipe.py)）：

- AdamW + cosine 衰减 + 5 epoch 线性 warmup（`min_lr_ratio=0.01`）；
- label smoothing 0.1；
- 长尾处理：logit adjustment（`tau=1.0`，Menon et al., ICLR 2021），
  取代逆频率类权重；`--logit-adjust-tau 0` 可关闭；
- 评估 loss 始终用未调整的交叉熵，argmax 用原始 logits。

## 基线三连（阶段 1 命令）

```powershell
# ① MobileNetV2 from scratch（灰度单通道）
python run_eval_protocol.py --arch mobilenet_v2 --epochs 150 `
  --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name baseline_mnv2_scratch

# ② 灰度适配 ImageNet 预训练 MobileNetV2（首层权重按通道均值折叠）
python run_eval_protocol.py --arch mobilenet_v2 --pretrained --epochs 150 `
  --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name baseline_mnv2_pretrained

# ③ v4 搜索最优 rl_arch_135（18.8K 参数）
python run_eval_protocol.py `
  --candidate-path "hls_lut_builder\board_harness\results\pareto_route_gate_phase0_v4_sonar_stage3_k3_lowdsp\candidates\003_rl_arch_135.candidate.json" `
  --epochs 150 --folds 0,1,2,3,4 --seeds 42,43,44 --device cuda `
  --run-name baseline_rl_arch_135
```

每个命令产出 `results/protocol/<run-name>/protocol_summary.{json,md}`，
含 15 次运行（5 折 × 3 seed）的 mean ± std 与 per-class F1。

## 判读规则

- 两个模型的差异只有在 |Δmean| > 合并 std 时才值得讨论；
- 若 ② 显著高于 ③，说明当前搜索空间的精度收益低于一次预训练初始化，
  论文叙事应以"约束下可部署性"为主，精度为辅；
- 与 PLUD (83.68) / DMCL (89.47) 的对比必须注明协议差异
  （它们是开集 5 折协议，不直接可比）。
