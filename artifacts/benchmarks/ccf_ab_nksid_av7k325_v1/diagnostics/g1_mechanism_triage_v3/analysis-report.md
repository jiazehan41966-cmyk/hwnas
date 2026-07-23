# G1 机制诊断总报告

## 结论

最终门禁：**CAPACITY_SWEEP_HOLD__RECIPE_FACTOR_VALIDATION_NEXT**。

训练曲线与 15 对 checkpoint 重评确认 `rl_arch_135` 在当前冻结配方下存在真实训练侧
拟合差距；但这不等价于已经隔离「参数容量」。micro-overfit 排除了小子集上的粗粒度
实现/可训练性故障；单折全训练因子实验的状态为
`LOSS_OR_REGULARISATION_SENSITIVITY_CONFIRMED__SINGLE_FOLD_DIAGNOSTIC`。因此当前动作是：在 fold0 的 seed43、44 上重复最小配方因子对照；然后再决定是否值得投入正式 15-run 协议比较。

## 证据阶梯

1. **在线增强训练曲线（15 对完整运行）**：NAS=0.6824，
   scratch=0.9878，配对差=0.3054，
   95% CI [0.3010, 0.3102]。
   该指标受随机增强与 train mode 影响。
2. **best checkpoint 的无增强训练索引重评（15 对）**：clean top-1 差
   0.1610，95% CI
   [0.1471, 0.1746]；clean macro_f1
   差 0.2429。真实 clean-fit gap 成立。
3. **96 张固定类平衡子集**：至少一个新鲜 NAS 模型达到 0.99 以上，说明该架构并非
   连小样本都无法记忆；学习率存在可见敏感性。
4. **全训练索引的单折三臂诊断**：只使用 fold0/seed42 的 train 与 inner，未访问 outer。
   它只能定位机制并决定下一门禁，不能产生正式泛化结论。

## micro-overfit 结果

| 变体 | best clean top-1 | best clean macro_f1 | 首次 top-1≥0.99 epoch |
|---|---:|---:|---:|
| frozen_regularization_lr1e-3 | 1.0000 | 1.0000 | 79 |
| plain_ce_lr3e-4 | 0.9792 | 0.9791 | 未达到 |
| plain_ce_lr1e-3 | 1.0000 | 1.0000 | 98 |
| plain_ce_lr3e-3 | 1.0000 | 1.0000 | 57 |

## 单折全训练因子结果

| 变体 | best clean top-1 | best clean macro_f1 | best inner macro_f1 | epochs |
|---|---:|---:|---:|---:|
| clean_input_frozen_loss_lr1e-3 | 0.9394 | 0.8868 | 0.6122 | 150 |
| clean_input_plain_ce_lr1e-3 | 1.0000 | 1.0000 | 0.8266 | 90 |
| clean_input_plain_ce_lr3e-3 | 1.0000 | 1.0000 | 0.7564 | 65 |

历史原配方 fold0/seed42 best checkpoint：clean top-1
0.8228，clean macro_f1
0.7476；原记录 inner macro_f1
0.6733（不同训练过程，仅作同一运行背景）。

## 实验决策

- capacity sweep（容量扫描）：**HOLD**。
- distillation（知识蒸馏）：**HOLD，等待配方因子验证**。
- 四臂预处理：**HOLD，等待源图/采集场次分组 split 合约**。
- 下一门禁：在 fold0 的 seed43、44 上重复最小配方因子对照；然后再决定是否值得投入正式 15-run 协议比较。
