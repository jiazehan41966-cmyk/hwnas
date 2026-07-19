# 图片目录

## diagnostic_paired_macro_f1

- 用途：判断性能下降是孤立现象，还是在配对单元间一致出现。
- 来源：`data/paired_fold_seed_metrics.csv`。
- 读图重点：每个 NAS 点都低于两种 MobileNetV2 参照方法。
- 决策作用：支持“精度—容量权衡”的诊断与受控暂停，不支持静默重试。
- 限制：NAS 仅完成 14/15，本图不是正式图。

## diagnostic_per_class_f1

- 用途：按类别定位 macro-F1 的差距来源。
- 来源：`data/per_class_f1.csv`。
- 读图重点：fishing_net 与 small_propeller 的差距最大。
- 决策作用：更支持容量不足或长尾敏感性，而不是统一的标签或预处理故障。
- 限制：support 会在不同 seed 间重复，仅作描述。
