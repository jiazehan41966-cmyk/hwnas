# scratch-v2 fold 0 接纳记录

## 结论

- 状态：`PASS_CONTINUE`。
- fold 0 已完成 seeds 42、43、44，共 `3/3` 个单元。
- fold 0 mean macro_f1（平均宏 F1）：`0.9180176354201574`。
- fold 0 mean top1（平均 Top-1 准确率）：`0.9737179487179487`。
- 三个单元相对旧 scratch 同 fold/seed 的 macro_f1 绝对差均为 `0.0`。
- 三个单元的 checkpoint、逐样本预测、source freeze 和 manifest-bound patch 在线核验均通过。

## 单元记录 SHA256

| 单元 | run record SHA256 |
|---|---|
| fold0/seed42 | `da3a82a4dfd025e00b574fdf7076e75819b0704cf19c05ce3b12222a5326438e` |
| fold0/seed43 | `b046a3e155c75cfbe90b54a6bdcb5eb8101921530feb1b6c16b20f5233d7225d` |
| fold0/seed44 | `04dba2cdcf2cbfa3f8e4c12dd2eb9f11b47c7e3200f8d9c98bb43747f18e0e5d` |

## 证据边界

scratch-v2 在线进度为 `3/15`，允许进入 fold 1。正式 summary 尚未生成，因此测量优先总账继续保持 `PENDING 30/45`。本记录不恢复旧 scratch 的 provenance，也不授权 SURE、鲁棒性、开放集、HLS、route、COM5、板级或功耗实验。

