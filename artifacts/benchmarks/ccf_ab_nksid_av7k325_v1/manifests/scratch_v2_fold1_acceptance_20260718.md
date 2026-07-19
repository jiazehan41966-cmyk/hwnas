# scratch-v2 fold 1 接纳记录

## 结论

- 状态：`PASS_CONTINUE`。
- fold 1 已完成 seeds 42、43、44，共 `3/3` 个单元。
- fold 1 mean macro_f1（平均宏 F1）：`0.8948260827840461`。
- fold 1 mean top1（平均 Top-1 准确率）：`0.9653846153846154`。
- 三个单元相对旧 scratch 同 fold/seed 的 macro_f1 绝对差均为 `0.0`。
- 三个单元的 checkpoint、逐样本预测、source freeze 和 manifest-bound patch 在线核验均通过。

## 单元记录 SHA256

| 单元 | macro_f1 | top1 | run record SHA256 |
|---|---:|---:|---|
| fold1/seed42 | 0.9094514931 | 0.9750000000 | `cc8f31c1de6a82a6c58454d7ef6e7d4f2953166130b37b975d933067340e2b1a` |
| fold1/seed43 | 0.8544892405 | 0.9442307692 | `a8fe92ec054f843ebcacce4a3e95cd999fe500aaef00bcf3b142c4bf8a9f24c3` |
| fold1/seed44 | 0.9205375147 | 0.9769230769 | `0fe6e73acf7bc28af3fe602adfa20db1fda9db19e664d3b3401f635d12447f0e` |

## 证据边界

scratch-v2 在线进度为 `6/15`，允许进入 fold 2。正式 summary 尚未生成，因此测量优先总账继续保持 `PENDING 30/45`。最低单元 macro_f1 为 `0.8544892405`，高于预声明的 `0.80` 中断阈值；没有触发关键决策暂停。未启动 SURE、HLS、route、COM5、板级或功耗实验。

