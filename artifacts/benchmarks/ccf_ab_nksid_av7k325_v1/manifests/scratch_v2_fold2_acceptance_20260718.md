# scratch-v2 fold 2 接纳记录

## 结论

- 状态：`PASS_CONTINUE`。
- fold 2 已完成 seeds 42、43、44，共 `3/3` 个单元。
- fold 2 mean macro_f1（平均宏 F1）：`0.9145918032226666`。
- fold 2 mean top1（平均 Top-1 准确率）：`0.9717948717948718`。
- 三个单元相对旧 scratch 同 fold/seed 的 macro_f1 绝对差均为 `0.0`。
- 三个单元的 checkpoint、逐样本预测、source freeze 和 manifest-bound patch 在线核验均通过。

## 单元记录 SHA256

| 单元 | macro_f1 | top1 | run record SHA256 |
|---|---:|---:|---|
| fold2/seed42 | 0.9450248111 | 0.9826923077 | `9abcfeafe61c6bd0869a61e9c2425aa314d1c8b0a00b676f7e2e4cdc29e503ea` |
| fold2/seed43 | 0.8717048899 | 0.9538461538 | `50d661e8f7e257714e8d60df50b353f9e2868a43112e2cf33525b326ed43bf55` |
| fold2/seed44 | 0.9270457087 | 0.9788461538 | `53ea3f131af732a3f3293c9f2683d1885a89fdd69ef39cd8d7df8570743766d4` |

## 证据边界

scratch-v2 在线进度为 `9/15`，允许进入 fold 3。正式 summary 尚未生成，因此测量优先总账继续保持 `PENDING 30/45`。最低单元 macro_f1 为 `0.8717048899`，高于预声明的 `0.80` 中断阈值；没有触发关键决策暂停。旧 scratch 的 manifest 与损坏后的 patch 实测 SHA 均保持原值，未被改写。未启动 SURE、HLS、route、COM5、板级或功耗实验。

