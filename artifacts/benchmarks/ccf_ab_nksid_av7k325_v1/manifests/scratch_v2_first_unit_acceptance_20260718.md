# scratch-v2 首单元接纳记录

## 接纳结论

- 状态：`PASS_CONTINUE`。
- 单元：fold 0、seed 42。
- macro_f1（宏平均 F1）：`0.9324732117481844`。
- top1（Top-1 准确率）：`0.9826923076923076`。
- 旧 scratch 同单元诊断值：`0.9324732117481844`。
- macro_f1 绝对差：`0.0`，未触发 `0.05` 中断阈值。
- 当前 macro_f1 高于 `0.80` 中断阈值。

## 完整性证据

- run record SHA256：`da3a82a4dfd025e00b574fdf7076e75819b0704cf19c05ce3b12222a5326438e`。
- checkpoint SHA256：`f521870e6fe489ea76e383795383b1935f50415fa4621036bf09a0a5fbbf019e`。
- 逐样本预测 SHA256：`ea689c26fdbda32f7fc58e1a5cf9962201398236acec29ffa52c52a429c51ac2`。
- source freeze 状态：`PASS`。
- source freeze manifest SHA256：`cfbc7ec9373e762c39385d733e07682a39ef843f87c2a25100c3fb7bfb824f32`。
- manifest-bound patch SHA256：`ccb0feef0dafc34a2b4fb0e2f751b698ad4a1acd9c3696fbafb6b88df5dc7280`。

## 旧证据保全

- 旧 scratch manifest SHA256 仍为：`ba13135545258d3f8c3667782341229d1a8f942143e0e56a57f75cff07f7d8d7`。
- 旧 scratch patch 实测 SHA256 仍为：`0e92055318fd5b96436b7852ca4be030c61f3bd1024a2913da6b0d42c86ef3d1`。
- 本记录不改变旧 scratch 的 `FAIL_PROVENANCE` 状态，也不把新旧 run 合并。

## 执行边界

首单元没有异常，允许按相同冻结协议继续剩余 14 个单元。scratch-v2 在线进度为 `1/15`；正式 summary 尚未生成，因此测量优先总账仍按 `PENDING 30/45` 计数。只有 scratch-v2 15/15、独立审计和总账复核全部通过后才允许改为 PASS。未启动 SURE、HLS、route、COM5、板级或功耗实验。
