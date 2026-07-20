# 对标 campaign 正式就绪状态（中文伴随档案）

## 来源与绑定

- Campaign：`ccf_ab_nksid_av7k325_v1`。
- 英文机器报告：`formal_readiness.md`。
- 英文报告 SHA256：`d1f7ef8d8e72a09a5e84537dc810eb18a78850b3f2700645b9642bddd5c7a16e`。
- 当前状态：`NOT_READY`；通过 `2/9`。
- 本文件只解释当前证据，不增加正式实验结果。

## 九项要求

| 要求 | 状态 | 阻塞项 |
|---|---|---|
| 论文、代码版本与许可证/对应关系 | `PASS` | 无 |
| 论文专用隔离环境 | `PASS` | 无 |
| 闭集四方法统一协议 | `PENDING` | 闭集四方法的 15 单元正式结果不完整 |
| 开放集与长尾三方法统一协议 | `PENDING` | 开放集三方法的 15 单元正式结果不完整 |
| 四种 NAS 策略正式比较 | `PENDING` | 等预算 NAS 正式比较缺失或不完整 |
| HLS/route 代理可靠性 | `PENDING` | HLS/route 正式样本阈值和 G2 尚未满足 |
| AV7K325 板级与功耗 | `PENDING` | AV7K325 三候选板级/功耗证据不完整 |
| 测量优先门禁 | `PENDING` | 测量优先门禁尚不允许完整正式 campaign |
| T1-T9 与 F1-F12 可重建归档 | `PENDING` | 正式表格和图片仍未全部完成 |

## 关键解释

R3 只接受逐样本预测齐全、source freeze 文件与归档哈希可复核、且 tracked patch 内容哈希一致的 15 个 fold-seed 单元。旧 scratch 的 patch provenance 失效后，不得继续被 readiness 计为正式证据；当前映射使用 scratch-v2。SURE 仍为 0/15，正式执行开关保持关闭。

## 当前边界

smoke、代码存在、作者原始数值、跨 FPGA 平台结果和历史代理值都不能替代本项目统一协议结果。SURE、HLS、板卡或功耗实验仍须单独授权。
