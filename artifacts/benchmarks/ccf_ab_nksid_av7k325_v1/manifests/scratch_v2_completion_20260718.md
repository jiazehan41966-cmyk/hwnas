# scratch-v2 最终完成与暂停记录

## 完成结论

方案 1 已完成：provenance 修复、测试、重新冻结和 scratch-v2 正式执行均已闭合。scratch-v2 在冻结 NKSID 协议下完成 5 outer folds × seeds 42–44，共 15/15 个配对单元；独立审计为 `PASS`，测量优先总账 G1 为 `PASS`（45/45）。一次性可执行授权已改名为 `scratch_v2_execution_authorization_20260718.consumed.json.txt`，消费前后 SHA256 均为 `c63d31b4ac203ce07cef40c6c1d4297f30328e3d6191ae76ae171a3f3402e3ce`。

## 正式软件分类结果

| 指标 | 15 单元均值 | 标准差 |
|---|---:|---:|
| macro_f1（宏平均 F1） | 0.909902 | 0.024014 |
| top1（Top-1 准确率） | 0.972089 | 0.011606 |
| weighted-F1（加权 F1） | 0.975405 | 0.009432 |

各 outer fold 的三种子 macro_f1 均值依次为：fold 0 `0.918018`、fold 1 `0.894826`、fold 2 `0.914592`、fold 3 `0.908849`、fold 4 `0.913223`。15 个 scratch-v2 macro_f1 与旧 scratch 诊断值逐项完全一致；这证明数值复现，但不修复旧 scratch 已被覆盖的 patch provenance，旧结果继续保持不可声明。

类别层面的当前薄弱项是 fishing_net（渔网），mean outer F1 为 `0.5707`；其余类别不能掩盖该长尾/难类问题。该结果不触发 scratch-v2 失败停止，因为预声明停止策略的所有单元均通过，且与旧诊断完全复现，但后续方法比较必须单列该类别。

## 三方法阶段性配对分析

- 从零训练 MobileNetV2：macro_f1 `0.909902 ± 0.024014`。
- ImageNet 预训练 MobileNetV2：`0.931632 ± 0.024188`。
- 冻结 NAS 候选 `rl_arch_135`：`0.693619 ± 0.029237`。
- 预训练相对 scratch-v2 的配对均值差为 `0.021730`，95% 分层 bootstrap CI `[0.010925, 0.032833]`，Holm 校正 p=`0.00213623`。
- scratch-v2 相对冻结 NAS 的配对均值差为 `0.216283`，95% CI `[0.202079, 0.229091]`，Holm 校正 p=`0.00018311`。

因此，`rl_arch_135` 可以作为“硬件可行但精度较弱”的负结果，不得描述为精度竞争方法。SURE 尚未执行，三方法分析只是阶段性诊断，不是完整 T2；SURE 加入后必须重新计算完整实验族的 Holm 校正。

## 关键证据与哈希

- source freeze manifest：`cfbc7ec9373e762c39385d733e07682a39ef843f87c2a25100c3fb7bfb824f32`；snapshot ZIP：`990d1dd08b1e966c4e85ff76661b4ceffb0986200a25859c8540cafe359a6ed3`；冻结核验 557/557。
- scratch-v2 协议 JSON：`e70fcc028b5ad16897ec6ed0c034bfd8ec8672ed028d551373e850632e1ba636`。
- 训练绑定的 content-addressed patch：`provenance/code_patch_ccb0feef0dafc34a2b4fb0e2f751b698ad4a1acd9c3696fbafb6b88df5dc7280.diff`，文件名哈希与内容哈希一致。
- scratch-v2 独立审计 JSON：`0ee6e3d982ad902762350c6d5913fe00aacecfbf00ed08e6a54b2c1cce869e3b`。
- 三方法统计包 manifest：`731e2520cf11fd697d3b0d689a237a6fc89b0f58b6d3f00d4e50372f585af112`。
- 三方法统计包独立审计 JSON：`295712b0e71c93c2a05b6ffb9526d3d22e74c4c613419ae8336f14a3075e90a0`，机器审计 `PASS`；两幅最终 300 dpi PNG 已人工目视检查，中文字体、坐标轴、图例和置信区间无截断或乱码。
- 守护脚本退出码回归测试 JSON：`3a713b9e05662f7ba31cc562faea4162fa0b9fd37a548bd3a605423d43db2b41`，状态 `PASS`。
- 测量优先总账 JSON：`c889d9c2f0e61cbea3a438c3e75e5454c6e208c7875265ce4c631835f7e63f7e`；G1 `PASS`，总体仍为 `IN_PROGRESS`。

## 异常处置

训练完成后的“非零退出”记录已确认是 ExitCode 读取时机导致的守护脚本误报，原始中断档案被完整保留。修复后的脚本通过 0 与 7 两种子进程退出码回归测试；没有为验证修复而重跑训练。详见 `scratch_v2_false_interruption_resolution_20260718.md`。

## 暂停边界与待用户决策

本轮到此自动暂停。没有启动 SURE、corruption、开放集、NAS 新搜索、HLS、route、COM5、AV7K325 板级或功耗实验。当前状态仍为 G2 `PENDING`、G3 `FROZEN`、G4 `PENDING`、功耗 `NOT_MEASURED`、G5 `PAUSED`。下一项关键决策是是否授权执行 SURE 的 15 个闭集 fold-seed 单元；在用户决定前不得继续。
