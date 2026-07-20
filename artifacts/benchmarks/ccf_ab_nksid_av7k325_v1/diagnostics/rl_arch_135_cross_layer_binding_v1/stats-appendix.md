# 统计附录

## 软件分类比较

- 主指标：macro-F1，越高越好。
- 分析单位：fold-seed 配对单元；5 folds × 3 seeds，n=15。
- 描述统计：NAS `0.6936187231 ± 0.0292365606`；预训练 `0.9316318801 ± 0.0241884368`。
- 配对均值差：`-0.2380131570`。
- 不确定性：按 fold 分层的 10,000 次配对 bootstrap，seed 20260717，95% CI `[-0.2639684924, -0.2126668049]`。
- 精确双侧配对 sign-flip permutation：p=`0.00006103515625`。
- 效应量：Cohen's dz=`-5.4963062324`。
- 多重比较：本 bundle 只检验一个预先声明的 macro-F1 主比较，不进行 Holm 调整；完整 T2 实验族仍需 scratch-v2 和 SURE 后统一校正。

## 硬件描述统计

- route 只有一个架构实例，不做推断统计。
- COM5 有 5 次相同 bitstream、相同确定性输入的重复测量；latency 标准差为 `0.0000000000 ms`。零方差只表示该 harness 在这五次运行中稳定，不能代表真实 NKSID 输入分布。
- 搜索 proxy 与 COM5 的 `6.733021×` 比例是单候选描述，不是代理校准模型。
- 板级分类准确率未运行，功耗未测量，因此没有相应均值、CI、p-value 或效应量。

## 主要限制

- formal checkpoint 与 bitstream 权重未绑定。
- 架构由历史 fold0 流程选择，`nas_generalization_claimable=false`。
- acquisition/mission group 元数据缺失。
- 旧 Vivado stdout 存在编码替换字符；核心数值字段不受影响，但文本 provenance 不完整。
