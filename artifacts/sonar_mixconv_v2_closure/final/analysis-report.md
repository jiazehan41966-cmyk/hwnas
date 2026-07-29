# 声呐部分完整闭环结果

最终状态：`NO_OPERATOR_GAIN`。

## 结论

正式训练配方修正成立：关闭 logit adjustment 后，相对历史配方的 outer macro_f1 平均增益为 `+0.1283`，分层配对 bootstrap 95% CI `[+0.1045, +0.1508]`，5/5 个 fold 为正。因此默认配方改为 `logit_adjust_tau=0`。

成熟 MBConv-5×5 相对 MBConv-3×3 的平均增益为 `+0.0135`，但分层 95% CI 为 `[-0.0216, +0.0448]`，且只有 `3/5` 个 fold 均值为正，未达到 CI 下界大于 0 且至少 4/5 fold 为正的门禁。

MixConv-v2 在 inner-only 筛选中相对 3×3 为 `+0.0138`，但相对成熟 5×5 为 `-0.0164`，已在正式 outer 前停止。

所以本轮没有把自研算子或 5×5 大核写入正式搜索空间；正式主线仍为 `conv/mbconv/skip`。得到的可用改进是训练配方和 letterbox 数据接口，不是一个通过门禁的新声呐算子。

## 数据与稳健性

letterbox_224 在随机 split 和推断压力 split 上的三种子平均方向均为正，因此进入正式实验；它只保持宽高比，不被表述为保留原生尺度。

MBConv-5×5 的 inner synthetic robustness 平均差为 `+0.0083`，通过 −0.01 非劣门槛；但稳健性通过不能覆盖正式精度门禁失败。

## 硬件边界

MixConv-v2 的实际 Vitis HLS 估计时钟为 `5.301 ns`，目标为 5 ns，因此 200 MHz 门禁为 `False`。这是隔离算子 HLS，不是完整网络 route。由于 MixConv-v2 在 inner 筛选失败、MBConv-5×5 在正式精度门禁失败，后续 INT8/HLS/route 按顺序门禁停止，没有把未执行项写成 PASS。

## 主要限制

- 算子和配方先在 fold0 inner-only 上筛选，并非完全 nested selection；
- inferred_stress 分组只能说明近重复泄漏压力，不能代表航次或源帧泛化；
- 合成斑点、对比度和模糊是分类稳健性测试，不是 PSNR/SSIM 复原证据；
- letterbox 不保留样本间原生像素尺度；fixed_scale_pad 明显降低了小目标分辨率；
- 未通过前置门禁的候选没有继续烧录完整 route，状态明确为顺序停止而非通过。
- 声呐聚焦测试为 95 passed；全仓测试为 497 passed、17 failed、3 skipped。17 个失败来自隔离工作树缺少被忽略的外部 benchmark checkout/archive 和生成型 HLS/LUT/route 产物，不被改写为全仓 PASS。

完整逐运行路径与 SHA256 见 `complete_record.json`。
