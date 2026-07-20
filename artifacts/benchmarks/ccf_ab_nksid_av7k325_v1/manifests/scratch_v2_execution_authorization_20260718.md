# scratch-v2 一次性执行授权

## 用户决定

- 用户原话：`应用 provenance 修复、补测试、重新冻结，再决定/执行 scratch-v2。`
- 解释：在修复、测试和新源码冻结全部通过后，允许执行新的 scratch-v2 正式基线；不允许把旧 scratch 原地修补为通过。
- 授权状态：`APPROVED_ONCE`。

## 前置证据

- provenance 专项及相关测试：`23 passed`。
- 全量回归：`478 passed, 1 warning, 28 subtests passed`；唯一警告为既有学习率调度顺序警告，与本修复无关。
- 新源码冻结：`g1_20260718_v3`，核验 `557/557 PASS`。
- 冻结 manifest SHA256：`cfbc7ec9373e762c39385d733e07682a39ef843f87c2a25100c3fb7bfb824f32`。
- 冻结归档 SHA256：`990d1dd08b1e966c4e85ff76661b4ceffb0986200a25859c8540cafe359a6ed3`。
- 正式入口 SHA256：`8f649be022273a0bdd9633795aeba3020e401b845967eab34c21297ee31b043f`。
- G1 总账脚本 SHA256：`2a7b6941d1d47b0be10fa71882c3cc386ba134b22d582f645fcbbb9ebbc16b19`。

## 唯一允许范围

- 数据：`data/NKSID`。
- 任务：闭集分类。
- 模型：从零训练的 `MobileNetV2`。
- 协议：outer folds 0–4，seeds 42、43、44，共 15 个配对单元。
- 训练：150 epochs、batch size 8、梯度累积 4、AMP、保存 checkpoint。
- 运行名：`g1_mobilenet_v2_scratch_v2`。
- 输出根：`results/protocol/g1_clean_20260718/`。
- 方法标识：`scratch_mobilenet_v2`。
- 选择来源：`baseline_predeclared`。

## 预先声明的中断条件

遇到下列任一条件，必须中断，不得自动改变协议或继续下游实验：

1. 冻结 manifest、归档、正式入口或审批文件 SHA256 不一致。
2. 训练进程非零退出，或出现 NaN/Inf、CUDA/文件系统异常。
3. 任一已完成单元缺少 checkpoint、逐样本预测或记录，或者其 SHA256 不一致。
4. 新单元 macro_f1（宏平均 F1）低于 `0.80`。
5. 新单元 macro_f1 与旧 scratch 相同 fold/seed 的诊断值绝对差超过 `0.05`；该条件仅用于发现异常，不恢复旧 scratch 的可声明性。
6. 检测到旧 scratch 的 `run_manifest.json` 或 `code_patch.diff` 被改写。
7. 新 run 的 manifest-bound patch、source freeze、data、split、config 或 code-state SHA 不一致。

## 明确禁止

- 不启动 SURE、corruption、开放集、NAS 新搜索、HLS、route、COM5、AV7K325 板级或功耗实验。
- 不改变候选、归一化、fold/seed、训练配方或数据划分。
- 不覆盖、移动或修补旧 scratch 结果。
- scratch-v2 完成后先进行独立审计与 G1 总账复核；任何后续方向仍需用户单独决定。

