# SURE 正式实验就绪性与执行决策卡

- 状态：**可提交用户决策，但尚未授权**（`READY_FOR_USER_DECISION_NOT_AUTHORIZED`）。
- 本轮没有启动训练；SURE 正式结果仍为 `0/15`。
- 作者代码 commit：`5ce0193bc93e73b1c7f1f53aeda8854e997011e2`；远程：`https://github.com/Intellindust-AI-Lab/SURE.git`。
- 许可证：仓库未发现 LICENSE/COPYING/NOTICE，只允许隔离目录本地重评，禁止再分发作者源码。
- 专用 CUDA 环境实时 probe：`PASS`。
- MobileNetV2 + 作者 cosine classifier 静态装配：`PASS`；只做 CPU 随机张量前向，没有训练。
- 现有方法 smoke：SimpleCNN、fold 0、seed 42、1 epoch；不能证明正式 MobileNetV2 的显存与稳定性。
- 唯一正式入口：`run_eval_protocol.py`，且必须绑定新的 source freeze。
- 分阶段续接：`--max-new-units 1` 不进入实验 fingerprint；首单元后可在代码状态不变时安全复用并续跑。
- 授权模板状态：`NOT_AUTHORIZED_TEMPLATE`，守护程序拒绝用模板触发训练。

## 为什么不能直接启动 15 单元

正式配置开关仍为 `false`，scratch-v2 授权已经消费且不能复用。SURE 使用 SAM 双步更新、CRL、RegMixup、SWA 和 cosine classifier；当前配方关闭 AMP、梯度累积固定为 1。RTX 3050 Ti 仅 4 GB，SimpleCNN 64×64 smoke 不能排除 MobileNetV2 224×224 的 OOM 或显著变慢风险。

## 三种可选执行方式

| 方案 | 流程 | 会导向的结果 | 建议 |
|---|---|---|---|
| S-A | 新冻结 → MobileNetV2 1-epoch smoke → 首个 150-epoch 单元 → 强制暂停请用户决策 → 再决定剩余 14 单元 | 最早暴露 OOM、数值崩溃、训练时间或首单元效果问题；保留负结果，又不自动消耗全部预算 | **推荐** |
| S-B | 新冻结 → smoke → 一次运行全部 15 单元 | 最快获得完整 T2，但首单元很差也会继续，违背关键决策交给用户的要求 | 不建议 |
| S-C | 只做 MobileNetV2 1-epoch smoke | 只证明接口和显存可运行，不能形成论文比较结果 | 适合暂不投入长训练 |

## 推荐停止规则

技术 smoke 出现 OOM、NaN/Inf、作者源码哈希或 source freeze 不一致、逐样本预测缺失时立即 fail-closed。S-A 的首个 150-epoch 单元无论好坏都强制暂停，提交 macro_f1、top1、ECE、NLL、Brier、AURC、failure AUROC/FPR95、逐类 F1、显存和耗时，由用户决定是否继续；不预设会掩盖负结果的自动精度淘汰线。

## 时间与证据边界

scratch-v2 的 15 单元实测耗时约 `8.16` 小时。SURE 正式时间尚未测量；基于双步 SAM、SWA BN 更新和关闭 AMP，容量规划暂按 18–30 小时，必须由 1-epoch MobileNetV2 smoke 校准。该估计不得写入论文结果。

本决策卡不授权 SURE、HLS、route、COM5、AV7K325 板级或功耗实验。
