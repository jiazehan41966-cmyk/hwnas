# 固定 4-Stage 声呐算子实验结论边界

本页仅记录可由当前 Protocol V2 证据支持的论文表述边界。硬件部署结论需等待整网 INT8、HLS/RTL、route、bitstream、COM5 与外部功耗证据闭环后再更新。

## 当前可写主结论

在固定 4-Stage 声呐骨架下，成熟 MBConv-k5 系列获得算法与精度层面的正式支持。Stage2-K5 在基础 8 结构的 2×2×2 因子实验中通过完整 5-fold × 3-seed Protocol V2 评价；Stage4-K5 在完成 28×28、32→32、stride=1 exact-shape micro-harness route 门禁后，被纳入扩展正式实验，并在 3 个 Stage2 背景上通过预注册精度门槛。因此当前总体状态为 `GENERAL_OP_SELECTED`。

Dir-MBConv3 的方向依据诊断通过，但 `dir_mbconv3_split11_e3_v1` 未通过正式精度准入门槛，应作为阴性结果报告。论文中不能写“方向算子有效”，只能写“方向性假设经训练数据诊断成立，但该 Dir-v1 设计未带来稳定准入收益”。

## 不应写出的结论

- 不应把 Dir-v1 继续包装成可准入算子。
- 不应把 Stage4-K5 exact-shape micro-harness route 写成完整网络 route。
- 不应把 strict LUT proxy（严格 LUT 代理）、micro-route、full route、COM5 latency（板级串口延迟）和 power（外部仪表功耗）混为同一层证据。
- 没有外部仪表实测 CSV 前，power 一律为 `NOT_MEASURED`。

## 下一轮整网部署闭环对象

整网硬件闭环只围绕已冻结的 3-4 个代表候选开展，不再扩大搜索空间，不再优化 Dir-v1。候选由 `artifacts/sonar_fourstage_operator_v2/fourstage_deployment_candidate_selection.json` 记录，按固定规则从既有正式结果中选择：

1. 原始基线：Stage2 K3-E3 + Stage4 K3-E3。
2. Stage2-K5 代表：既有正式结果中 Stage2-K5 且 Stage4 为 K3 或 Skip 的最佳候选。
3. Stage4-K5 代表：已通过 Stage4-K5 预注册精度门槛的最佳候选。
4. 低成本代表：既有正式结果中最佳 Skip 候选。

当前已完成真实 checkpoint 导出、训练数据 activation calibration（激活校准）、Python full-network INT8 reference（完整网络整数参考），以及 4 个冻结代表候选的 Vitis HLS C-sim zero mismatch（C 仿真整数输出零差异）。C-sim 使用每个候选 fold0/seed42 真实 checkpoint 和训练集前缀样本，4 个候选均为 8 个样本 × 8 类输出、mismatch=0。该证据仍不等于 RTL co-sim、完整网络 HLS synthesis、完整网络 route、bitstream、COM5 或外部仪表功耗。

后续门禁顺序固定为：可综合完整网络 HLS 实现、RTL co-sim、AV7K325 5ns Place & Route、bitstream、COM5 板级 latency，最后才是外部仪表功耗。

最终可部署空间尚未冻结。若 Stage4-K5 完整网络硬件闭环通过，可部署空间为 Stage2 4 种 × Stage4 3 种，共 12 种；若 Stage4-K5 完整网络硬件闭环失败，可部署空间回退为 Stage2 4 种 × Stage4 2 种，共 8 种。
