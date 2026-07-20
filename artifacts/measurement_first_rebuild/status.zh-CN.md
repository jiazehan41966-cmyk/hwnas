# 测量优先总账状态（中文伴随档）

本文件对应机器生成的 `status.md`，原件 SHA256 为 `de32009103f515336bd799f6c476ea78a1146e09328c4dd545cb807ca1937d74`；2026-07-18 最终复审生成的 `status.json` SHA256 为 `c889d9c2f0e61cbea3a438c3e75e5454c6e208c7875265ce4c631835f7e63f7e`。

- 总体状态：`IN_PROGRESS`（进行中）。
- G0 协议：`PASS`。
- G1 准确率基线：`PASS`，45/45 项检查通过。
- G2 硬件测量：`PENDING`。
- G3 搜索：`FROZEN`。
- G4 INT8 板级验证：`PENDING`。
- 功耗：`NOT_MEASURED`。
- G5 声呐消融：`PAUSED`。

## 当前阻塞项

G2 尚缺四个冻结的独立完整网络 probe、至少八个语义安全完整网络样本、完整的区间筛选质量门和候选 HLS shortlist 覆盖。G4 尚缺 PTQ/QAT 准确率、HLS 位精确一致性、完整 outer validation 板测、零数值不一致及无缺失板测样本。功耗尚未通过外部仪器 CSV 验收；G5 的去噪和边缘算子仍暂停。

## 证据边界

软件门的实现或通过不等于硬件实验结果。缺失 csynth、route、COM5、AV7K325 板测或外部功率计证据的结论继续保持 `PENDING`、`FROZEN`、`PAUSED` 或 `NOT_MEASURED`。
