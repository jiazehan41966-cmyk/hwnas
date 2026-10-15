# C01b：已确认的 C01 66 通道与无 DFL 6 通道比较

用户已明确选择沿用已完成的 C01：E0＋作者 DDetect，`reg_max=16`。P2/P3/P4/P5 每尺度**原始输出为 66 通道**，由 64 个分布回归值和 2 个分类值组成；解码后为 4 个框坐标和 2 个分类分数，共 6 通道。硬件末端 1×1 卷积的通道合同指原始输出，不能用解码后的 6 通道代替，也不能把历史 C01 改标成 `reg_max=8` 对应的 34 通道。

C01b 每尺度原始输出为 6 通道：4 个直接距离和 2 个分类值。保留 E0 backbone/neck、TAL 分配、质量软标签 BCE 分类损失、回归/分类隐藏宽度 64/48，以及同 seed 的共享层初始化。框回归改为 GIoU，删除 DFL；直接距离在训练和解码中均用 ReLU 保证非负。这是距离表示、DFL、ReLU 参数化及 CIoU→GIoU 的组合比较，不能把效果单独归因于关闭 DFL。

## 冻结训练与判断规则

- C01b 从头训练 seed 42、43、44，各 80 轮；与历史 C01 的同三个 seed 配对。批量 4、输入 640、FP32、AdamW（学习率 0.001、权重衰减 0.0001），无 scheduler、AMP、EMA 或梯度累积，workers=0。沿用现有服务器 venv/overlay，不安装依赖；仅挂载 train/valid，test 保持封存。
- 第 76–80 轮每轮保存完整 99 图预测和 GT、SHA256 与收据，共 15 份 C01b 预测。每个 epoch 独立重算 AP，先取五轮算术平均，再计算同 seed 的 `C01b−C01`，最后取三个差值的均值；不合并预测，不按 best 或末轮选择。
- 仅当 full99 的 **mAP50–95 差值均值 ≥ −0.005，且 AP75 差值均值 ≥ −0.005**，才选择无 DFL 的 6 通道；否则保留实际 C01 的 66 通道。恰好下降 0.5 个百分点包含在通过边界中，浮点比较容差为 1e−12。
- 配对 seed＋图像分层 bootstrap 为 2000 次，随机种子 20261013，报告 95% 百分位区间。同图抽样跨两方案、三个 seed 和五轮共享；置信区间仅报告，不追加选择门槛。89 图去相似子集仅作辅助。点估计门槛通过不等于统计证明非劣效。

## 复现接口与产物

在仓库根目录 `E:\1\hwnas\hwnas` 运行以下已实现的远端控制接口。`stage` 只允许首次创建隔离 namespace；`preflight` 为 GPU1 上的独立受控进程，顺序执行 prepare、合同测试和候选 smoke；正式队列每 seed 上限 2 小时且不自动重试。

```powershell
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py stage
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py preflight
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py preflight-status
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py collect
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py upload-review
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py launch
.venv_cuda/Scripts/python.exe tools/sonar_head_c01b/remote.py status
```

`upload-review` 默认读取 `results/sonar_head_c01b_20261013_01/independent_review.json`；该文件必须是检查真实远端预检后形成的最终独立审查，含 `status=PASS`、`independent=true`、`scope=C01b_66_to_6_readiness`，并绑定执行清单、预检完成记录、smoke gate 三个精确 SHA256。静态初审不能替代它。正式启动同时要求已确认的 `baseline_resolution.json` 和匹配执行清单的预检/smoke PASS；正式授权固定确认记录、审查、预检完成记录和 smoke 的哈希。

| 内容 | 路径 |
|---|---|
| 新合同配置 | `configs/sonar_head_c01b_20261013.yaml` |
| 判断规则与用户基线确认 | `results/sonar_head_c01b_20261013_01/PREREGISTRATION.md`、`baseline_resolution.json` |
| 已完成 C01 参考根 | `/home/jiajia/experiments/sonar_head_factorial_20261012_02_stable/results/sonar_head_factorial_20261012_02_stable/remote` |
| C01b 隔离 namespace | `/home/jiajia/experiments/sonar_head_c01b_20261013_01_revision2` |
| 远端运行产物（相对 namespace） | `results/sonar_head_c01b_20261013_01/remote` |
| 本地传输记录与收集证据 | `results/sonar_head_c01b_20261013_01/transport_revision2`、`remote_snapshot_revision2` |

首次预检因相对配置路径规范错误止于prepare阶段，没有启动正式训练。其服务器namespace与本地`transport`、`preflight_attempt1`记录保留；修正后在revision2重新冻结代码并预检，实验配置及判断规则不变。

训练入口 `train.py` 提供 `prepare / smoke / worker / queue / summarize / status / verify`，参数为 `--config`、`--output`（或 `--result-root`）、`--device`；`worker` 另需 `--seed`。正式 queue 完成全部三个 run 后自动 summarize，重算配对统计并生成根目录的 `c01b_summary.json`、`paired_analysis/c01b_analysis.json`，以及本轮 `D17_HEAD_CONTRACT.json` 和 `.md`。D17 补充记录真实逐尺度通道选择，不改写历史 D17 通过记录，不启动 NAS、HLS 或上板。

`collect` 只下载有大小上限且经 SHA 校验的 JSON/CSV/MD/LOG/NPZ，checkpoint 留在服务器。源码、控制器及预检通过本身不是性能结论；性能与通道结论必须以完整三 seed、全部逐轮预测及最终合同产物为证据。
