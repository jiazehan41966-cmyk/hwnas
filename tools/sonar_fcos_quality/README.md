# 实验2：FCOS / UATD 质量评分与边界分布回归

这是独立的六次训练：AAA 与 QFL_DFL 各跑 seeds 42、43、44，每次30轮。
配置为 `configs/sonar_fcos_quality_20261012.json`。原 AAA 代码、配置、参数及历史结果不改写。
服务器目录为 `/home/jiajia/experiments/sonar_fcos_quality_20261012_01`，
输出在其中的 `results/sonar_fcos_quality_20261012_01`。

AAA 直接使用已有 torchvision FCOS ResNet50-FPN 构建函数。
QFL_DFL 保留同种子的骨干、FPN、分类塔和回归塔初始张量及 FCOS 样本分配；
以 QFL（质量焦点损失）的 IoU 软标签训练分类分数，移除中心度分支，
用 DFL（分布焦点损失）学习四条边的离散分布。其余训练设置继承已完成的 AAA 恢复配置。
该版本是将 QFL/DFL 接入 FCOS 的固定方案，不是完整 GFL 检测器复现。
QFL beta=2，回归每边17个取值0–16，DFL权重0.25，GIoU权重1。
超出边界分布范围时明确报错，不静默截断标签。公式来源为
[GFL 原论文](https://arxiv.org/abs/2006.04388)。

训练沿用640输入、batch1、累计4步、SGD学习率0.0025、momentum0.9、weight_decay0.0001，
20/25轮降学习率，1轮warmup、起始因子0.001，AMP及已有溢出组重算机制。
不改共享环境，复用原FCOS依赖目录、公开COCO初始参数及已确认的数据清单。
训练6070张，验证1517张（2692个目标）；不加载测试集。
划分是历史已确认的公开数据探索方案，未核实采集序列隔离。

按原 AAA 规则选择验证 COCO mAP50–95 最高的最早权重；六次均采用相同选择规则。
主表在该权重上报告 AP75、AP85、mAP50–95；同时在每次 summary.json 保留固定第30轮指标。
AP单位为0–100百分点。三个病因指标沿用历史完整验证诊断口径：

- 好框：同类框与目标的 IoU≥0.75；阶段覆盖不是一对一匹配召回率。
- NMS首次丢好框：raw→score>0.001→每层top1000→类别内NMS0.6→top100→合法前景框，
  目标第一次从有好框变为无好框的阶段为NMS。
- 原始无好框：全部原始点解码框中，没有同类IoU≥0.75框的目标数。
- 错类误检：最终score≥0.25预测，与异类GT的IoU≥0.5、与任一同类GT的IoU<0.5的预测框数。

历史333/448/577仅作参考。六次全完成后，按种子配对的 AP75 平均增量≥2个百分点，
且新版本NMS首次丢好框平均数相对本次AAA平均数下降≥25%，才标记PASS。
分母为0时不视为下降门槛通过。负结果完整保留。
最终生成 `branch_results.csv`、`branch_results.json`、`支线结果表.md`，包含均值、样本标准差和逐种子差值。
按六次最佳验证mAP中的最大值导出 `pretrained/uatd_fcos_best.pt` 及来源清单。
这是FCOS ResNet50-FPN权重；若双类网络架构不同，需要单独核对可迁移层，不承诺整网直接加载。

## 启动与独立核验

在项目根目录的PowerShell中运行。凭据由已有本机Credential Manager桥读取。

```powershell
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote probe
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote stage
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote preflight
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote status
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote collect
```

preflight在独立目录执行本支线测试、真实数据准备和两个方案各四张训练图的一次梯度累计更新，
以及两张验证图的追踪预测。它不生成正式训练权重。
独立审阅人员按本页原样执行 status 和 collect，核对预检命令全部成功、
真实GPU更新/追踪一致性及 `smoke_gate.json.status=PASS`，审阅模型、调度、数据和判断门槛。
复核记录必须绑定当前 `execution_manifest.json` 的SHA256；不得由作者自报独立PASS。
通过后上传独立报告，再启动有限队列：

```powershell
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote upload-review
.venv_cuda/Scripts/python.exe -B -m tools.sonar_fcos_quality.remote launch
```

全部配对运行只用物理GPU0，GPU1永久留给主线。GPU0有其他计算进程时等待，
已运行时检测到外部占用则请求本支线退出；不终止外部进程。
每轮完成后保存模型、优化器、AMP及随机状态。让出GPU时放弃未完整提交的一轮，恢复后重算，
不丢弃训练样本或声称没有额外计算。`PRIMARY_GPU_REQUEST` 文件让队列释放GPU并等待；
`STOP_QUEUE` 用于退出本支线队列。具体文件位置和状态见调度回执。
队列到六次完成或遇到错误后退出，不创建周期性任务。

状态与轻量结果继续使用 status / collect 获取。只将实际完成的训练写入支线成绩表；
预检、启动和中间轮次均不能当作完成结果或通过证据。
