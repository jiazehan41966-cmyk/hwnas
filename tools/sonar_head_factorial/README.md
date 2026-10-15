# E0 / YOLOv9-T 检测头析因实验：稳定评测版

配置：`configs/sonar_head_factorial_20261012.yaml`。新版本为
`sonar_head_factorial_20261012_02_stable`，固定输出
`results/sonar_head_factorial_20261012_02_stable/remote/`。
历史 `_01` 结果保持原样，不填入本次实验的任何格子。修改前源码与 SHA256
保存在 `results/sonar_head_factorial_20261012_02_stable/source_before/`。

| backbone + neck | obj×softmax | 质量感知方案 |
|---|---|---|
| E0，原生 P2–P5 | seeds 42/43/44/45/46 | seeds 42/43/44/45/46 |
| YOLOv9-T 主分支，原生 P3–P5 | seeds 42/43/44 | seeds 42/43/44 |

预先固定 16 次从零训练，取消自适应增加 seed 的规则。四格使用同一台服务器的
同一 GPU1、解释器和已有 YOLO 依赖 overlay，依次运行。同一 backbone 和 seed 的
两种头保持相邻，完整记录初始化和实际增强数据的配对证据。YOLO 与 E0 的结构效应
和交互项只使用共同 seeds 42–44，E0 换头主比较使用全部五个配对 seed。

每次 80 轮、batch4、640 灰度输入、AdamW(lr=.001, wd=.0001)、FP32，复用冻结的
增强、采样、解码、NMS 和 101 点验证器；不使用 EMA、学习率调度或梯度累积。
YOLO 将增强后的同一灰度张量复制为三通道。train330 / valid99 的图像、标签和划分
均保持原样；test 不挂载。

主指标为 **mAP50–95**。稳定口径是每个 seed 第 **76、77、78、79、80** 轮各自
mAP50–95 的算术平均，不从这五轮中择优。每轮保存完整 99 图的预测和 GT：
`formal/<backbone>__<head>/seed_<seed>/epoch_predictions/epoch_076_valid_records.json`
至 `epoch_080_valid_records.json`，相邻 `epoch_*_receipt.json` 记录图号、GT 身份、
来源绑定和预测 SHA256；其路径和哈希也写入 checkpoint 内的 history。
准备阶段直接通过既有冻结 valid loader 读取原始 GT 并生成规范化哈希，不引入新的
坐标转换。该哈希绑定执行清单、run binding 和每份缓存记录；漏掉、移动或改类的 GT
不能只靠“五轮彼此相同”通过验收。

best checkpoint 仍按 valid mAP50–95 的最早最大值保存，但只作次要对照；第 80 轮
结果与旧式 `(AP50+AP75)/2` 也只作诊断列，均不能替代主稳定口径。
任一晚轮记录缺失、图号不是完整唯一 99 张、来源或内容哈希变化、预测无法复现
该轮日志，稳定评估与正式完成均失败。显式 `--resume` 在恢复之前核查已提交的
晚轮文件，不用 best、终点或日志均值补造预测，也不覆盖冲突的未提交晚轮文件。
`latest.pt` 是原子提交点；意外中断留下的逐轮日志只在它是 checkpoint 历史的精确
前缀时允许修复。此前所有有效预测文件保持不变。

额外验证视图绑定
`results/experiment0_similarity_supplement_20261012_01/subset_contract.json`
及其来源文件哈希：完整 valid99 为主，排除 10 张跨划分相似候选后的 valid89 仅作
辅助敏感性分析。分别报告两套稳定结果，同样从每个 epoch 的逐图预测重算 AP，
不能直接筛选已聚合 AP，也不重新选择 checkpoint。相似候选不等于已确认重复或泄漏。

**D17 必须同时满足两个条件：**完整 valid99 上 E0 五个配对 seed 的稳定均值提升
`>=0.03`（3 个百分点），且配对 seed＋图像的分层 bootstrap 95% 区间下界 `>0`。
“所有 seed 都为正”不能替代置信区间条件。bootstrap 固定 2000 次、随机种子 20261012，
每次成对重采样训练 seed，再成对重采样验证图像；同一图像样本用于两种头及全部五轮，
逐轮重新计算 AP 后取平均。另报告只重采样图像的区间，明确它不包含训练 seed 波动。
valid89 的方向一致性单列，不擅自增加或替代 D17 门槛。

稳定统计实现位于 `tools/sonar_head_factorial/stable_analysis.py`，训练入口先核验全部
16 个唯一正式运行及其来源、环境、初始化、完整历史、晚轮预测，再委托该模块汇总。
记录每轮、每 seed、每格均值及标准差、E0/YOLO 换头差、共同 seed 的结构效应与交互项；
保留负结果。这仍是既定验证集上的开发决策，不是独立测试或跨采集序列泛化证明。

质量感知方案使用固定作者源码的单分支 DDetect＋TaskAlignedAssigner（任务对齐分配，
topk10/alpha.5/beta6）＋质量软标签 BCE＋CIoU＋DFL，不带 auxiliary/PGI 分支。
比较对象是头、损失、分配和评分的组合；结构因素包含骨干、颈部、特征尺度与容量差异。
不从这些结果推断 FPGA latency（推理延迟）、资源或功耗。

D17 达标只记录“允许后续按计划安排”，不自动改变默认头、不启动 NAS 或其他研究运行。
**D01 只由实验4提供证据。** 当前算子池与融合的小改动继续暂停。

## 本地检查和远端执行入口

在仓库根目录执行本地协议检查；这些测试不训练模型：

```powershell
.venv_cuda/Scripts/python.exe -B tests/test_sonar_head_factorial_protocol.py
.venv_cuda/Scripts/python.exe -B -m pytest -q tests/test_sonar_head_factorial_stable_analysis.py
```

远端传输脚本使用独立 `_02_stable` 名称；`stage` 拒绝已有目标目录。执行范围以用户
对本轮任务的授权为准，文档列出命令不表示已经执行或自动获准：

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py probe
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py stage
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py preflight
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py status
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py collect
```

正式队列要求实际四格 batch4 前向/反向及 valid4 smoke、共同环境锁和独立审查全部通过；
不能只凭 import 或本地单元测试宣称可训练。独立审查需绑定远端执行清单 SHA256。

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py upload-review
.venv_cuda/Scripts/python.exe -B tools/sonar_head_factorial/remote.py launch
```

每次训练保存源文件、配置、数据和环境绑定、初始权重哈希、全部逐轮指标、best/latest
权重及晚轮预测。通过实际进程和首轮日志核验启动，不把预检或部分 epoch 当成正式完成。
凭据只使用既有本机 Credential Manager 传输桥，不进入配置、命令参数或输出。
