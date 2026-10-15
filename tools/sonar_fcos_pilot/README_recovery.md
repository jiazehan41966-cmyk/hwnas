# 声呐检测训练异常后的隔离恢复

这次恢复继续已有的 `AAA_seed42`：三处都使用普通卷积，随机编号为 42。
原来的 7 轮完整训练已经保存，复制并核验这些状态后，从第 8 轮开始。
原目录保留，不覆盖它的代码、数据、模型参数或日志。

原目录：`/home/jiajia/experiments/sonar_fcos_pilot_20261008`。

恢复目录：`/home/jiajia/experiments/sonar_fcos_pilot_20261008_recovery_v1`。

本地恢复记录：`results/sonar_fcos_pilot_20261008/recovery_transport/`。

使用原来已有的 Python（运行程序的解释器），只使用服务器上编号为 1 的显卡。
读取原始图片、划分清单和用户已确认的数据方案；读取原目录的私有依赖和公开初始参数。
不会改其他实验的运行环境。

## 恢复的检查范围

恢复前先检查原代码的保存副本和服务器上的原代码是否一致。
允许的计算代码变化只有 `train.py`（训练循环）的数值异常修复，以及新增的
`numerics.py`（数值重试工具）和 `recovery.py`（本恢复工具）。
网络结构、候选操作、数据划分、学习速度、总轮数和随机编号都保持原设置。
配置只改变输出和参数保存位置，并增加 `recovery_parent`（原运行的来源记录）。

将原 `last.pt`（最近一轮完整训练状态）和各个完整的最佳模型复制到恢复目录。
复制保存的验证预测、成绩及 7 轮历史。重新绑定的意思是把新代码和新配置的摘要
写入复制后的文件，原文件不改。摘要是由文件内容计算出的校验编号，文件变化时编号也会变化。

以下实际训练内容必须逐项保持一致：

- 模型参数。
- 优化器状态，即更新参数时已经积累的信息。
- 混合精度状态，即为了节省显存而使用较低精度计算时的数值控制状态。
- 随机状态，即下一轮继续使用的随机过程位置。
- 完整训练历史及已完成轮数。

恢复记录逐一列出原文件和复制后文件的路径、摘要，并记录保持不变的训练状态摘要。
`AAA_seed42` 按同一个候选续训，不增加第 15 个候选；每个候选仍共 30 轮，最多 14 个不同的组合与随机编号。
中断的第 8 轮会从头重算，额外计算开销单独记录，不能说完全没有重算。
最终报告必须写清：前 7 轮使用原实现，后续使用数值异常修复后的实现。

## 先检查，再恢复

在本机项目目录 `E:\1\hwnas\hwnas` 中，用 PowerShell（Windows 的命令窗口）依次运行。
前六条命令不会启动正式训练，独立检查人员也按这六条原样复现。

1. 将修复后的代码固定在新目录。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery stage
```

2. 验证并复制 7 轮完整训练状态。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery migrate
```

3. 用真实显卡检查网络计算及数值异常重试；使用人工构造的数据，不算正式训练。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery preflight
```

4. 重新核验数据、模型参数、来源摘要及上述显卡检查。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery check
```

5. 查看状态，检查尚未正式启动。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery status
```

6. 下载轻量检查材料，不下载大型模型参数。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery fetch
```

代码固定后再修改源文件会使检查拒绝；重新固定不能覆盖已经保存的恢复版本。
已有同一份代码时，重复前两条只核验并复用，不重复改写保存状态。

## 独立检查与实际启动

独立检查人员在 `independent_doc_review.json` 中写入 `status: PASS`，以及六条命令的实测证据。
必须包含当前 `check` 输出的七个摘要字段：
`source_freeze_sha256`、`environment_lock_sha256`、`preflight_sha256`、
`initialization_audit_sha256`、`numerics_witness_sha256`、`migration_sha256`、`check_sha256`。
它们分别绑定代码、环境、网络计算、所有候选的初始参数、数值重试、迁移和最后核验。

将真正完成的独立报告上传，不允许拿旧报告代替。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery review-upload --review-file results/sonar_fcos_pilot_20261008/recovery_transport/independent_doc_review.json
```

确认独立报告通过后，运行下面命令正式续训。脚本再次核对文件摘要和原数据，
检查没有旧训练进程且编号为 1 的显卡空闲。它拒绝重复启动。

```powershell
.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.recovery launch
```

后台继续原流程：先训练完 `AAA`，根据原约定分析错误，再决定是否比较其他组合。
`status` 和 `fetch` 可以继续使用；`check` 是正式恢复前的核验，续训后参数正常变化，不能再用它核验第 7 轮参数。
