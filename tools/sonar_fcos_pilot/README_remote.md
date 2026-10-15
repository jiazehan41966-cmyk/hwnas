# 既有服务器准备与核验

前六条命令只准备服务器上的环境和模型。后面的数据传送与正式启动命令有独立门槛：当前数据没有通过“同一采集序列不能跨集合”的核验，因此不会启动正式训练。模型核验使用人工生成的图像，不能当作检测成绩。

服务器目录：`/home/jiajia/experiments/sonar_fcos_pilot_20261008`。复用既有 Python 解释器，额外的评估包单独放进此目录的 `vendor`，不修改共享环境。使用物理显卡 1，程序内显示为显卡 0；其他显卡和任务保持原状。

在项目目录 `E:\1\hwnas\hwnas` 的 PowerShell 中依次运行：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote probe
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote stage
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote environment
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote smoke
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote status
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote fetch
```

- `probe`（查询）：检查服务器版本和显卡当前占用。
- `stage`（传送）：只传送这次新增的代码、配置和经过哈希核对的官方初始参数。哈希是文件内容的指纹，用来检查传送前后是否一致。不会传送历史网络、实验参数或数据。
- `environment`（环境准备）：安装 `pycocotools`（读取和评价目标检测标注的工具）到独立目录，并执行一次实际计算检查。
- `smoke`（运行前检查）：在 640 像素图像上检查普通卷积和扩张卷积方案，完成前向计算、反向求梯度和一次参数更新，同时核对八种方案的初始参数完全一致。这里的更新不计入正式实验。
- `status`（状态）：查看已有报告与显卡占用。
- `fetch`（取回）：把报告按原始字节取回 E 盘，逐项检查文件指纹。

`stage` 会寻找已经下载的官方参数。如果默认位置不存在，可以明确给出路径：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote stage --weight-path F:\sonar_fcos_pilot\weights\checkpoints\fcos_resnet50_fpn_coco-99b0c9b7.pth
```

远端检查实际执行的命令为：

```text
/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python -m tools.sonar_fcos_pilot.model --output-dir /home/jiajia/experiments/sonar_fcos_pilot_20261008/results/sonar_fcos_pilot_20261008/remote/model_preflight --cache-dir /home/jiajia/experiments/sonar_fcos_pilot_20261008/weights/torch
```

桥接程序设置 `CUDA_VISIBLE_DEVICES=1`（只让程序使用物理显卡 1）与 `PYTHONPATH`（让程序读取这次独立安装的包）。无需在命令里填写密码；凭据由既有 Windows 凭据管理器读取。

通过时应得到 `ENVIRONMENT_AND_MODEL_WITNESS_PASS`（环境和模型计算检查通过）。这不代表数据核验或正式训练已经通过。独立复现检查尚未完成时，报告明确记录 `PENDING`（待完成）。如新增或修改代码，再运行 `stage`、`environment`、`smoke`，让新报告绑定新的文件指纹；已有正式训练记录或正在运行的本轮任务时，禁止重新传送代码。

本地证据在 `results/sonar_fcos_pilot_20261008/remote_transport`。`fetch_receipt.json` 记录最近一次取回的文件列表、指纹和快照位置；每次取回单独保存，不混用旧报告。

## 数据通过后，怎样在服务器运行

默认仍使用原计划的严格数据核验。若另选“按公开资料划分的探索方案”，必须先取得该方案的明确确认，保存确认记录，并让配置与清单指向同一份记录；未确认的草稿不能传送或训练。探索方案不会被写成“已经核实真实采集序列”。

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote data --config configs\sonar_fcos_pilot_v1.json
```

`data`（传送通过核验的数据）先在本地检查门槛。通过后才打包原始图像、原始标注和新清单，逐张核对图像与标注的指纹，再上传服务器。已排除的图片不上传，但保留排除记录。程序把清单中的本地路径换成服务器路径，保存原批准清单的指纹，并生成 `configs/sonar_fcos_pilot_remote_v1.json`（服务器使用的配置）。训练轮数、八种组合和最多十四次正式训练的预算保持相同。

**独立复现步骤：**代码最后修改完成后，先重新运行 `stage` 和 `environment`。另一位代理按本页原样运行 `probe`、`smoke`、`status`、`fetch`，核对结果，再把真实复核记录写到服务器 `results/sonar_fcos_pilot_20261008/remote/independent_doc_review.json`。复核记录必须写明 `status=PASS`（通过），以及当前源码、环境说明、环境版本报告、模型检查报告和初始化检查报告的五个指纹。对应字段是 `source_freeze_sha256`、`environment_spec_sha256`、`environment_lock_sha256`、`preflight_sha256`、`initialization_audit_sha256`。作者自己的检查不能代替这次独立复现。

数据、源码、模型检查与独立复现全部通过后，正式启动命令才可使用：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote launch --config configs\sonar_fcos_pilot_v1.json
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote training-status
.\.venv_cuda\Scripts\python.exe -m tools.sonar_fcos_pilot.remote fetch-training
```

- `launch`（正式启动）：再次检查通过的数据、原批准清单、源码和独立复现记录；显卡 1 已被占用、已有本轮任务或启动记录时拒绝重复启动。启动后，即使关闭本地窗口，服务器上的任务继续运行。
- `training-status`（训练状态）：查看当前训练轮数、完成方案、控制进程是否仍在，以及最后一段日志。
- `fetch-training`（取回训练结果）：只取回小型成绩表、说明、运行记录和最多十二张示例图。完整模型参数和完整日志保留服务器，参数文件的指纹记录在运行报告中。

当前可以运行 `data` 和 `launch` 来核验阻止路径；应得到 `BLOCKED_DATA`（数据条件没有通过），并显示 `ssh_invoked=false`（连传送数据和启动的服务器命令都没有执行）。这两条检查不会绕过数据门槛。
