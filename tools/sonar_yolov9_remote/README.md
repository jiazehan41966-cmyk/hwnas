# 服务器 COCO 微调单组

本次只启动 `YOLOv9_T_COCO`，seed 42、80 epoch、batch 4、640 灰度输入，
使用既有冻结数据、损失和评估器。服务器根目录是
`/home/jiajia/experiments/sonar_yolov9_baseline_20261005_remote_coco`，结果位于
`results/sonar_yolov9_baseline_20261005/remote_coco`，GPU 为物理编号 2。
不启动服务器 L1 或 YOLO scratch 正式训练，也不挂载 test。

服务器复用 `/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python`，
Torch 2.7.1+cu128、torchvision 0.22.1+cu128、NumPy 2.4.4。
新增依赖是 Linux 原生 `--target --no-deps` 安装，不修改共享基础环境。
远端实际规范和版本见 `remote_environment_spec.json` 与
`remote_environment_lock.json`。原 `environment_spec.json` 和
`env_overlay_lock.json` 仅作为本地冻结来源保留。

本地使用 Windows、Torch 2.7.0+cu126 和 RTX 3050 Ti，服务器使用 Linux、
Torch 2.7.1+cu128 和 RTX 5090。因此结果属于相同数据与训练协议的运行，
不能声称仅预训练条件变化或跨环境数值等价。

在 `E:\1\hwnas\hwnas` 根目录按顺序执行以下 PowerShell 命令。准备和
安装只执行一次；已完成的 smoke 与正式启动拒绝覆盖。

```powershell
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py probe
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py stage
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py environment
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py prepare
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py verify
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py smoke --group L1_scratch
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py smoke --group YOLOv9_T_scratch
```

独立环境验证者在以上步骤完成后，逐字执行下列尚未执行的命令：

```powershell
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py smoke --group YOLOv9_T_COCO
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py fetch
```

COCO smoke 必须实际完成两批训练共 8 图和 4 图验证，损失、梯度有限且模型
权重发生变化；验证主支裁剪图一致。检查三组 GPU smoke 与训练 RNG/图序
一致，manifest 与远端环境绑定一致，精确冻结源/配置、train330/valid99 和
官方 checkpoint SHA 校验通过。独立验证者检查证据后才写 review PASS。

审查文件 `independent_review.json` 至少包含 `status: PASS`、
`execution_manifest_sha256`、`remote_environment_spec_canonical_sha256` 和
`remote_environment_lock_sha256`；后两者分别是排序紧凑 JSON 的 SHA256 与
lock 文件原始字节 SHA256。审查上传到远端结果目录后才能执行：

```powershell
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py launch
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py status
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_remote\remote.py fetch
```

启动器核验 GPU 利用率不超过 10%、剩余显存覆盖真实 smoke 峰值，并再次
校验冻结输入、review、环境和重复进程。使用独立后台进程，输出日志、PID、
启动命令和显存快照。启动验收需要首 epoch 的 99 图验证、检查点和日志。

本地队列的 COCO 调度由 `local_handoff.py` 单次交接处理：仅 suspend 队列
调度进程，保留正在运行的 scratch 子进程；待其完整退出后清理队列父进程。
不改训练源码、配置或冻结 manifest。凭据只经既有 Credential Manager
传输接口读取，不写入任何研究输出。
