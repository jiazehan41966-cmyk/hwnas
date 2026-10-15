# L1 / YOLOv9-T 三组控制实验

本实验只使用冻结的 train 330 / valid 99，不读取 test 的性能。统一灰度
BILINEAR 640、既有增强与类均衡采样、batch 4、AdamW 0.001、weight decay
0.0001、80 epoch、无 scheduler/AMP/EMA。seed 为 42；按 valid mAP50_95
最早最佳 epoch 选 checkpoint。三组分别是 L1 从零、作者 YOLOv9-T 从零、
作者 YOLOv9-T COCO 迁移。

作者源码为 `reference/yolov9`，固定 commit
`5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`。使用原生 DualDDetect 和
`utils.loss_tal_dual.ComputeLoss`。灰度完成增强后复制三通道。保持作者损失
与标签分配；训练预算、增强、采样及后处理采用本项目冻结条件。这是统一
预算控制实验，并非作者完整 COCO 训练 recipe 的复现。

作者公开的 Tiny checkpoint 是 `yolov9-t-converted.pt`。预训练组按明确
语义映射迁移 backbone、main neck 和 main regression head；辅助分支及
两类分类塔中不匹配的张量保持随机。两组 YOLO 使用相同的双分支训练结构。
每个运行的 `initialization_identity.json` 保存全部加载键、映射与未加载键。
不得称为完整双分支 COCO 初始化。

预测使用主分支原生 DFL 解码和 sigmoid 类别分数，按最大类别取每位置一条
框；统一 global top500 → class NMS 0.5 → global top100。AP 不做 0.5
分数过滤；P/R 在 score 0.5、IoU 0.5 报告。Params/MACs 同时保存双分支
训练图和裁去辅助支的单分支推理图，两者经过数值一致性检查。

本地解释器 `.venv_cuda/Scripts/python.exe` 复用既有 Torch/NumPy。
新增作者依赖位于 `results/sonar_yolov9_baseline_20261005/env_overlay`，
train.py 自动加入该目录；依赖规范和版本列表分别在 `environment_spec.json`
与 `env_overlay_lock.json`。关闭作者自动安装，不重建共享环境。

在项目根目录逐条运行，首次准备必须在源码稳定之后；重复准备只允许内容
完全相同。正式训练不覆盖旧运行，故障保留证据并停止队列。

```powershell
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\train.py prepare
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\train.py verify
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\train.py smoke --group L1_scratch --device cuda
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\train.py smoke --group YOLOv9_T_scratch --device cuda
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\train.py smoke --group YOLOv9_T_COCO --device cuda
```

独立审查必须核对三个真实 GPU smoke、最终源码冻结、数据身份、主支数值
一致性与 COCO 部分迁移范围，然后写出绑定 manifest SHA 的
`independent_review.json`。启动脚本不自行生成审查 PASS。

```powershell
.venv_cuda\Scripts\python.exe tools\sonar_yolov9_baseline\launch.py --gpu 0
```

Windows WDDM 的利用率包含桌面绘制；Windows 启动要求当前显存占用低于
500MiB且剩余显存能覆盖真实batch4 smoke峰值。Linux另要求利用率不超过10%。
Windows 后台进程隐藏窗口，三组在同一 GPU 顺序运行。查看 `queue_status.json`
和 `formal/<group>/seed_42/status.json`；日志在各组 `stdout.log`。
每个 epoch 保存 latest checkpoint、完整 RNG 与采样器状态。单个运行
故障后的显式 `train --resume --group ...` 仅恢复其自己的 epoch 边界，
不自动重启队列。完成后生成每组 `result.json` 与 `comparison.json`。

两批 smoke 只验证接口、损失/梯度和显存。80 epoch 的单种子对照不能单独
定位 backbone 或证明数据不可学。若三组都低，下一步需训练小样本过拟合
检查并核查收敛；当前结果不能被当成 FPGA 延迟、资源或功耗证据。
