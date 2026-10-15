# 声呐联合选择 V2

独立于旧 `sonar_joint_v1`；复用原架构、真实训练、PTQ 和精确成本接口。
旧代码、冻结、远端环境和结果不覆盖。

这里 V2 是流程版本，网络仍是旧 V0.1 `sonar_candidate`。L1/L0 轻量
neck/head、V0.2 和 elastic pilot 尚未接入；旧成本筛查不能代表这些新模型。

每个预算 40/60/80 都执行当前预算 checkpoint 的三个 PTQ profile；校准仅用
冻结 train 前 64 图，精度仅来自 valid。候选是架构×量化×映射，训练按架构
去重。先筛硬件，再按量化 mAP 与四项硬件成本的非支配层晋升，保留临界
候选，并用同预算身份和量化协议进行弃选复核。最终三种子确认继续使用
原精度下降与相对 W8A8 改进门槛；早期单种子不视为正式量化准入。

`configs/search/sonar_joint_v2.yaml` 绑定已有约束来源：检测查表筛查 120 ms；
LUT≤161280、DSP≤1209、BRAM18≤436。算子峰值/HLS 串行和只用于前置
筛查；整网布局布线含 shell、板端 25 ms、动态功耗/能量、测量质量门槛
独立保留在部署预算及原始预登记文件中。`deployment_budget.assess` 只检查
数值门槛，不替代整数精度、参数包、映射、bitstream 和原始板返回的准入。

2026-10-06 本地实际预检：2592/2592 成本可查，但 **0 个通过预算**，全部
超过 120 ms。最快串行估计为 4509321.045 μs（4.509321 s）。保留这个负
结果；不放宽约束、不自动运行新训练。当前映射需先取得符合预算的测量。

以下预检只读既有源/成本，并将报告写入新的空目录：

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_joint_v2/run.py plan --output results/NEW_V2_PREFLIGHT
```

`freeze` 需要另一个空目录，绑定配置、源、train/valid、成本和源 ZIP。
没有硬件合格候选时，在读取训练数据或启动训练前拒绝冻结。已有合格成本
及相应执行授权后，`run --resume` 复验绑定并继续任务；任务收据验证每个
预算 checkpoint、校准、profile 和结果。当前不提供可启动的合格 freeze。

完成软件筛选后生成 `deployment_queue.json`，固定硬件 seed42、保存候选/
合同/权重/校准/映射身份，`launch_hardware=false`。现有 B1/C/D 部署白名单
和 W8A8 串行执行器不能无适配接收任意新候选/混合精度/优化映射。

详细现状和既有板测证据见 `docs/SONAR_JOINT_V2_IMPLEMENTATION_20261006.md`。
测试使用合成夹具，验证编排和拒绝条件，不产生科研精度或硬件测量。
