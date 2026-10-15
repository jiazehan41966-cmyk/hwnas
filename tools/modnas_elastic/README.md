# MODNAS 新版检测器本地适配

2026-10-07：新版软件接入通过；正式 FPGA 硬件搜索仍受成本标签缺口阻断。

接入对象为 `tools/detector_elastic_pilot/model.py` 的 E0/E1 四尺度检测器。
完整固定 B1 骨干保留，权重全部参与训练。没有启用旧 144 骨干候选枚举。
旧 E0/E1 pilot 的源码、配置和既有运行均未被本次适配修改。

## 本次验证结果

- `tests/test_modnas_elastic_{author,model,hardware,protocol}.py`：39 项通过。
- Dataset A 真实灰度 640×640 输入，CPU、4 线程、seed42、batch1、1 次优化。
  权重在 train 更新，架构在 valid 更新；3 个偏好条件均有有限非零梯度。
  验证架构更新前后的模型权重和 BatchNorm（批归一化）状态完全一致。
- 3 个偏好条件导出的独立静态模型与超网对应 one-hot（单一路径）输出误差均为 0。
  Params（完整模型参数量）与独立成本图一致；四层预测尺寸为 160/80/40/20。
- 导出后仅使用一个 train batch 验证批归一化校准接口，再以一个 valid batch
  检查解码及评价接口。这些数值没有最终重训精度的含义。测试集没有加载或评价。
- 正式性字段为 `formal_eligible=false`、`performance_claim=false`。
  本次没有完成搜索预算、独立候选重训、四数据集对照、HLS、布局布线或上板实验。

证据位于 `results/modnas_elastic_adapter_20261007/`：

| 工件 | 内容 |
|---|---|
| `source_manifest.json` | 官方核心文件固定提交、下载 URL、SHA256、适配说明 |
| `unit_tests.xml` | 39 项测试的实际结果 |
| `local_smoke_v1/execution_manifest.json` | 配置、源码闭包、train/valid 图像及标签哈希、环境 |
| `local_smoke_v1/smoke_report.json` | 实际单步更新、导出等价性及证据范围 |
| `local_smoke_v1/smoke_checkpoint.pt` | 超网、条件化网络、优化器、随机状态及源码绑定 |
| `local_smoke_v1/exported_candidate_smoke.pt` | 第一个偏好条件导出的诊断模型与 train 校准记录 |
| `local_smoke_v1/hardware_gap.json` | 每个候选的成本缺口和准确形状的补测任务 |
| `independent_review.json` | 独立代理静态审查及工件来源核验，注明未独立重跑 smoke |
| `artifact_validation.json` | 142 份源码及工件哈希核查、从保存的检查点恢复偏好架构和导出权重、成本门槛退出码 |

## 注册的新版候选空间

空间版本为 `elastic_neck72_v1`，具体配置见
`configs/modnas_elastic_adapter_v1.yaml`。保留四尺度、48 通道 neck（特征融合模块）、
膨胀比 3、深度 1、直接 1×1 检测头及原检测损失、锚框和解码协议。

| 架构变量 | 取值 |
|---|---|
| P2/P3/P4 三个自顶向下融合块卷积核 | 各自 3×3 或 5×5 |
| 整个 PAN（自底向上特征聚合）分支 | 关闭或开启 |
| 开启 PAN 时 P3/P4/P5 三个融合块卷积核 | 各自 3×3 或 5×5 |

关闭 PAN 有 8 个候选，开启 PAN 有 64 个候选，总计 72 个。
关闭 PAN 时，其卷积核变量从架构编号与概率计数中移除，避免重复候选。
架构包含完整检测头；Dataset A 的 2 类头为每尺度 21 通道。
模型和成本图支持其他正整数类别数，但当前运行入口严格绑定 Dataset A 的 2 类协议。
其余数据集的类别、冻结数据与训练协议接入尚未执行。

该空间是首版保守适配范围。正式比较前，Ours、Random Search 及其他方法必须
注册并使用同一空间；本次单独接入没有证明方法间公平比较已经完成。

## 保留与改变的作者算法

官方来源：https://github.com/automl/MODNAS

固定提交：`21ddf2fee3e34db8e5828ddf6749fe037704e181`。
作者源码位于 `reference/_local/MODNAS-core-<commit>`，保留原始字节并在执行前校验。
当前下载的是 7 个核心源码及元数据文件的依赖闭包，未下载完整仓库、预训练权重和数据。
没有批量安装作者整套 requirements。

| 部分 | 本地接法 |
|---|---|
| 偏好条件化 HyperNetwork（架构参数生成网络） | 保留作者离散偏好嵌入；6×5 输出改为 7×2 |
| 硬件条件化专家混合 | 保留作者 sin-softmax 混合；诊断硬件输入是 10 个零，不能称为测得硬件特征 |
| GDAS（可微离散架构采样） | 调用作者原采样源码；设置温度并加入单 epoch 调度保护 |
| MGD（多梯度聚合） | 调用作者最小范数求解器；保留对齐的参数梯度，增加单组/零梯度保护 |
| 多设备训练 | 改为单一 ZCU104 的多个偏好条件，这是工程适配，尚非多设备复现实验 |
| 超网和任务损失 | 使用新版检测器与现有检测损失；架构训练对 E0/E1 的检测损失施加 PAN 门控 |
| 硬件预测器 | 当前只有来源绑定的有限测量表期望接口，属于工程替代，尚未训练作者式多设备预测器 |

MGD 聚合的是各条件的标量化损失梯度；不能写成直接对精度和硬件目标各自梯度
进行作者式多设备训练。原作者 FW 求解循环没有递增迭代计数；当前单步诊断已收敛，
正式长时间运行前仍须处理收敛与运行上限。NumPy 2 的作者源码兼容警告目前不影响测试。

权重更新采用硬采样的选中路径，只有选中分支的批归一化统计被训练。
架构更新执行所需可微分支，但冻结模型参数并置为 eval，不使用 valid 更新统计量。
最终正式候选应统一从头独立训练，以 valid 按冻结规则选择；超网和单 batch 分数
不能填入主表的最终测试精度。

## FPGA 成本边界与剩余接入项

本次 Params 目标是可精确核验的软件诊断，完整模型范围为 460172–528572。
它不代表 FPGA 延迟、LUT（查找表）、DSP（数字信号处理单元）或 BRAM（块 RAM）。
全部 72 个候选的完整 FPGA 指标仍为 null，状态为 `BLOCKED_MISSING_NEW_SPACE_COST`。

当前能精确复用 20 种固定 B1 primitive（基础算子）的已有 HLS 证据。
需要补充 41 种新版算子：27 种卷积、3 种上采样、4 种加法、4 种残差加法、3 种池化。
准确的输入/输出形状和精度约束已经写入 `missing_measurement_tasks`。
该数字是独立算子任务数，不是训练或候选评价次数。

正式硬件反馈需要以下依赖按顺序闭合：

1. 新融合块、残差、PAN 和检测头的 W8A8（权重/激活各 8 位）整数及接口契约。
2. 相同 ZCU104 器件、2022.2 工具、5ns 时钟、映射及缓冲设置下的来源绑定测量。
3. 完整 72 候选成本表，包括逐节点源码、C-sim 输入输出一致性和综合报告哈希。
4. 固定搜索约束、偏好输入、标量化归一化、预算、候选选择及 seed42/43/44 重训协议。
5. 搜索、独立候选重训及最终整网部署验证。

外部表加载器会检查候选完整性、图/架构哈希、算子形状、平台、报告数值和来源。
当前接口的延迟口径是算子 HLS 延迟串行求和，资源口径是算子峰值；shell 和 DDR
代价未知。这一层即使将来标签齐全，也不能直接声称整网布局布线或板上可行。
测试中的合成成本表仅为临时 TEST fixture（测试样例），没有进入真实结果目录或指标。

## 本地命令与重现

在仓库根目录运行，使用现有 `.venv_cuda` 环境。以下是本次已执行的真实 smoke 命令：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_elastic.run smoke --output results/modnas_elastic_adapter_20261007/local_smoke_v1 --device cpu --steps 1 --threads 4
```

为保留已完成工件，再次执行请使用新的目录，例如：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_elastic.run smoke --output results/modnas_elastic_adapter_20261007/local_smoke_replay_01 --device cpu --steps 1 --threads 4
```

预检、成本门槛及测试命令：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_elastic.run preflight --output results/modnas_elastic_adapter_20261007/local_preflight_replay_01 --device cpu --threads 4
.\.venv_cuda\Scripts\python.exe -m tools.modnas_elastic.run hardware-check
.\.venv_cuda\Scripts\python.exe -m pytest tests/test_modnas_elastic_author.py tests/test_modnas_elastic_model.py tests/test_modnas_elastic_hardware.py tests/test_modnas_elastic_protocol.py -q
.\.venv_cuda\Scripts\python.exe results/modnas_elastic_adapter_20261007/validate_delivery.py
```

未提供测量表时，`hardware-check` 应返回 BLOCKED 和退出码 2。
补齐真实测量后可通过 `--cost-table <path>` 验证外部表；当前没有正式硬件搜索运行入口。
未启动后台队列、远程训练、HLS 或上板。

执行清单绑定源码、配置、作者文件以及 train/valid 图像和标签，复用输出目录时
拒绝源码、数据或环境漂移。诊断报告的计时仅覆盖更新、导出、单 batch 校准/评价及
保存工件，不包括初始化和来源预检，不得当作正式方法总搜索成本。CPU 实际运行
`GPU_h=0`；正式比较仍须完整核算方法专属预训练、预测器及筛选成本。
