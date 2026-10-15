# MODNAS 与 Ours：正式 ZCU104 对比的硬件准备

本轮用户已选择正式 ZCU104 硬件约束比较，先完成硬件门槛。
本轮独立结果目录为 `results/modnas_vs_ours_20261007/`，不复用或覆盖旧运行。

## 已完成与正在准备的真实实验

2026-10-07 已在原有用户服务器上完成固定 B1 stem（首个卷积）的三项独立
数值仿真与 HLS 综合：PE8/16/32（输出通道并行度）、SIMD1（输入通道并行度）。
输入为完整 1×640×640，输出为 32×320×320，保持 W8A8、原整数规则、
256 位输出接口、ZCU104 器件和 5ns 时钟。此处使用带来源哈希的合成整数 witness
验证算子实现，不使用训练或测试图像。

每项依次执行 C-sim 和 CSYNTH，队列并发为 1、每项上限 900 秒。
仅处理本轮新目录下的进程与文件；没有修改已有环境、停止既有训练或访问板卡。
实际启动收据：`mapping_probe/remote_launch.json`。
准备源码、输入输出见 `mapping_probe/preparation_manifest.json`；全部输入、权重、
偏置、量化参数及参考输出均绑定 SHA256，并经独立代理完整整数回放。

三项输出的全部 3,276,800 字节均与整数参考一致，原始 XML 复核后得到：

| stem 映射 | HLS 延迟（μs） | LUT | DSP | BRAM18 | 时序/资源筛查 |
|---|---:|---:|---:|---:|---|
| PE8/SIMD1 | 688323.53 | 3388 | 11 | 11 | 通过 |
| PE16/SIMD1 | 612073.19 | 3555 | 19 | 19 | 通过 |
| PE32/SIMD1 | 566735.15 | 3818 | 35 | 3 | 通过 |

三项均未通过 120ms 串行代理延迟预算。完整来源与负结果见
`mapping_probe/verified_results.json`。最好 stem 替换后，其余已知骨干映射
保留原样的部分和仍为 2,105,006.43μs，尚不是完整候选延迟。
报告中共享的六周期 `finish_r` 限制了并行收益。单项 PE8 后续试验只增加
两条数值函数综合 INLINE（内联）指令，已实际完成并取回复核：197848.37μs、
LUT11173、DSP32、BRAM18=11，完整整数输出仍一致，共享模块已消失。
该点仍超 120ms，因此进一步实际测量 PE16/32 两项同数值内联版本。

PE16/32 同数值内联测量现已实际完成，输出逐字节一致：

| 内联 stem 映射 | HLS 延迟（μs） | LUT | DSP | BRAM18 |
|---|---:|---:|---:|---:|
| PE16/SIMD1 | 105111.47 | 20004 | 64 | 19 |
| PE32/SIMD1 | 49469.33 | 37237 | 128 | 3 |

两个单算子均通过时序、资源和 120ms 延迟筛查。最佳 stem 代入后，其余已知
骨干保留原映射的部分和仍为 1,587,740.61μs，整网仍未通过预算；必须继续
优化其余骨干，而不是仅补 41 项新 neck 成本。完整六项测量、负结果及下一步
统一记录在 `results/modnas_vs_ours_20261007/campaign_status.json`。

这是正式比较的前置硬件实现实验。**MODNAS 和 Ours 的网络搜索尚未启动**，
当前没有新精度、完整候选硬件指标或方法优劣结论。

## 为什么先处理骨干映射

`hardware_lower_bound.json` 实时核验了原 H3 库中与新版模型完全一致的算子证据。
在当前 `reference_min_latency` 与算子串行求和口径下，固定 B1 的 20 个已知算子
部分和为 2,717,069.97μs，超过既有检测搜索预算 120,000μs 约 22.64 倍。
stem 单项为 1,178,798.69μs。新增 neck 的未知延迟是非负项，无法降低这个部分和，
因此暂缓仅为当前不合格映射补齐 41 种新 neck 算子的测量。

该证明限定于声明的 HLS 串行代理指标，不是完整网络的实测延迟，也不是板上不可行证明。
新映射仍须实际测量，不能按 PE 比例推算或预填加速比。stem 加速也不代表其余
19 个已知算子以及新版 neck/head 已通过预算。

## 方法和协议接入状态

- MODNAS 新版架构 72 空间的软件接入已有记录，旧 Params 单步诊断保留原样。
  新增 `safety.py` 包装原作者 MGD 求解器，以外部迭代和 Python 行边界时间保护
  防止原作者循环计数缺失造成无限运行。收敛时结果与原求解器严格一致；异常直接
  中止，绝不以平均梯度代替。原生 Torch/CUDA 阻塞另需进程级总时间限制。
- 用户已确认 Ours 使用 `sonar_joint_v2`：独立候选训练、40/60/80 预算及量化
  Pareto 晋升。原入口仍绑定旧 144 网络；本轮按新版 72 架构另建后端，保留
  架构×量化×映射、边界保留、弃选复核和三种子确认机制。
- 新后端 `joint_v2_elastic.py` 的独立训练、断点恢复、train64 校准和 full-valid
  量化评估接口已实现，但未运行。直接复用原 Pareto 晋升和三种子准入函数，
  72/54/36 阶段人数及 E0/E1 弃选复核显式登记。32 项契约测试通过。
  最新预检位于 `joint_v2_newspace_preflight/reviewed_v2/plan.json`：真实三档
  软件来源核验通过，1296 个完整硬件组合仍未知，没有合格组合；计费和共同
  启动 driver（执行入口）尚未实现，配置标志不能绕过训练拒绝门。
- 旧 H2 PTQ 会改写旧 FPN 前向，不能直接用于新 TD/PAN、MBConv 残差和 ModuleDict。
  本轮新增 `quantization.py` 用实际新版 DAG（运算图）适配校准及数值参考。
  浮点 fakequant（量化模拟）与真正整数参考分开验证；局部测试不视为整网 FPGA 通过。
- 真实 Dataset A 冻结 train64 已完成 E0/E1 的 W8A8 软件核验：首张真实 train 图
  47/68 节点、35,856,400/40,291,600 元素均零整数差异。E0 的原 FP32 BN
  折叠 1e-5 门失败并保留；新登记的 FP64 配对数学参考门通过，不意味着原门
  已通过。权重仍是一步诊断，无精度保持或硬件部署结论。来源快照与记录见
  `ptq_real_dataset_a/attempt_v2/real_data_ptq_gate.json`。
- 独立 `quantization_mixed.py` 已完成三档软件适配，W8A8 原模块和 gate 保持原哈希。
  真实 train64 的 E0/E1 × 三 profile 六组全节点整数比较均零差异；两个 head16
  profile 各显式增加四个 body8→head16 重定标节点，分别使用独立 H4 头合同，
  不改 H2 INT8 合同。实际结果和 23 个执行源快照见
  `ptq_real_dataset_a/mixed_head_v1/real_data_mixed_ptq_gate.json`，说明见
  `ptq_real_dataset_a/MIXED_PROFILE_SOFTWARE_GATE.md`。这些转换节点、16 位 head
  成本和整网帧/loader 的硬件证据仍未知；诊断权重不证明量化精度保持。
- 联合 V2 必须保留架构×量化×映射的核心机制；当前 MODNAS 只有架构变量
  和 W8A8/单映射接口，双方共同决策维度、总 GPU·h、资源档位和最终重训协议
  仍须冻结。不能删除 Ours 核心机制，也不能用随机搜索替代 Ours。
- `joint_encoding.py` 已完成异构 23-logit（输出分数）编码和原 GDAS 采样：
  七个二选一架构变量、三选一量化、六选一映射，共 1296 个有效组合。
  4 项测试验证真实梯度及关闭 PAN 的合并；混合精度搜索损失、完整硬件表、
  设备描述及正式 driver 尚未接通，不称为完整联合 MODNAS 搜索。

共同范围、数据分割边界和待冻结预算见 `PROTOCOL_DRAFT.md`。

## 正式网络搜索的启动门槛

1. 至少有满足相同资源与延迟筛查预算的真实新映射。
2. 新版卷积、上采样、加法、残差、池化及检测头的数值/接口合同和来源绑定成本齐全。
3. Ours 的新版后端与 MODNAS 的共同合法空间、量化/映射维度通过核验。
4. 对两方法冻结相同数据、搜索总计算预算、三独立种子和最终重训/验证选择规则。
5. 方法专属超网、条件化网络、预测器和筛选费用计入搜索；公共 HLS 建库另报。
6. 搜索用 valid，权重训练与量化校准用 train，最终测试封存；保存同一候选架构哈希、
   权重和导出配置，最终部署逐个核验，不替换为更容易部署的网络。

当前不自动放宽 120ms 代理预算或 25ms 板端预算，也不将 Params 当硬件代价。

## 复核实际硬件队列

在仓库根目录执行。凭据使用现有 Credential Manager（凭据管理器）传输。

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_comparison.remote status --attempt stem_parallel_v1
```

只有队列完成后，第一次取回完整证据：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_comparison.remote fetch --attempt stem_parallel_v1
```

已执行的部署命令保留在启动收据中；不能重复部署同一运行目录：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_comparison.remote stage --attempt stem_parallel_v1
.\.venv_cuda\Scripts\python.exe -m tools.modnas_comparison.remote launch --attempt stem_parallel_v1
```

worker 只在全部字节一致且综合报告平台/时钟匹配时记录仿真和综合 PASS。
5ns 时序与 LUT/DSP/BRAM18 资源筛查是另两个字段；PASS 不表示整网可部署。
原始 HLS 日志、XML、整数输出和源哈希随证据保留，失败和超时也保留。

目前三批有限队列均已结束并取回证据。汇总当前记录，不发起新任务：

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.modnas_comparison.status
```

单项内联批次为 `stem_inline_v1`，两项内联批次为 `stem_inline_scale_v1`，
对应 `mapping_inline_probe` 和 `mapping_inline_scale_probe`。各目录中的
`remote_stage.json`、`remote_launch.json`、`remote_evidence/status.json` 和
`verified_results.json` 保留实际命令、进程收据、时间和原始测量。
