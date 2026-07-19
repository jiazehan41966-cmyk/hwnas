# 多目标 Aging Evolution 设计与 RL 对比协议

## 结论

本仓库新增 `aging_evolution`（多目标进化搜索兼容名称）方法，但保留原 RL 搜索器及历史配置。新方法不把分类质量、声呐鲁棒性和 FPGA 部署代价压缩成单一 reward（奖励值），而是采用硬可行性约束、Pareto 非支配排序、拥挤距离、双亲交叉和变异进行搜索。

现阶段只能声称实现和结构验证完成。正式的 NKSID GPU 对比必须等待实验门禁允许且当前 GPU 任务结束后再运行；零训练轮次冒烟结果不能用于判断哪种方法的精度或 GPU 效率更高。

## 搜索逻辑

每个候选先进行硬件可行性检查。违反 latency（推理延迟）、LUT（查找表资源）、DSP（数字信号处理单元）、BRAM（块 RAM）、带宽或片外存储限制的候选不会进入父代种群。可行候选按以下顺序参与选择：

1. Pareto 非支配等级越低越优；
2. 同等级内拥挤距离越大越优，以维持解集多样性；
3. 子代进入种群后，默认按 Pareto rank + crowding distance 进行环境选择；同 rank 下保留拥挤距离较大的候选；
4. 父代由 tournament（锦标赛）选择，子代采用双亲 stage 级均匀交叉；交叉失败时回退到单亲变异；
5. 外部 Pareto archive（非支配解档案）保留全部历史非支配解，不受有限种群容量影响；
6. 架构编码使用 SHA-256 去重，并支持随机注入和搜索空间耗尽检测。

该过程没有固定的线性权重和单一 reward。`objective_weights` 仅为历史 RL 配置兼容保留；aging 搜索应优先使用显式 `search.pareto.objectives`。

## 当前 Pareto 轴与代表解

新对照配置显式采用四目标：

- `f_clean`（干净 inner-validation macro-F1，最大化）：作为类别不均衡声呐分类的主要任务指标；
- `f_robust`（冻结扰动集合上的平均 macro-F1，最大化）：衡量分类鲁棒性；
- `latency_ms`（推理延迟，最小化）：作为部署效率代理；
- `energy_mj`（单次推理能耗估计，最小化）：作为部署能效代理。

LUT、DSP、BRAM、带宽、片外存储和其他板级限制保留为硬可行性门禁。这样可避免在仅 200 个候选上形成过高维 Pareto 空间，导致大量候选互不支配、进化选择压力过弱。`top1` 作为次要分类指标报告，但不与高度相关的 `f_clean` 同时进入 Pareto 排序。

搜索不产生唯一“总最优架构”，而从完整 Pareto 前沿交付三个可重合的代表角色：`accuracy_first` 最大化 `f_clean`；`sonar_robust` 最大化 `f_robust`；`deployment_balanced` 最小化四个等影响、逐轴 min-max 归一化目标到理想点的距离。后者是透明的 generalized knee approximation（广义膝点近似），不是唯一数学定义的精确 knee，也不是训练 reward 权重。

`latency_ms`、`energy_mj` 和 `power_w` 在搜索阶段仍是估计量，不等同于板上实测值。实测功耗、route/HLS、COM5 板上验证和最终重训练必须保持独立证据层。

## 声呐图像指标边界

`f_robust` 使用同一 frozen inner-validation 标签，在固定 seed 下分别施加 speckle variance 0.01、speckle variance 0.04、contrast 0.70 和 3x3 blur，并对四个条件的 macro-F1 取均值。每次运行落盘协议 SHA-256、逐条件指标和最差条件 macro-F1。它只表示确定性合成扰动下的分类鲁棒性，不等同于真实任务域鲁棒性或图像复原质量。

PSNR/SSIM 等图像质量指标不应在没有成对干净参考图的情况下直接加入 Pareto 排序；若以输入图自身作为参考，会偏向恒等映射，并可能惩罚对分类有益的去噪或边缘增强。

如后续获得可靠的成对干净目标，可把 PSNR/SSIM 作为独立的算子质量门禁或新增 Pareto 轴，并先验证它与分类错误率的相关性。在此之前，建议把去噪质量、分类质量和板级指标分别报告，避免形成不可解释的综合分数。

## 公平对比协议

RL 与 aging 必须满足以下条件才允许形成方法优劣结论：

- 相同数据集、fold、预处理、搜索空间、硬件约束和训练设置；
- 相同候选评估数和每候选训练轮次；
- 相同且配对的随机种子集合，推荐至少 3 个种子；
- 在同一 GPU 型号上顺序运行，避免资源竞争；
- 正式 CUDA 运行前必须确认没有其他 Python/CUDA 进程；编排器默认自动拒绝共享 GPU；
- 多种子运行默认按 RL→aging、aging→RL 交替，减少固定先后顺序和热状态偏差；
- 每个方法/种子组合只有一个完成记录；
- 所有激活的 Pareto 目标都有完整数值。

比较指标分为两类：

- 质量：best `f_clean`、best `f_robust`、可行候选数、联合 Pareto 贡献数、`C(A,B)` 支配覆盖率；
- 开销：完整子进程 `job_gpu_reserved_hours`、搜索调用 `gpu_reserved_hours`、CUDA event 时间、峰值 CUDA 显存和每 GPU-hour 候选数。

多种子结果按相同 seed 配对，统一报告 `aging - RL` 差值、均值与标准差、配对均值差的 95% Student-t 置信区间、配对 Cohen's dz 效应量，以及双侧精确 sign test（符号检验）和 Holm 多重比较校正。少于 3 个完整配对种子时，`inference_ready=false`，只允许描述性比较。即使只有 3 个种子，双侧精确符号检验也不可能达到 `p<0.05`，效应量亦不稳定，因此只能报告原始配对值和区间，不能声称统计优越。质量和计算成本若方向冲突，不声明单一“总冠军”。

其中 `job_gpu_reserved_hours` 是主 GPU 成本指标，由父编排器计量完整子进程 wall-clock，包括解释器启动、数据/模型初始化、搜索调用和结果落盘，代表独占 GPU slot 的实际占用成本。`search_efficiency.json` 中的 `gpu_reserved_hours` 只表示搜索调用本身，作为次级拆分指标；CUDA event 时间只近似活跃加速器工作，三者不能混用。恢复运行时，完整任务与搜索调用均按分段累计。

当前仓库对高维 hypervolume（超体积）的实现是粗略占位值，因此不能把该字段作为主要结论。联合 Pareto 贡献和支配覆盖率使用两个方法的完整候选共同重新计算。

## 运行方式

先只检查将执行的命令：

```powershell
python scripts/run_aging_vs_rl_benchmark.py --seed 42 --seed 43 --seed 44 --dry-run
```

门禁允许且 GPU 空闲后顺序运行：

```powershell
python scripts/run_aging_vs_rl_benchmark.py --seed 42 --seed 43 --seed 44 --device cuda --budget 200 --train-epochs 3
```

正式搜索还必须同时满足 Gate0 完整正式工作量、G2、G4；搜索空间包含 `denoise` 或 `edge` 时还必须通过 G5。G3 批准文件必须由人工决策产生，并显式包含
`"methods": ["rl", "aging_evolution"]`；代码实现、测试通过或本文件均不构成批准。门禁还会核对每个请求启动的方法确实位于批准列表中，防止用一种方法的批准误启动另一种方法。
可从 `configs/experiment/stage3_replan_approval.template.json` 开始人工审查；该模板固定为 `approved:false`，不能直接作为门禁证据。

中断后继续：

```powershell
python scripts/run_aging_vs_rl_benchmark.py --seed 42 --seed 43 --seed 44 --device cuda --budget 200 --train-epochs 3 --resume
```

恢复时，已达到预算且存在 `results/job_efficiency.json` 的方法/种子任务会直接跳过，不会因无效重启重复累计 GPU-hour。若旧完成结果缺少完整任务计时，正式编排会拒绝猜测或回填成本，并要求使用新 run prefix 重新运行。

脚本保留原 RL 代码，分别运行新 RL 对照配置与 aging 配置，并在结束后调用 `scripts/compare_search_methods.py` 生成 `comparison.json`、`comparison.csv` 和 `comparison.md`。每个 run 的 `results/job_efficiency.json` 记录完整子进程成本；`benchmark_manifest.json` 记录配对种子、预算、方法顺序、完整命令、独占 GPU 要求、完成任务数和失败原因。比较器只有在质量、协议、配对种子、联合 Pareto 和独占 CUDA 完整任务计时均有效时才设置 `comparison_ready=true`。

`--allow-shared-gpu` 只用于明确标记为非正式的调试运行，不应出现在正式 GPU-hour 对比命令中。若检测到其他 Python/CUDA 任务，默认行为是停止并报告 PID，而不是并行抢占 GPU。

当 `--device cuda` 时，编排器优先使用项目本地 `.venv_cuda` Python，并把实际解释器路径写入 manifest；必要时可用 `--python` 显式指定其他已验证环境。

## 关键文件

- `src/hwnas_fpga/search/aging_evolution_searcher.py`：多目标老化进化搜索器；
- `src/hwnas_fpga/search/efficiency.py`：搜索 wall-clock 与 GPU 时间账本；
- `configs/search/nksid_aging_mobile_anchor_mobilenet_v2_av7k325_200.yaml`：aging 配置；
- `configs/search/nksid_rl_pareto3_mobile_anchor_mobilenet_v2_av7k325_200.yaml`：新鲜 RL 公平对照配置；
- `configs/search/nksid_rl_mobile_anchor_mobilenet_v2_av7k325_200.yaml`：保留的历史 RL 配置，未删除；
- `scripts/run_aging_vs_rl_benchmark.py`：顺序运行与多种子编排；
- `scripts/compare_search_methods.py`：公平性校验和证据打包。
