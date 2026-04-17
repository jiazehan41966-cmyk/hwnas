# 搜索对比实验执行整理

## 目标

在同一套搜索空间、同一套 FPGA 约束、同一数据集划分下，完成 `Random`、`RL`、`ProxylessNAS` 三种搜索策略的公平对比，形成论文核心实验表格。

建议按四个阶段推进，而不是把三种搜索方法混在一起：

1. 确认搜索空间配置
2. 跑随机搜索 baseline
3. 跑 RL 搜索
4. 跑 ProxylessNAS 搜索

---

## 仓库里已经对应上的入口

### 1. 搜索空间与探测

- 基础搜索空间配置：`configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml`
- 搜索空间探测脚本：`run_search_space_probe.py`

### 2. 三种搜索方法配置

- Random：`configs/search/nksid_random_baseline_mobile_anchor_shufflenet_av7k325_200.yaml`
- RL：`configs/search/nksid_rl_mobile_anchor_shufflenet_av7k325_200.yaml`
- ProxylessNAS：`configs/search/nksid_proxyless_mobile_anchor_shufflenet_av7k325.yaml`

### 3. 统一搜索入口

- 统一入口脚本：`run_search.py`

### 4. 宏结构实现位置

- 搜索空间定义：`src/hwnas_fpga/search_space/space.py`
- 模型构建：`src/hwnas_fpga/models/builder.py`

当前仓库中的 `stage_based` 宏结构实际对应为：

`Stem -> post_stem_downsample -> Stage2/Stage3/Stage4 -> Conv Head -> GAP -> FC`

这和你要的 `Stem -> Stage2/3/4 -> Conv5 -> FC` 是一致的，区别只是在代码里把 ShuffleNetV2 常见的下采样过渡显式拆成了 `post_stem_downsample_stride`。

---

## 阶段一：确认搜索空间配置（1 天）

### 目标

冻结一份后续三种方法共用的搜索空间 YAML，保证 Random / RL / ProxylessNAS 的对比是公平的。

### 需要确认的项目

| 项目 | 目标值 | 当前 mobile_anchor 配置状态 |
|---|---|---|
| 宏结构 | `Stem -> Stage2/3/4 -> Conv5 -> FC` | 已满足 |
| `macro_type` | `stage_based` | 已满足 |
| `kernel_choices` | `{3, 5}` | 已满足 |
| `width_multipliers` | `{0.5, 0.75, 1.0, 1.25}` | 当前为 `{0.5, 0.75, 1.0}`，缺少 `1.25` |
| Stage2 depth | `{3, 4, 5}` | 当前更浅：`{1, 2, 3}` |
| Stage3 depth | `{6, 7, 8, 9}` | 当前更浅：`{2, 3, 4}` |
| Stage4 depth | `{3, 4, 5}` | 当前更浅：`{1, 2, 3}` |
| FPGA 约束 | `DSP/BRAM/LUT` 上限明确 | 已满足，当前为 `DSP=840, BRAM=445, LUT=50950` |

### 建议作为正式对比前的统一配置

- `family_profile: mobile_anchor`
- `macro_type: stage_based`
- `stem_channels: 24`
- `stage_base_channels: [24, 48, 96]`
- `width_multipliers: [0.5, 0.75, 1.0, 1.25]`
- `stage_depth_choices: [[3, 4, 5], [6, 7, 8, 9], [3, 4, 5]]`
- `kernel_choices: [3, 5]`
- `constraints.max_dsp: 840`
- `constraints.max_bram: 445`
- `constraints.max_lut: 50950`
- 如果要启用 LUT 估算，建议同步设置 `hardware.lut_path`，例如 `artifacts/kintex7_analytical_lut.pkl`

### 建议动作

1. 以 `configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml` 为母版。
2. 将 width 和 stage depth 调整到最终要写论文的版本。
3. 让 Random / RL / Proxyless 三份 YAML 全部复用同一套 `search_space` 和 `constraints` 段。

### 配置确认命令

```bash
python run_search_space_probe.py --config configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml --num-samples 200 --run-name probe_mobile_anchor_shufflenet_av7k325
```

### 这一阶段的验收标准

- 硬件约束写死在 YAML 中
- 三种搜索方法使用同一搜索空间
- `probe_summary.json` 中可行架构比例大于 50%

### 已有证据

截至 2026-04-07，现有随机 baseline 正式运行目录
`results/random_baseline_mobile_anchor_shufflenet_av7k325_200_formal/`
显示 `200/200` 全部可行，说明当前较保守版本的搜索空间本身是可行的。

---

## 阶段二：随机搜索 baseline（1 到 2 天）

### 目标

随机采样 200 个架构，建立统一 baseline，并观察精度与 FPGA 代价的分布范围。

### 推荐配置

- `configs/search/nksid_random_baseline_mobile_anchor_shufflenet_av7k325_200.yaml`

如果阶段一改了搜索空间，这份配置也要同步更新。

### 运行命令

```bash
python run_search.py --config configs/search/nksid_random_baseline_mobile_anchor_shufflenet_av7k325_200.yaml --run-name random_baseline_mobile_anchor_shufflenet_av7k325_200_formal
```

### 重点产物

- `results/<run>/results/candidates.jsonl`
- `results/<run>/results/candidates.csv`
- `results/<run>/results/summary.json`
- `results/<run>/results/pareto_front.json`

### 这一步要回答的三个问题

1. 搜索空间是否合理：可行比例是否大于 50%
2. 随机搜索 baseline 是多少：后续 RL 和 Proxyless 是否明显优于它
3. 精度和硬件代价分布范围如何：是否存在明显 Pareto 前沿

### 已有状态

截至 2026-04-07，`results/random_baseline_mobile_anchor_shufflenet_av7k325_200_formal/` 已完成：

- `total_evaluated = 200`
- `feasible = 200`
- `infeasible = 0`
- 当前最好结果约为 `macro_f1 = 0.5836`

如果你后面修改了搜索空间，建议这一步重跑；如果搜索空间不变，这个结果可以直接作为 baseline。

---

## 阶段三：RL 搜索（2 到 3 天）

### 目标

在和随机搜索完全相同的搜索空间中运行 RL 搜索，看是否能找到更优的架构。

### 推荐配置

- `configs/search/nksid_rl_mobile_anchor_shufflenet_av7k325_200.yaml`

### 运行命令

```bash
python run_search.py --config configs/search/nksid_rl_mobile_anchor_shufflenet_av7k325_200.yaml --run-name rl_mobile_anchor_shufflenet_av7k325_200_formal
```

如果沿用已有未完成运行，可使用：

```bash
python run_search.py --config configs/search/nksid_rl_mobile_anchor_shufflenet_av7k325_200.yaml --run-name rl_mobile_anchor_shufflenet_av7k325_200_formal --resume
```

### 推荐奖励函数方向

- 提高 `accuracy` 权重
- 不再使用固定 `-10` 的不可行惩罚
- 改成与超约束程度相关的渐进式惩罚

### 当前实现状态

`src/hwnas_fpga/search/rl_searcher.py` 已支持：

- `infeasible_penalty_mode`
- `infeasible_base_penalty`
- `infeasible_penalty_scale`

并且当前 `run_search.py` 已经会把 YAML 中的 `search.reward_cfg` 透传给 `RLSearcher`。因此现在可以直接通过配置切换：

- 固定惩罚
- 按 violation ratio 的渐进式惩罚

建议正式实验时，把实际采用的 `reward_cfg` 一并保存在对应运行目录中，便于复现实验表格。

### 重点产物

- `results/<run>/checkpoints/controller_latest.pt`
- `results/<run>/checkpoints/controller_best.pt`
- `results/<run>/checkpoints/search_state.json`
- `results/<run>/results/candidates.jsonl`
- `results/<run>/results/best_candidate.json`

### 已有状态

截至 2026-04-07，`results/rl_mobile_anchor_shufflenet_av7k325_200_formal/` 尚未完成，但已留下中间状态：

- `total_evaluated = 31`
- `feasible = 24`
- `infeasible = 7`
- 当前最好结果约为 `macro_f1 = 0.5209`

因此 RL 这一步当前更适合定义为：

1. 先固定最终采用的 `reward_cfg`
2. 再决定是续跑现有目录，还是删除实验偏差后重跑

---

## 阶段四：ProxylessNAS 搜索（2 到 3 天）

### 目标

在同一搜索空间上完成 ProxylessNAS 搜索，并与 Random / RL 做公平对比。

### 推荐配置

- `configs/search/nksid_proxyless_mobile_anchor_shufflenet_av7k325.yaml`

### 运行命令

```bash
python run_search.py --config configs/search/nksid_proxyless_mobile_anchor_shufflenet_av7k325.yaml --run-name proxyless_mobile_anchor_shufflenet_av7k325_formal
```

### 这一阶段的优势

Proxyless 配置链路目前比 RL 更完整，硬件相关正则已经能直接通过 YAML 配置：

- `grad_reg_loss_type: mul#log`
- `grad_reg_loss_alpha: 0.2`
- `grad_reg_loss_beta: 0.3`
- `target_hardware: latency_ms`
- `ref_value: 50.0`

这基本已经对应你提到的“用梯度/硬件正则替代简单固定惩罚”的思路。

### 重点产物

- `results/<run>/logs/console.log`
- `results/<run>/results/search_space_summary.json`
- `results/<run>/results/search_space_pruned_summary.json`
- 完成后应补齐 `summary.json`、`candidates.*`、`best_candidate.json`

### 已有状态

截至 2026-04-07，`results/proxyless_mobile_anchor_shufflenet_av7k325_formal/` 尚未完成：

- 日志显示已进入 `warmup=40, search_epochs=160`
- 当前日志停在 `Warmup 35/40`
- 结果目录里还没有最终 `summary.json`

因此这一步目前应视为“已有启动基础，但正式结果还没收口”。

---

## 建议的最终执行顺序

### 如果你要追求论文对比的严格公平性

1. 先冻结一份新的统一搜索空间 YAML
2. 重跑 search-space probe
3. 重跑 Random baseline
4. 固定 RL 的 `reward_cfg` 后重跑 RL
5. 用同一搜索空间重跑 ProxylessNAS

### 如果你要优先快速形成初版实验表

1. 直接采用当前 Random 正式结果
2. 判断 RL 是否续跑还是重跑
3. 先把 Proxyless 正式跑完
4. 用当前统一约束先出一版 `Random vs RL vs Proxyless` 表格
5. 再决定是否扩大搜索空间到 `width=1.25` 和更深的 stage depth

---

## 论文表格建议字段

建议三种方法统一汇报以下指标：

- Best `macro_f1`
- Best `latency_ms`
- Best `DSP`
- Best `BRAM`
- Best `LUT`
- Feasible ratio
- Search budget
- Search time

如果要做图，优先画两类：

- `macro_f1` vs `latency_ms`
- `macro_f1` vs `DSP` 或 `LUT`

---

## 一句话结论

这项工作最合理的推进方式不是“直接连跑三种搜索”，而是先冻结统一搜索空间，再以 `Random -> RL -> ProxylessNAS` 的顺序完成公平对比。当前仓库已经具备大部分入口和结果目录，但正式对比前仍有两个关键动作最值得优先处理：

1. 把 mobile_anchor 搜索空间收敛到你最终想写论文的版本
2. 在正式运行前固定 RL 的 `reward_cfg`，避免同名实验使用了不同惩罚策略
