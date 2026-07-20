# NAS 四方法协议修订：HW-PR 证据边界（中文伴随档案）

- 英文原件：`nas_four_method_protocol_amendment_20260716.md`；SHA256：`94b43ada59c752e791e0942a115d47b0b2fc6e49fa0d23107d47a586ca602529`。

## 修订原因

固定在 commit `296c6576fbae2b277e56c704ff3b6e648ec4c2be` 的 HW-PR-NAS checkout 不是可执行的作者 artifact。README 提到但仓库缺少相应 surrogate modules；`search_algo.py` 调用未定义的 `valid_loss()`；`test.py` 调用不兼容的搜索入口。因此不能把该仓库描述为已成功复现的作者 runtime。

四方法对比继续保留，但第四方法冻结为 `hwpr_paper_spec_local_adapter`，属于 B 类方法迁移。所有表、图和 manifest 必须显示 `NOT_AUTHOR_RUNTIME` 与 `paper_encoder_equivalent=false`；禁止把作者仓库中的数值导入 T5。

## 共同正式协议

- 方法：`random`、`rl`、`aging_evolution`、`hwpr_paper_spec_local_adapter`。
- Pilot：每方法 50 次实际 evaluator 调用，seed 42、43、44。
- Formal：每方法 300 次实际 evaluator 调用，seed 42–51。
- 每次 evaluator 调用在相同冻结 NKSID inner split、epoch 预算、数据变换、搜索空间与解析硬件估计器下训练/评估一个候选。
- rejected、duplicate、surrogate-only 与 failed evaluator call 分开记录，不得静默占用或扩充成功评估预算。
- 方法顺序按 seed 使用四方法 Latin-square rotation（拉丁方轮换），GPU 顺序且独占执行，避免 wall-clock 漂移总是偏向同一方法。
- 既有搜索期 `f_robust` 保持固定四条件定义；后续图像域 SNR 实验是扩展，不重定义搜索目标。

## 精确 Pareto/HV 合同

- 归一化最小化向量：`(1-f_clean, 1-f_robust, latency_ms/50.0)`。
- 参考点：`(1, 1, 1)`。
- LUT、DSP、BRAM 是正式硬约束。energy/power 估计只作诊断列，不是第四个 Pareto 目标，也不是实测功耗证据。
- 归一化 latency > 1 的候选不能在固定参考点外贡献 dominated volume，不得通过裁剪伪造改进。
- T5/F2/F3 使用经过测试的精确 hypervolume 实现，并保留每次调用的 anytime 点。
- 次指标：双向 Pareto coverage、唯一 feasible non-dominated 数量、feasible ratio、top-k recall、NDCG@k、GPU-hours、wall-clock、峰值 CUDA 显存、duplicate/rejection/failure 数量以及每个 seed 的最终分布。

## 本地 HW-PR paper-spec adapter

- 架构表示：已声明的本地 stage-tabular encoding，不是作者缺失的 feature+GCN+LSTM encoder。
- Target：在共同三目标合同下，从已评估候选计算 exact Pareto rank。
- Surrogate：三层 MLP，使用论文描述的 ListMLE Pareto-rank loss。
- Warm start：20 个随机已评估候选。
- 更新：每新增 10 次成功 evaluator 调用后重新拟合。
- Proposal pool：每次从同一冻结搜索空间采样 64 个唯一且未评估架构。
- 选择：surrogate score 最高者，并保留预声明 0.10 随机探索；归档 proposal、score、model seed 与 training loss。
- 可在统一项目协议下数值比较，但结论只能指向本地 paper-spec migration，不能指向缺失的作者实现。

## 实现门禁

英文原件禁止在当时 G1 依赖的 code state `bdf1a9aab4f50b6de0eddcf7a9493bd4e3b70ee46c596b041e02c73a6ae82471` 下新增或修改 `.py`、`.yaml`、`.json`、`.toml`。当前 source freeze 仍保持不变；adapter 只能在独立的新 source snapshot 与 campaign fingerprint 下实现。此前 T5 状态为 `PENDING_LOCAL_ADAPTER_IMPLEMENTATION`，当前仍未有四方法正式结果。

本中文伴随档案不授权启动 NAS，对性能中断阈值也不替代用户决定。
