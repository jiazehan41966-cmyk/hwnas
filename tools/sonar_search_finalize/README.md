# 联合搜索的独立收尾器

本工具修复新 `sonar_joint_v1` 的结果汇总范围错误：冻结训练入口的 `joint_points()` 调用旧 H4 Pareto 函数，旧函数只接纳 B1/C/D。新搜索选出的其他架构即使通过三种子量化准入，也会被旧白名单排除。

这里仅重建结果，不改正在执行的训练、冻结文件、旧 B1/C/D 的 Q(x) 或硬件代价库。它复用现有 `CostDatabase`，按本轮归档配置中原有的量化精度门槛，构造本轮确认候选集的 Q(x)。完整收据到齐之前返回 `PENDING_CAMPAIGN_EVIDENCE`，不会发布部分候选构成的最终前沿。

## 输入与验收

- 原始 `freeze.json`、`snapshot.json`、`source_snapshot.zip`、`cost_snapshot.json`；逐项检查归档 SHA256、协议和数据清单身份。
- 三阶段排序、晋升、弃选抽查、12 架构 shortlist；每个数值回溯至原 train receipt，不允许用之后的训练结果替代当时预算。
- 全部三种子 train/quant receipts，以及其列出的 checkpoint、校准清单、量化 profile 和 result JSON；检查文件哈希，但不反序列化 checkpoint、不读取数据图像。
- 校准仅使用被冻结的前 64 张 train 图像清单；精度来自 valid；禁止 test 进入选择。
- 量化层位宽、per-tensor 粒度、INT32 累加、INT64 requant 和无 FP32 旁路必须满足原合同。混合精度只改变 head。
- 硬件数据与合同载荷须仍匹配原 freeze；允许使用修复后的代价读取器，但记录其当前源码哈希，并核对原 `reference_min_latency` 代价快照。

新输出的 `CAMPAIGN_Q_ADMITTED` 是本次运行的准入，不覆盖旧 `reports/phase5e_mixed_precision/Q_OF_X.json`。资源仍为算子峰值代理，延迟仍为算子串行 HLS 估计；不代表整网 FPGA 部署或全空间全局最优。

## 使用

先从已完成的远端隔离运行收集上述完整产物到一个新的本地目录，保留相对路径。不要将本地已变化的源码覆盖正在运行的远端冻结树。收尾器位于冻结扫描范围之外。

在仓库根目录运行，输出必须是原 campaign 之外的新目录：

```powershell
.venv_cuda/Scripts/python.exe -B tools/sonar_search_finalize/run.py --campaign results/COMPLETE_CAMPAIGN_COPY --output results/NEW_FINALIZATION
```

只有 `status=COMPLETE` 或 `COMPLETE_NO_ADMITTED_POINTS` 才是完整收尾。命令退出 0 且 `status=PENDING_CAMPAIGN_EVIDENCE` 仅表示缺项已被正确识别；本地缺失也不等于远端没有运行。证据被篡改或身份不一致会报错，不输出最终前沿。

本工具尚未在本轮真实完整训练结果上收尾，因为该训练仍在进行。回归测试使用临时目录内的合成分数及 checkpoint 字节，不构成实验精度证据。

```powershell
.venv_cuda/Scripts/python.exe -B -m pytest tests/test_sonar_search_finalize.py -q
```
