# 声呐任务指标同步记录

实现入口：[`metrics_ledger.py`](../src/hwnas_fpga/metrics_ledger.py)，schema 为 `sonar-metrics-ledger-v1`，共 **77 项**。模块负责完整记录、来源校验及可靠追加，不负责计算指标，也没有自动改造旧训练入口；各 runner/wrapper 必须显式接入。

截至 **2026-09-22**，[`METRICS.jsonl`](../reports/sonar_deployment_admission_v1/METRICS.jsonl) 有 **18 条历史 FP32/INT8 导入事件**，不是18个新增实验，也不代表每epoch、搜索轮或板测已经全部接入。`METRIC_COVERAGE.csv` 显示覆盖情况。**PENDING 不是已测，记录完整不等于实验完成。**

## 何时写入

| 事件 | 同步记录时机与身份 |
|---|---|
| 每epoch | epoch训练/valid评价结束后立即写；`epoch`、`global_step`、`precision`、`evaluation_id`放入identity。没有实际评价的指标保留PENDING，不能复制上一epoch冒充新结果。 |
| 每个候选评价 | 记录architecture ID、seed、budget、checkpoint SHA、split/evaluator/protocol SHA、评价ID；写入成功后才能晋级或标记完成。 |
| 每轮搜索 | 记录round ID、累计/本轮口径、候选集manifest、时间、GPU消耗、最优值和Pareto；评价失败也保留事件及原因。 |
| 每次量化评价 | FP32/INT8绑定同checkpoint、split、evaluator及校准manifest和numeric contract；不得混合不同来源求drop。 |
| 每个硬件阶段 | 分开记录HLS、综合、布局布线、板端运行和功耗测量；绑定候选/bitstream SHA、设备、工具版本及测量协议。 |

每条记录必须有稳定`identity.run_id`。默认event ID由任务信息和**完整identity**确定；同run不同epoch、precision或评价应有不同identity。来源修正、补齐待测字段必须**创建新event**，通过`identity.parent_event_id`关联原event；不能覆盖旧记录。同ID同内容幂等，同ID异内容报错。JSONL有跨进程锁、flush/fsync；截断残尾先保全再恢复，完整行损坏拒绝继续。

## 指标口径与证据

| 分组 | 指标与适用性/来源要求 |
|---|---|
| 分类任务 | Macro-F1、Top-1、Precision、Recall、Per-class F1、Confusion Matrix来自同一冻结分类评价器；保留类别顺序和平均方式。 |
| 检测任务 | 保留`map50/map75/map50_95`及Precision/Recall。检测F1、Per-class F1、混淆矩阵须明确GT匹配IoU、分数阈值、类别及背景规则；不能由AP转换。**图像分类Top-1不能套用检测任务**，标NOT_APPLICABLE。 |
| 专项/鲁棒性 | NKSID Tail-class F1须冻结尾类定义；Cylinder2/SIMD Small-target F1/Recall须尺寸和匹配定义；Noise Robustness、ΔMacro-F1须同一噪声协议、参考条件和指标口径。 |
| 图像质量 | PSNR/SSIM**仅用于有干净参考的合成噪声实验**，记录参考manifest SHA、noise protocol、data range；无干净参考为NOT_APPLICABLE。 |
| 搜索/Pareto | Search Time、GPU-hours、候选评价数、Sampling Efficiency、Hardware-feasible Rate、Best-so-far；效率/可行率保存分母、定义及约束。HV、非支配解数和Pareto Front绑定候选集合、目标方向；HV还需固定reference point及normalization。 |
| 复杂度/量化 | Params、MACs、FLOPs绑定输入及统计方法；FP32/INT8分别保存本任务指标和drop。分类用Macro-F1/Top-1，检测用AP/Precision/Recall；`int8_*_drop = FP32 − INT8`，不能混入相反符号。 |
| 代价模型准确性 | 完整网络Latency/LUT/DSP/BRAM的MAE、MAPE、Spearman需配对prediction manifest、实测network manifest和样本数。算子LUT覆盖或单算子误差不能冒充完整网络预测准确性。 |
| FPGA资源/时序 | LUT/FF/DSP/BRAM及利用率绑定明确证据层、设备和工具；BRAM单位须注明BRAM18/36等。WNS/TNS/Timing Met要求实际布局布线来源，HLS成功不等于时序通过。 |
| 板端/能效 | Latency、FPS、Actual Clock、Power_w、Energy/Inference必须有board_measurement来源、artifact SHA、设备、测量方法。延迟/FPS绑定precision、batch、input、warmup/repeats；功耗记录rails/整板范围、是否扣除idle。MACs、算子求和、vectorless估计不能写成板端实测。 |

每项统一含`status/value/unit/provenance/reason`。MEASURED必须给`provenance.source`，建议同时绑定源文件SHA；NaN/Inf拒绝保存。PENDING/NOT_APPLICABLE的value只能为null，不能用0占位。AP/F1等为0～1，利用率为百分数；MAPE为比例。特殊指标的附加provenance约束见schema实现。

## 最小接入示例

下例变量均由实际runner产生，`required`应在该阶段协议中冻结；不能为了通过检查临时删掉缺失指标。

```python
from hwnas_fpga.metrics_ledger import (
    build_metric_record, require_measured_metrics, append_metric_record,
)

identity = {
    "run_id": run_id, "epoch": epoch, "global_step": global_step,
    "precision": precision, "evaluation_id": evaluation_id,
    "checkpoint_sha256": checkpoint_sha256,
    "split_manifest_sha256": split_sha256,
    "evaluation_protocol_hash": evaluator_sha256,
}
required = ["map50", "map75", "map50_95", "precision", "recall"]
record = build_metric_record(
    task_type="detection", dataset=dataset, network=architecture_id,
    seed=seed, stage="candidate_valid", identity=identity,
    measured={key: {"value": valid_metrics[key], "provenance": {
        "source": str(metrics_path), "source_sha256": metrics_sha256,
        "evaluation_protocol_hash": evaluator_sha256,
    }} for key in required},
)
require_measured_metrics(record, required)  # 缺失/PENDING/N/A则失败
append_metric_record(ledger_path, record)  # 成功追加/幂等确认，flush+fsync
# 仅在以上成功后，runner才可以标记评价完成或更新晋级状态。
```

如果阶段失败但已有部分结果，也先以独立失败事件保存已有数据与失败原因，必需项保持PENDING，禁止标记阶段完成。精度门采用另行冻结的D相对基线，ledger本身不会推定应用精度合格，也不会触发训练、上板或访问test。
