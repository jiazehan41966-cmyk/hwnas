# ZCU104 声呐专项指标与温度稳定性协议

## 1. 为什么算法层必须加入声呐指标

只报告干净测试集的 Macro-F1，能够回答“模型在当前数据分布上的分类能力”，但不能单独支撑“适应强噪声、低对比度声呐图像”的结论。因此，算法层采用两个相互独立、不得相互替代的主要终点：

1. `clean_macro_f1`：冻结 outer split 原始图像上的 Macro-F1；
2. `sonar_robust_macro_f1`：冻结声呐退化协议下，各退化族和严重度等权聚合的 Macro-F1。

如果论文或项目报告要明确声称“对强噪声、低对比度具有鲁棒性”，第二项必须作为算法层共同主要终点（co-primary endpoint），并在读取正式 outer 结果前冻结。若只作为探索性次要指标，则结论必须收缩为“在原始 NKSID 测试分布上有效”。

算法层主要终点、部署主要终点和性能主要终点分别回答不同问题：

| 层次 | 终点 | 回答的问题 |
|---|---|---|
| 算法 | `clean_macro_f1` | 原始声呐数据上能否正确识别 |
| 算法/声呐 | `sonar_robust_macro_f1` | 冻结噪声和低对比度条件下能否保持识别能力 |
| 部署 | `total_energy_per_inference` | 整板完成一次推理消耗多少能量 |
| 性能 | `system_e2e_latency_p50/p95` | 完整系统一次推理通常和尾部需要多长时间 |

## 2. 冻结声呐退化协议

### 2.1 退化族

正式协议至少覆盖以下三个核心退化族：

| 退化族 | 物理/数据含义 | 正式级别 |
|---|---|---|
| multiplicative speckle | 乘性散斑噪声，模拟声呐相干成像中的颗粒噪声变化 | 核心 |
| additive noise | 加性电子/传感噪声，按冻结 SNR 或噪声标准差生成 | 核心 |
| contrast attenuation | 目标—背景对比度衰减或灰度动态范围压缩 | 核心 |
| blur/resolution loss | 模糊或分辨率下降；只有确认与采集链路相符时使用 | 条件性次要 |
| shadow/nonuniform artifact | 阴影或不均匀照明；只有具备验证过的物理模型时使用 | 探索性 |

不得使用可能改变目标语义或遮蔽类别判别区域的任意图像增强来冒充传感器退化。每个退化族设置 mild、moderate、severe 三档；具体参数只能用 training split 的先导分析确定，然后在读取 outer 结果前冻结。禁止根据正式结果重新调严重度。

### 2.2 公平性和可复现性

- 同一 `sample_id × corruption_family × severity` 对所有模型使用完全相同的退化图像；
- 随机退化种子由 `sample_id` 和全局冻结种子确定，不按模型变化；
- 退化前后保持图像尺寸、通道数、数值范围和标签不变；
- 保存退化配置、实现版本、随机种子、样本清单和 SHA256；
- 每个 fold/seed 分别计算指标，不把所有预测先拼接后伪造独立重复；
- clean、每个退化单元和聚合指标均须报告，不允许只报告平均值隐藏最差条件。

### 2.3 指标定义

对 fold/seed 单元 `u`、退化族 `j`、严重度 `k`：

```text
MF1(u,j,k) = 该退化单元的 Macro-F1
absolute_drop(u,j,k) = clean_MF1(u) - MF1(u,j,k)
retention(u,j,k) = MF1(u,j,k) / clean_MF1(u)

sonar_robust_macro_f1(u)
  = mean_j( mean_k( MF1(u,j,k) ) )

worst_condition_macro_f1(u)
  = min_j,k( MF1(u,j,k) )
```

先在每个退化族内对严重度等权平均，再对退化族等权平均，避免某一退化族因档位更多而获得更高权重。跨 5 folds × 3 seeds 报告 mean、std、95% CI，并对预注册模型比较使用配对统计。

必须同时报告：

- `sonar_robust_macro_f1`；
- 每个退化族和严重度的 Macro-F1 曲线；
- `absolute_drop` 和 `retention`；
- `worst_condition_macro_f1`；
- 退化条件下的 per-class F1、minority-class recall 和 confusion matrix；
- FP32 与 INT8 的上述指标，防止量化在困难声呐条件下产生额外退化。

## 3. 声呐图像结构指标的证据边界

### 3.1 PSNR、SSIM、MSE

当前 NKSID 是分类数据集，不是成对的 noisy/clean 图像恢复数据集。使用 `reference_policy=input_as_reference` 时：

- PSNR、SSIM、MSE 只说明 Denoise/Edge 等算子相对输入改变或保持了多少结构；
- 不能说明输出更接近未知的“干净声呐图像”；
- 不能替代 Macro-F1 或声呐鲁棒 Macro-F1；
- 报告中必须显式写出 `reference_policy`。

只有获得同一场景的成对干净参考或可信仿真真值后，才允许使用 `paired_clean_target` 并讨论恢复质量。

### 3.2 CNR、ENL 和 SNR

这些指标有较强声呐意义，但依赖额外标注：

| 指标 | 需要的输入 | 当前使用规则 |
|---|---|---|
| CNR（目标—背景对比噪声比） | 目标 ROI 和背景 ROI | 有可靠 ROI/分割标注后再列为正式指标 |
| ENL（等效视数） | 已标注的均匀背景区域 | 有均匀区标注后再计算 |
| measured SNR | 信号/噪声定义或标定采集 | 有物理或标注依据后使用 |

仅凭分类标签自动猜测 ROI 会引入测量偏差，因此在没有 ROI/分割或采集标定时，CNR、ENL 和 measured SNR 记为 `NOT_AVAILABLE`，不生成伪值。生成退化所用的“目标 SNR”是实验控制参数，不等同于真实采集 SNR。

## 4. 温度稳定性是什么

温度稳定性不是分类效果指标，而是板级运行条件、可靠性指标和能耗比较的混杂因素控制。连续推理会使芯片结温上升，可能导致：

- 静态/漏电功率随温度变化；
- 功率、延迟或吞吐随热状态漂移；
- 触发频率保护、过温告警、reset、timeout 或计算错误；
- 候选 A 在冷板、候选 B 在热板测量，造成不公平比较。

因此需要回答：“在相同散热和稳态规则下，候选能否持续运行，并且温度变化是否伴随能耗、延迟或错误漂移？”

## 5. ZCU104 能否测温

可以。ZCU104 的 Zynq UltraScale+ MPSoC 内含 PS/PL System Monitor（系统监视器），能够读取芯片结温。它测量的是芯片内部结温，不是室温。

### 5.1 Linux 连续采集：正式实验首选

板端 Linux 需要启用 AMD/Xilinx AMS 驱动（`CONFIG_XILINX_AMS`）并具有正确 device-tree 节点。不同 PetaLinux/Linux 版本的 IIO 节点名称不同，应先枚举而不是硬编码：

```bash
for f in /sys/bus/iio/devices/iio:device*/in_temp*_input; do
  [ -r "$f" ] && printf '%s %s\n' "$f" "$(cat "$f")"
done
```

`*_input` 通常为毫摄氏度，记录时转换为 °C 并同时保留原始整数。较新驱动通常暴露 PS LPD、FPD remote 和 PL 温度通道；旧版节点可能带 `ps_temp`、`remote_temp`、`pl_temp` 名称。正式采集前必须把实际节点、label、单位和驱动版本写入 `board_capability_card.json`。

### 5.2 Vivado/JTAG：bring-up 交叉核验

Vivado Hardware Manager 可通过 System Monitor 查看温度，也可用 Tcl 查询 `get_hw_sysmons` 和 `TEMPERATURE` 属性。它适合能力确认和交叉核验；正式长时间运行建议由板端 Linux 以统一时间戳连续采样，以便和功率、延迟日志对齐。

PS/PL 初始化期间不要轮询 PL 温度；应在 bitstream 配置和 runtime 稳定后开始采集。

### 5.3 环境温度

芯片 System Monitor 不能替代环境温度计。若要报告 ambient temperature（环境温度），需增加外部温度探头/热电偶并放在固定位置。没有外部探头时：

- 可记录实验室空调设定值，但必须标为 `room_setpoint`；
- `ambient_temperature` 必须标为 `NOT_MEASURED`；
- 不得把 PS/PL 结温写成环境温度。

## 6. 温度采样和验收协议

### 6.1 采样

- 采样率建议 1 Hz，至少覆盖 idle、warm-up、active 和 cool-down；
- 每条记录包含 monotonic timestamp、PS/PL 温度、候选、clock、fan mode、功率窗口、推理计数和状态码；
- 若能读取风扇转速则记录 RPM；不能读取时只记录固定 fan mode，不虚构 RPM；
- C2/C3 长期稳定性运行 30–60 min；正式功率 randomized block 对所有候选使用同一稳态规则。

### 6.2 稳态规则

建议先导运行后冻结以下示例规则：连续 5 min 的温度线性斜率绝对值不超过 `0.2 °C/min`，且窗口内无过温、降频或 reset。该数值是待预注册的实验阈值，不是 ZCU104 官方规格。

正式记录至少包括：

- `temp_start_c`、`temp_mean_c`、`temp_max_c`、`temp_end_c`；
- `temp_slope_c_per_min`；
- `time_to_thermal_steady_s`；
- 前后半程 latency、throughput、power 的相对漂移；
- latency/power 与温度的配对时间序列或回归斜率；
- overtemperature、throttle、reset、hang、timeout、CRC/argmax mismatch 事件计数。

温度稳定性通过至少要求：无过温/保护事件，无 reset/hang/timeout/CRC 错误，且延迟、吞吐和功率漂移均处于预注册范围。温度本身不要求候选完全相同，但必须在相同散热和稳态规则下测量并完整报告。

## 7. 官方能力依据

- [AMD Zynq UltraScale+ MPSoC AMS](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842163/Zynq%20UltraScale%20MPSoC%20AMS)
- [AMD Vivado UG908 System Monitor](https://docs.amd.com/r/en-US/ug908-vivado-programming-debugging/System-Monitor)
- [AMD ZCU104 Evaluation Board User Guide UG1267](https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd)
