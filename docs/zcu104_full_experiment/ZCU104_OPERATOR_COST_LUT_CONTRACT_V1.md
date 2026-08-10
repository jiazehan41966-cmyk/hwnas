# ZCU104 算子实测代价表合同 V1

## 1. 适用范围

本合同把 `zcu104_full_campaign_v1` 的当前主线改为“先测算子、再冻结搜索空间”。本轮所有硬件代价必须在 ZCU104（XCZU7EV-FFVC1156-2-E）上重新建立。

以下内容仅作历史留档，禁止作为本轮搜索输入、对照网络、标定点、部署对象或代价表补丁来源：

- 以前冻结的四个完整候选网络；
- 与这四个网络关联的 60 份 checkpoint 和 outer prediction；
- AV7K325 或其他器件上的历史算子代价表；
- 未绑定本合同、ZCU104 bitstream 和原始采样文件的功耗数据。

保留且继续约束本轮实验的内容是：独立工作树、源码冻结机制、NKSID 数据集、Protocol V2、服务器环境记录和 ZCU104 连接信息。

## 2. 新算子池与精确形状

本合同中的“候选”是算子定义，不是历史网络候选。算子身份来自 `src/hwnas_fpga/fourstage_operator.py` 当前实现。

| 位置 | 算子 ID | 参数 | 精确输入形状 C×H×W | 精确输出形状 C×H×W | 状态 |
|---|---|---|---:|---:|---|
| Stage2 | `k3_e3` | MBConv, k=3, e=3, s=2 | 16×112×112 | 24×56×56 | 纳入重新测量 |
| Stage2 | `k3_e6` | MBConv, k=3, e=6, s=2 | 16×112×112 | 24×56×56 | 纳入重新测量 |
| Stage2 | `k5_e3` | MBConv, k=5, e=3, s=2 | 16×112×112 | 24×56×56 | 纳入重新测量 |
| Stage2 | `k5_e6` | MBConv, k=5, e=6, s=2 | 16×112×112 | 24×56×56 | 纳入重新测量 |
| Stage4 | `mbconv_k3_e3` | MBConv, k=3, e=3, s=1 | 32×28×28 | 32×28×28 | 纳入重新测量 |
| Stage4 | `skip` | identity, k=1, e=1, s=1 | 32×28×28 | 32×28×28 | 纳入重新测量 |
| Stage4 | `dir_mbconv3_split11_e3_v1` | directional MBConv, k=3, e=3, s=1 | 32×28×28 | 32×28×28 | 纳入重新测量 |
| Stage4 | `split_dw3_control` | mechanism control, k=3, e=3, s=1 | 32×28×28 | 32×28×28 | 纳入重新测量 |
| Stage4 | `mbconv_k5_e3` | MBConv, k=5, e=3, s=1 | 32×28×28 | 32×28×28 | 纳入重新测量 |

Stage2 虽然是成熟算子，也不得复用旧代价。本轮必须形成 9 个独立的 ZCU104 exact-shape（精确形状）算子测量行。只有 Stage2 四行和通过 Stage4 硬件门的行均有完整证据后，才允许构造最多 `4 × 5 = 20` 个网络组合。当前不启动网络搜索。

## 3. ZCU104 Device Contract V1

| 项目 | 冻结值或状态 |
|---|---|
| Board | AMD ZCU104 |
| Device | XCZU7EV-FFVC1156-2-E |
| Board part | `xilinx.com:zcu104:part0:1.1` |
| 工具链 | Vivado/Vitis HLS 2022.2，以服务器实测环境为准 |
| 目标时钟 | 200 MHz（5.0 ns） |
| 器件总量 | LUT 230400；FF 460800；DSP 1728；BRAM36 312；URAM 96 |
| Route timing | WNS ≥ 0 ns 且 TNS = 0 ns |
| 部署预算 | `PENDING_PREREGISTRATION` |

器件总量不是可部署预算。DSP、LUT、BRAM、latency、dynamic power 和 energy 的硬门限必须在看到候选测量结果之前预注册；需要为 shell、DMA、PS/PL 接口、调试、拥塞和时序收敛保留余量。不得把旧的 `DSP <= 700` 或器件总量 `DSP <= 1728` 直接当作新门限。

## 4. Hardware Cost LUT V1 字段

每一行必须同时绑定算子、精确形状、实现参数和证据文件。建议机器可读表使用以下字段组：

| 字段组 | 必需字段 |
|---|---|
| 身份 | `operator_id`, `stage`, `shape_id`, `kernel`, `expansion`, `stride`, `input_shape`, `output_shape` |
| 来源 | `source_commit`, `source_freeze_sha256`, `toolchain_version`, `board_serial`, `target_part` |
| 实现 | `hls_config_sha256`, `xsa_sha256`, `hwh_sha256`, `bitstream_sha256`, `runtime_sha256`, `clock_period_ns` |
| HLS/Route | `latency_hls_cycles`, `latency_hls_ms`, `lut`, `ff`, `dsp`, `bram18`, `bram36`, `uram`, `wns_ns`, `tns_ns` |
| 板级计时 | `latency_board_ms`, `latency_scope`, `fps`, `warmup_count`, `timed_count` |
| Vivado 功耗 | `power_vivado_estimated_w`, `power_vivado_report_sha256` |
| 板载功耗 | `p_idle_w`, `p_active_w`, `p_dynamic_w`, `power_rail`, `power_source`, `requested_sample_rate_hz`, `effective_sample_rate_hz`, `sample_count` |
| 能量与热状态 | `energy_total_mj_per_op`, `energy_dynamic_mj_per_op`, `temperature_start_c`, `temperature_end_c`, `ambient_c`, `fan_mode` |
| 证据状态 | `hls_status`, `route_status`, `board_runtime_status`, `power_status`, `gate_status`, `failure_reason` |

`power_vivado_estimated_w` 与板载实测字段必须分列，不能互相替代。后续可在相同设计、相同活动定义下比较两者误差，但 Vivado 估计不晋级为板级实测证据。

## 5. 板载功耗 micro-benchmark 协议

### 5.1 采样范围

主功耗证据使用 ZCU104 U183 INA226 读取的 VCC12 板输入总功耗，接口为 Linux `ina2xx`/hwmon。IRPS5401 PMBus 分电源轨数据可以记录，但在 V1 中仅作辅助分析，不能与 VCC12 总功耗混算。

在首次正式测量前必须实机核验传感器身份、I2C 拓扑、分流电阻、驱动、板卡序列号、requested/effective sample rate（请求/有效采样率）和时间戳单调性。主机轮询频率高于传感器更新率时，不得把重复读数计作新增独立样本。

### 5.2 配对测量

每个算子使用自己的 exact-shape bitstream 和同一运行程序完成至少 3 组配对测量：

1. 板卡启动并完成热稳定；固定时钟、CPU governor/affinity、风扇模式、DMA 和缓存策略。
2. 加载该算子的同一 bitstream，等待稳定；编程过程不计入测量窗口。
3. 空闲采样至少 60 s：加速器不运行，但 bitstream、时钟和软件栈保持不变。
4. 活动采样至少 60 s：连续执行算子，实际完成次数 `N_completed >= 1000`；运行回执必须绑定 bitstream 和 runtime SHA256。
5. 记录原始 CSV、开始/结束温度、有效采样率、超时、失败和资源超限。不得只运行一次算子后读取一次功耗。

算子测量顺序应随机化或交错执行，避免温升和测量顺序成为混杂因素。idle 与 active 必须属于同一传感器范围和同一实现配置。

### 5.3 计算定义

设每组空闲时间加权均值为 `P_idle`，活动时间加权均值为 `P_active`：

```text
P_dynamic = P_active - P_idle

E_total/op   = 1000 × integral(P_active(t) dt) / N_completed
E_dynamic/op = 1000 × [integral(P_active(t) dt) - P_idle × T_active]
               / N_completed
```

能量单位为 mJ/op。`E_dynamic/op` 是 HW-NAS 的主能量指标；`P_dynamic × latency` 只在活动窗口接近 100% 占空比、且 latency 与功耗窗口具有相同边界时作为交叉检查，不能替代积分结果。若 `P_dynamic < 0`、有效采样不足或信号低于噪声/重复性门限，该行失败闭合，不截断成零，也不报告为通过。

## 6. 门禁与执行顺序

硬门禁候选字段为：`latency_board_ms`、LUT、DSP、BRAM、WNS/TNS、`p_dynamic_w` 和 `energy_dynamic_mj_per_op`。FF、总活动功耗、总能量和温度必须记录；是否成为淘汰门限需单独预注册。

正式顺序固定为：

```text
确认算子池
→ 冻结每种算子的合法形状
→ HLS + Route + bitstream + exact-shape micro-benchmark
→ ZCU104 板级 latency/power/energy 采样
→ 生成全新的 Hardware Cost LUT V1
→ 检查 9/9 算子行和搜索所需键的 100% 覆盖
→ 预注册并执行 Hardware Gate
→ 冻结仅由通过算子组成的新搜索空间
→ 新网络搜索
→ 重新训练搜索结果
→ 完整网络 HLS、Vivado 和 ZCU104 部署
```

局部 HLS、RTL 仿真、Vivado 功耗估计或 mock runtime 通过都不等于算子板级门通过。所有失败、超时和资源超限必须保留为正式行，不能从分母中静默删除。

## 7. 当前门状态

截至 2026-08-10，五个 Stage4 和四个 Stage2 算子身份及宏观精确形状可由当前源码确定；但 ZCU104 最新探测结果仍是 `DETECTED_PL_POWER_OFF`，板载 hwmon 路径、有效采样率、bitstream、板级 latency 和功耗均未完成实测。因此当前只能冻结测量合同，不能生成任何通过的功耗行，也不能启动搜索。

## 8. 官方依据

- AMD ZCU104 用户指南 UG1267：板卡电源系统和监测拓扑。
- AMD Zynq UltraScale+ MPSoC 数据表 DS891：XCZU7EV 器件资源总量。
- PYNQ PMBus 文档：`get_rails()`、`DataRecorder` 及 ZCU104 传感器配置支持。

