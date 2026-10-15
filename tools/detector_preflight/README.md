# 检测器硬件前置适配

> 2026-09-22 执行更新：本目录现已增加 `h2_dma.cpp`、真实 PYNQ DMA 驱动及独立构建工具。PW/FPN 已完成真实板卡零失配验证；完整图 A 已通过 HLS C 仿真、6 项 RTL 回归、实际 187.512 MHz 布线和真实板端 34 层零失配验证，三个检测头回传核验通过。当前证据见 `results/detector_execution_20260922/README.md`。下文的 HOST_PLAN_ONLY/UNIMPLEMENTED 指原 `prepare.py` 与固定 shell 适配层，不能用于判断新后端状态。新硬件是功能验证映射，不代表搜索最终候选或 H3 成本。

本目录独立于正在运行的 `sonar_joint_v1` 冻结源码集合。使用已有 H2 数值/接口合同和 graph_A/B/C 合成权重包，不读取训练/验证/测试图像，不挑选最终候选，不修改旧 shell，不执行 FPGA 烧录。

## 已实现

- `prepare.py`：验证 H2 canonical JSON 合同哈希和参数包 SHA256，生成逐节点串行调度描述符。
- 按最后一次消费者的生命周期分配 DDR 相对偏移；同一操作的输入与输出不重叠，FPN 跨层特征保留到最后一次使用，所有检测头保留到 CPU 消费阶段。
- 特征流沿用 H2 HWC INT8、256-bit DMA、64-byte 地址对齐、帧尾补零和满 TKEEP/TSTRB 规则。P5 真实输出 8400 bytes、DMA 输出 8416 bytes。
- 参数区与可复用激活区分离，权重、偏置和每层 128-byte 描述记录的访问检查不越界。
- `shell_adapter.py`：可注入驱动后端的固定帧 shell 主机传输适配。实际操作前核验 shell 身份、版本、位宽、固定拍数和空闲状态；接收先于生产者启动；检查 DMA/core 完成、错误和实际拍数；异常时要求后端停止 DMA 和 core。
- 该适配器不会把旧 shell 的只读帧长寄存器当成可写动态帧长寄存器。检测器帧不匹配时，在任何控制写入之前拒绝启动。

## 实测代码核查结果与集成缺口

| 层 | 可复用部分 | 尚需硬件实现/证据 |
|---|---|---|
| PS/DDR/DMA | 旧 BD 的 HP0/SmartConnect、256-bit simple DMA、26-bit length 配置 | 当前目标器件构建、地址映射导出和 shell 资源实测 |
| 固定算子 shell | AXI-Lite 状态、错误、计数器、先接收后发送顺序 | 动态帧长/节点选择 ABI；现有 INPUT_WORDS/OUTPUT_WORDS 是综合期参数 |
| 图调度 | 本目录 DDR 生命周期计划、依赖和参数切片 | 真正的逐节点执行后端；核身份、位宽、参数地址/控制寄存器映射 |
| FPN | H2 aligned Add 和上采样语义 | 双输入数据通路，低分辨率 DDR scratch 的硬件装载；一个 MM2S 流不能冒充两个独立输入 |
| 输出 | H2 21-channel 重打包协议和三/四尺度描述符 | repacker 接入完整数据通路与多输出完成信号 |
| 精度 | 本工具仅 H2 INT8 帧规划 | W8A16/W16A16 的激活帧宽、算术/参数包和 loader 合同需要单独扩展；不能把 INT8 内存计划用于混合精度 |

建议后续硬件集成沿用串行 DDR 中间结果方案，首先实现一个可复用算子执行后端和双输入 Add 通路，再绑定图调度；不直接把所有 HLS 算子同时实例化，也不把本工具的 DDR 需求当成 BRAM 需求或物理连续内存可用量。

## 执行边界

生成的是 **HOST_PLAN_ONLY**。`backend=UNIMPLEMENTED` 明确保留在每个节点中；现有接口不提供物理内存分配或 MMIO 实现。主机传输适配是单事务工具，不是完整检测器执行器。

后端接入时必须从实际分配的连续 DMA buffer 获得物理地址与长度，核验参数包、已加载 bitstream 和算子身份，进行 cache ownership 转移和内存屏障；不能使用本计划的相对偏移直接访问物理地址。超时后的 abort 必须保证 core 和两个 DMA 通道停止后才能释放/复用内存。

最终候选未绑定、训练权重未绑定、shell reservation UNKNOWN、整网数值一致性/RTL/P&R/板端延迟/功耗均未建立。H2 合成权重包不是搜索得到的模型。

## 复现

仓库根目录执行；输出目录必须不存在，避免覆盖证据：

```powershell
.venv_cuda/Scripts/python.exe -m pytest tests/test_detector_preflight.py tests/test_detector_shell_adapter.py -q
.venv_cuda/Scripts/python.exe tools/detector_preflight/prepare.py --output results/detector_preflight_NEW_RUN
```

测试使用合成 byte tag 回放验证内存生命周期，不声称重新执行完整整数卷积，也不声称通过 RTL 时序验证。
