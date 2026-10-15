# 硬件证据接口

`evaluate(genotype, quantization_plan, context, *, cache_path=None,
evidence_path=None)` 只读取本地证据。它不启动 HLS、综合、布局布线、板端程序或远程任务。
没有相同身份的证据时返回 `NEEDS_MEASUREMENT`，`metrics={}`，不会把缺失值补成零。

## 身份与证据层级

`HardwareContext` 为不可变数据类，包含 `device`、`clock_mhz`、
`mapping_policy`、`backend`、`tool_version`、`scope`，以及可选的
`checkpoint_sha256`、`quant_artifact_sha256`、`bitstream_sha256`、
`implementation_sha256`。所有字段均纳入 `context_hash`。
`arch_hash` 来自 Genotype，`q_hash` 来自量化规划合同，不代表训练权重或已校准产物。

| backend | scope | 所代表的证据 |
| --- | --- | --- |
| `hls_report` | `operator_hls` | 单算子 HLS 估算，不代表整网 |
| `hls_report` | `full_hls` | 完整实现 HLS 估算，不代表布线或板端运行 |
| `route` | `route` | 对应实现的布局布线结果 |
| `board` | `board` | 对应权重、量化产物和位流的板端结果 |
| `power` | `power` | 对应权重、量化产物和位流的独立功耗测量 |

未测量上下文可不提供工件摘要，并使用 `tool_version="UNMEASURED"` 查询未知状态，
避免为了构造对象而虚构工件或工具版本。实际导入时必须提供实际工具版本；
`board`/`power` 必须提供 checkpoint、quant_artifact、bitstream 三个实际文件摘要，
HLS/route 必须提供 `implementation_sha256`。该实现工件应是明确包含源码、指令、映射及编译常量的
可复现实现包；若常量权重或其它产物参与编译，必须将它们绑定到实现包，或另外提供相应摘要。
所有已提供的摘要都必须在来源列表中指向实际文件并复核内容。

`READY` 只代表当前声明层级的必要字段完整、身份与来源验证通过。
它不表示通过模型正确性、完整性、板端可执行性或功耗验收。
`pareto_eligible` 只允许完整四项指标的 `full_hls`、`route`、`board` READY 结果；
搜索仍须在配置明确指定的同一证据层级内比较，并应用自身预算和验收门。
`operator_hls` 和独立 `power` 结果不能作为整网 Pareto 候选。

## 规范化 JSON 合同

格式版本为 `sonar_hwnas.hardware.v1`，顶层字段必须恰好为：

```text
schema_version, arch_hash, q_hash, context, context_hash,
evidence_level, timing_status, metrics, provenance
```

`context` 必须等于 `HardwareContext.to_dict()` 的完整输出（包括值为 null 的可选字段），
`context_hash` 必须匹配，`evidence_level` 必须等于 `context.scope`。
`timing_status` 为 `PASS`、`FAIL` 或 `UNKNOWN`。失败返回 `TIMING_FAILED`；
未知或必要指标缺失返回 `INCOMPLETE`，均不可进入 Pareto。
`power` 文件中的 PASS 表示对应位流在所声明时钟下完成该测量；功耗本身不证明时序闭合。

`metrics` 中每项必须为 `{ "value": 数值, "unit": 规范单位 }`：

| 字段 | 规范单位 | 数值约束 |
| --- | --- | --- |
| `latency_us` | `us` | 有限正数 |
| `lut` | `count` | 非负整数 |
| `dsp` | `count` | 非负整数 |
| `bram18` | `18k_blocks` | 非负整数，18 Kbit 块计数 |
| `power_w` | `W` | 有限正数，仅供 `scope=power` |

四个非 power 层级要求 latency_us/lut/dsp/bram18 齐全才 READY。
功耗层级只接受 power_w；不能把布局布线的估算功耗标为实测功耗。
未测字段直接省略，禁止 null、NaN、Infinity、布尔值、负数、未知名称、隐式单位或重复 JSON key。
资源数为零是合法测量，不会代替缺失值。

`provenance` 必须恰好包含 `kind`、`producer`、`sources`。
`kind` 与 scope 相同，`producer` 为非空的实际生产工具/脚本标识。
`sources` 是非空列表，每项恰好包含 `role`、`path`、`sha256`。
`path` 可为绝对路径或相对规范化 JSON 所在目录的路径；每个 role 唯一。
需要一个 `role="report"` 原始报告文件；所有上下文中的工件摘要必须有同名 role：
`checkpoint`、`quant_artifact`、`bitstream`、`implementation`。
规范化 JSON 不能把自身当作原始来源。每个来源文件的内容 SHA256 都会重新计算。

这个合同验证本地数据的身份、单位、完整性和内容链；它不提供第三方签名或实验真实性证明。
board/route/power 的数值由已有可信测量流程导出，再由此接口核验来源链。
不能因为某份任意文本的哈希有效就声称该实验或其上游正确性已经独立复现。

## HLS XML 与 sidecar

`hw.backends.hls_report.parse_hls_xml(xml_bytes, scope=...)` 是不含文件 I/O 的纯解析器，
`parse_hls_report(path, scope=...)` 是只读文件包装器。
二者只接受显式的 `operator_hls` 或 `full_hls`，不会把 HLS 提升为 route/board/power。
解析器只提取 `AreaEstimates/Resources` 中已使用的 LUT/DSP/BRAM_18K，
不读取 `AvailableResources` 作为消耗。
延迟采用 `Worst-caseLatency × TargetClockPeriod / 1000` 转为微秒；
缺少周期/目标时钟时可独立解析带显式单位的 Worst-caseRealTimeLatency，
但 evaluate 仍要求完整的设备、目标时钟和工具身份。
`EstimatedClockPeriod > TargetClockPeriod` 即 FAIL。
缺少时序信息保持 UNKNOWN，不选择更有利的实时延迟或估算时钟掩盖失败。

直接把 `csynth.xml` 传给 evaluate 时，必须同时存在 `csynth.xml.json`，
内容为上述完整 JSON，且 report 来源必须绑定该 XML。
也可直接把同样的规范化 JSON 传给 evaluate。
HLS JSON 的 metrics、timing_status 还必须与原始 XML 重新解析结果一致；
XML 的 Part、TargetClockPeriod 和 ReportVersion/Version 必须与 context 一致。
因此仅改 sidecar 不能改变实际资源、时序、设备或工具身份。
原始 XML 本身不证明函数覆盖整网；scope 的真实性仍需对应实现包和既有验收记录支持。

## 缓存

SQLite 主键为 `(arch_hash, q_hash, context_hash)`。
缓存只保存证据路径和已核验的来源摘要，不把数据库中的 metrics 当作可信结果。
缓存命中也重新读取全部来源文件、复核摘要并重新验证规范化合同。
来源被修改或删除时返回 `INVALID_EVIDENCE` 并清除该缓存项，保留空 metrics。
显式传入新 evidence_path 时优先验证新文件；缓存缺失不会触发硬件执行。

测试中的报告和工件均为临时合成样例，仅验证软件合同，不是研究测量结果。
