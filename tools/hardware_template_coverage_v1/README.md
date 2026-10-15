# 待办6：当前65种原语的分块实现覆盖

范围是 `elastic_neck72_v1` 当前登记的 W8A8/HEAD_W8A16 共65个结构与精度键。
D01如果改变骨干、连接、通道或算子域，应重新盘点；本工具不声明覆盖尚未确定的新域。
本版使用合成整数输入和权重，未开展训练、量化校准或正式成本表采集。

`inventory.py` 保留全部65个精确原语ID，并绑定13个代表：大尺寸和混合头PW、
DW3/DW5各stride1/2、最近邻、加法、主体到头部重定标、stem卷积，以及全部三个池化尺寸。
原缺口集合为18个已有整帧模板超容量的键和4个缺模板键；缺模板中两项也超容量，
二者并集仍是22个键。完整列表及数组位数在新结果目录 `coverage_contract.json/md`。

## 实现与计时

`kernel.cpp` 将输入和输出完整帧保存在DDR，仅分配2行×32列的空间块。
标准卷积保留一个空间patch的全部输入通道和一个输出tile；深度卷积、池化和元素操作
每次缓存4个通道，卷积patch包含stride/padding所需的halo，尾块仅写有效坐标。
INT8权重、INT32回绕、INT64重定标、nearest-even和饱和规则保持明确；
maxpool是原始整数最大值，不进行重定标。所有fixture采用真实activation，
ReLU6整数上限12仅对应合成scale，不能当作训练模型的标定参数。

新的P3T1计数器接受 `1→2→3→1` 多块循环，最终 `3→0` 才提交一次调用计数。
compute为所有phase2区间之和，transfer为所有phase1/3区间之和，包含块间控制开销。
完整RTL相位审核、HWH时钟/复位连接和同bit五夹具probe必须先通过，才允许三批计时。
单独timer RTL测试、C仿真或HLS资源估计均不能替代这些门槛。

## 复现命令（PowerShell，仓库根目录）

已有attempt不可覆盖。重新执行时先选择新的 `attemptNN`。

```powershell
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.inventory
.\.venv_cuda\Scripts\python.exe -X utf8 -m unittest tools.hardware_template_coverage_v1.test_reference tools.hardware_template_coverage_v1.test_tiled_kernel -v
.\.venv_cuda\Scripts\python.exe -X utf8 -m unittest tools.hardware_template_coverage_v1.test_phase_timer tools.hardware_template_coverage_v1.test_phase_audit -v
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.prepare prepare --output results/hardware_template_coverage_20261013/attemptNN
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.prepare native --output results/hardware_template_coverage_20261013/attemptNN
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.remote deploy --attempt attemptNN
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.parallel_build launch --attempt attemptNN --stage coverage --jobs 3 --limit 65
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.remote status --attempt attemptNN
.\.venv_cuda\Scripts\python.exe -X utf8 -m tools.hardware_template_coverage_v1.remote fetch --attempt attemptNN --include-bit
```

`coverage` 只对13个源码绑定的代表做HLS＋布线，其余52个只做HLS。
HLS逐项校验五组输出二进制、器件及实际资源估计；资源超限保留证据并阻止布线。
构建进程最多3个，板端执行始终串行，失败、中断及条件偏差保留。

代表项布线完成后，用 `package_board_case` 生成不可变包，运行 `phase_audit inspect`。
独立审核精确的结构报告并保存其SHA绑定的review后，才调用 `run_board_case`：
该程序依次上传probe、比对五夹具、完成相位绑定，再重新加载bit执行3批50预热＋至少1000计时。
板端输出拷贝在计时窗内，逐元素比较在窗外。通过 `--help` 查看这些逐项命令参数。
CPU仅支持userspace时记录条件偏差，不把performance写成已达到；试做结果保持非正式资格。

板端启动会检查已知旧试做目录的任务锁、系统条件预约和运行进程；发现占用时停止启动。
2026-10-13续跑版还要求 `--ownership-config` 指向当前机器、启动周期和任务身份绑定的共享预约。
`board_ownership.py` 与D09共用机器级预约路径，原子创建预约，并在启动、worker入口、
runner入口及每次重新加载bit前核验预约、精确自有进程与已知任务锁。检查均在计时区间外。
自有进程绑定PID、启动计数、完整参数、工作目录、源码和请求SHA；不能仅凭PID免除冲突。
这是两个已知工作流共同遵守的预约协议，不是操作系统层面的全局互斥。
运行器还核对PYNQ记录的bit路径、加载时间戳和P3T1标识。运行中身份变化会永久隔离该次
会话的DMA缓冲；另一个设计返回idle不能解除隔离。这些检查不能代替多个工作流共同遵守的
板卡独占约定。发生冲突后，应先协调板卡与隔离进程的恢复，再用新job完整重跑三批。
`p3t1_pl_epoch_guard_v1`的单次`call_ns`包含启动前、完成后的两次PL身份检查；外层成组时间
还包含调用入口与计数采集后的检查。该主机时间不能与旧版未保护协议直接混合比较。

2026-10-13的首次边界模板计时遇到同板重编程冲突，整run无效。原始记录和只读诊断保存在
`attempt01/board_incidents/run_20261013T043229325640Z/incident.md`；此前五组probe与不完整计时
分别保留，不能从部分成功调用推导完整计时通过。报告会列出这些未获验收的历史尝试。

## 验收范围

全部65个形状需要完整尺寸native、Vitis C仿真和HLS容量门通过；
13个代表（含全部4个缺模板键）另需完整布线、bit生成、逐位板测和三批计时。
非代表项不声称已有布线/板测数据，代表成本不填给未测形状。
代码覆盖、HLS估计、布线资源、板测与正式成本资格分别记状态。
所有输入、源码、数值协议、bit/HWH和原始结果由SHA256绑定，旧试做目录保持独立。

## 当前执行状态与证据入口

截至2026-10-13北京时间22:38，完整验收已推进至7/13个代表、21000次有效计时、21个独立批次。
第7项DW5 stride1在21:31板端PASS后发生客户端快照等待超时，队列于21:33停止；
现有归档的56份文件已取回，完整逐位输出与三批统计复算通过，旧测量没有重新运行。
原始错误、空下载目录与成功恢复证据均保留在独立incident中。

剩余6项已于22:37在新有限队列启动，首项nearest的probe正在执行。新队列启用
`--recover-terminal-snapshot`：只在明确记录timing PASS、客户端快照超时且下载目录为空时，
尝试一次现有归档下载、完整导入审核及事件归档。取回工具还要求原worker/runner消失、
任务锁不存在。任一恢复检查失败仍停止；RUNNING、FAIL、未知设备状态及已有结果均拒绝恢复。
不重启测量，不新建远端归档；五项队列保护回归测试通过。四份已审查板端运行器源码保持原SHA。

- [七项完整复算覆盖报告](../../results/hardware_template_coverage_20261013/reports/run_20261013T143812466161Z/report.md)
- [余下六项新队列启动记录](F:/hwnas_t6_storage_20261013/queue_20261013T143733/launch_receipt.json)
- [DW5已通过测量的快照恢复证据](F:/hwnas_t6_storage_20261013/attempt01/board_incidents/fetch_timeout_t6_r2_dw5_d6c3bc140668_20261013T1244_timing/incident.json)

## 2026-10-13 20:43历史状态（已由上段更新）

截至2026-10-13北京时间20:43，已有6/13个代表完成完整逐位板测、三批计时和原始统计复算，
共18000次有效计时、18个独立批次；65项native/C仿真/HLS容量门与13项布线仍全部通过。
CPU条件偏差保持原记录，正式成本资格仍为false。

第7项DW5 stride1的旧任务在20:14通信异常后停止回传。用户随后重启了板卡；新启动标识已确认，
旧worker/runner均不存在。40份原始文件已完整取回并校验，该run只有部分批次，保留且不计验收。
两个精确自有旧锁已归档，新启动预约已建立；没有将旧部分记录拼接进新run。

E盘空间不足导致本轮一次重新打包未完成。用户明确批准迁移后，attempt01已转至
`F:\hwnas_t6_storage_20261013\attempt01`，原E盘路径由目录联接继续访问；17244个原文件
逐项SHA校验通过，迁移前原目录另保存在`attempt01_original`。
报告比较只为已存在且解析为同一目录的snapshot/output路径接受别名，所有其他字段仍严格校验；
28项验收/报告回归测试通过，迁移后的六项原始统计复算通过，报告issues为空。

剩余7项已在20:43作为有限后台串行队列启动：新DW5 stride1、nearest、requant_add、stem和三个pool。
队列跳过已验收的6项，遇到非零退出即停止，不自动重试测量；后台运行不代表完成验收。
本轮按用户要求结束前台实时监控。队列完成后仍需更新最终报告并在确认无占用后释放本任务预约。

- [迁移后六项完整复算报告](../../results/hardware_template_coverage_20261013/reports/run_20261013T124233423556Z/report.md)
- [本轮后台队列启动回执](F:/hwnas_t6_storage_20261013/queue_20261013T1244/launch_receipt.json)
- [本轮七项串行执行输入](F:/hwnas_t6_storage_20261013/attempt01/board_execution/serial_20261013T1244/inputs.json)
- [迁移后完整文件与源码清单复核](F:/hwnas_t6_storage_20261013/post_migration_validation.json)
- [本轮重启恢复参数与局部差异审查](F:/hwnas_t6_storage_20261013/second_reboot_recovery_20261013T1235/binding_review.json)
- [本轮新预约](F:/hwnas_t6_storage_20261013/second_reboot_recovery_20261013T1235/actual_reserve/new_ownership_config.json)
- [新运行包索引](F:/hwnas_t6_storage_20261013/attempt01/runtime_packages/20261013T123703283069Z/index.json)
- [旧DW5中断快照校验](F:/hwnas_t6_storage_20261013/interrupted_dw5_snapshot/verification.json)

## 2026-10-13 18:25历史状态（已由上段更新）

截至2026-10-13北京时间18:25，重启后的板卡已通过机器、启动标识和占用核查。
原中断任务的52个文件已取回并校验，两个精确绑定的旧锁已按原字节归档；本任务已建立新预约。
旧任务缺少完整三批记录，保留原样且不计验收。恢复过程未重启、终止进程或编程PL。
本段状态不能替代后续逐项验收报告；待办6的整体验收仍以13项完整板测和计时为门槛。

2026-10-13本次构建已结束：65/65 native、325组完整尺寸测试、65/65 Vitis C仿真与HLS容量门、
13/13代表布线通过。共享预约版13包通过运行器、源码和独立静态相位结构审查。
新边界实例和最大1×1卷积已完整通过三批计时与本地统计复算：共6000条计时、30组完整输出逐位一致。
最大1×1卷积在原测量PASS后发生客户端快照等待超时；原归档已完整恢复且复算通过，未重新测量。
其余11项正在串行执行；遇到非零退出即停止，不自动重试。混合精度预测头1×1卷积正在计时。
此前旧边界计时因同板重编程失效；本轮网络中断原因未定，不能与旧冲突混为同一故障。
目前验收计入的完整板测/计时为2/13；CPU仅提供userspace，未满足performance条件，正式成本资格仍为false。

12:53:41曾观察到两任务DMA隔离和本任务runner保护性暂停，此状态属于历史故障证据。
续跑前已确认板卡重启、旧进程消失；另一任务完成系统恢复核对、条件锁归档和共享预约释放，
43项独立交接检查通过。15:19:24的实时检查确认旧预约和四项已知任务/条件锁均不存在。
新的待办6预约在15:26:55原子创建，前后占用检查无冲突。

- [中间覆盖报告（2/13板测快照）](../../results/hardware_template_coverage_20261013/reports/run_20261013T102518292061Z/report.md)
- [新边界实例完整统计](../../results/hardware_template_coverage_20261013/attempt01/cases/t6_body8_to_head16_3e96d4a01dc5_tiled/board/t6_r3_boundary_20261013T0926_timing/summary.json)
- [余下11项串行执行记录](../../results/hardware_template_coverage_20261013/attempt01/board_execution/serial_20261013T1018/inputs.json)
- [客户端快照超时与成功恢复证据](../../results/hardware_template_coverage_20261013/attempt01/board_incidents/fetch_timeout_t6_r2_conv1x1_0bb0f100b132_20261013T0951_timing/incident.json)
- [原任务恢复与有限归档审查](../../results/hardware_template_coverage_20261013/attempt01/board_resume/reboot_recovery/run_20261013T082500Z/REVIEW_AND_USE.md)
- [本轮新启动预约回执](../../results/hardware_template_coverage_20261013/attempt01/board_resume/reboot_recovery/run_20261013T082500Z/actual_reserve/receipt_reserve.json)
- [新预约与首项板测证据复核](../../results/hardware_template_coverage_20261013/attempt01/board_integration_reviews/reboot_actual_20261013T0952/summary.json)
- [本轮网络中断停点与原job核查入口](../../results/hardware_template_coverage_20261013/attempt01/board_resume/interruption_20261013T080428532535Z/status.md)
- [22项缺口的实际HLS资源对照](../../results/hardware_template_coverage_20261013/attempt01/resource_summary/run_20261013T045843957268Z/summary.md)
- [共享预约版13个运行包索引](../../results/hardware_template_coverage_20261013/attempt01/runtime_packages/20261013T082017966217Z/index.json)
- [13项独立静态结构审核索引](../../results/hardware_template_coverage_20261013/attempt01/phase_reviews/20261013T050027Z/root_independent_reviews/index.json)
- [板卡冲突原始事件](../../results/hardware_template_coverage_20261013/attempt01/board_incidents/run_20261013T043229325640Z/incident.md)
- [独立交接审查](../../results/hardware_template_coverage_20261013/attempt01/board_resume/handoff_review_20261013T0722/review.json)
- [旧预约释放后的实时状态](../../results/hardware_template_coverage_20261013/attempt01/board_resume/run_20261013T071918336860Z/receipt_handoff_readonly.json)
- [本任务新预约回执](../../results/hardware_template_coverage_20261013/attempt01/board_ownership/20261013T072648459165Z/receipt_reserve.json)

原job `t6_resume_boundary_20261013T0740_timing` 的中断快照及原锁归档均保留。
每项新测试使用独立job和快照；旧失败job、旧package和部分计时不能覆盖或续填成完整通过。
