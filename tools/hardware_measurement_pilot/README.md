# 实验3 硬件测量试做

本目录独立保存`elastic_neck72_v1`的试做实现。所有结果仅用于流程验证，不进入正式成本表。用户最新范围为INT8主体、W8A16直接预测头及body8→head16边界；不测主体INT16，W16权重推迟到D06确认。当前网络与已检查工作簿均未确认concat必要性，默认排除。

当前14项＝INT8 PW/DW3/DW5/nearest/add各2映射（10项）＋W8A16头部PW（2项）＋I8→O16边界（2项）；每项5组整数夹具，共70组。旧attempt01的24项/120组仅保留为已被范围修订替代的本地软件探针，不能用于本次硬件进度。

已按授权更新原工作簿`E:/1paper汇报/声呐_HWNAS_全链条分阶段梳理_v2_算子上板前置_20261010.xlsx`：T21及W03/W04写入14项与实际PE/SIMD，D07 M列为“试做范围，非正式冻结”，D09 M列写草案说明及协议链接。“先确定事项”D06/D07/D09分别在第11/12/14行，F:H（最终值/确认/证据）保持空白，I列待冻结；D09保持DRAFT，执行本次pilot，实验5前正式冻结。备份、哈希与语义校验见`results/hardware_measurement_pilot_20261012/original_workbook_update.json`。

## 本地复现

在仓库根目录`E:\1\hwnas\hwnas`使用PowerShell。已有attempt不能覆盖；修改内核后创建新的attempt并重新生成全部绑定。

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.hardware_measurement_pilot.inventory
```

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.hardware_measurement_pilot.prepare prepare --output results/hardware_measurement_pilot_20261012/attempt_new
```

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.hardware_measurement_pilot.prepare native --output results/hardware_measurement_pilot_20261012/attempt_new
```

```powershell
.\.venv_cuda\Scripts\python.exe -m unittest tools.hardware_measurement_pilot.test_reference tools.hardware_measurement_pilot.test_measurement -v
```

本地native成功仅证明同一C++与Python参考的一致性，不是Vitis C仿真、RTL或板上结果。通过HLS工具执行每项生成的`run_hls.tcl`才是Vitis C仿真与综合；之后使用生成项的`build_shell.tcl`构建独立集成设计。始终保留工具日志和失败尝试。程序拒绝覆盖已有Vivado结果目录。

## 统计板上原始记录

每行CSV字段由`measurement.CSV_FIELDS`给出，绑定case/numeric/bitstream/expected_output四个SHA256、board_id、clock_hz、batch_id、iteration与warmup/measured阶段，保存compute_ns、transfer_ns、call_ns及输出逐元素比较信息；短算子还保存group_id/group_size/group_call_total_ns。数值单位ns。预热与正式样本分行保存，不能删除慢样本。

manifest包含cases身份列表和batches元数据。每项至少3个独立批次，每批重新加载bit或重启runtime，50次预热后至少1000次正式调用。短算子N=16时向上补齐为1008次（63整组），N=1时1000次；跨批N不同则host按N分层统计并标明可比性限制。记录每批PL起止频率、CPU governor、起止温度和时间；可写policy支持时设performance，结束恢复。接口或温度不可用时如实保留偏差。PL compute/transfer逐批及合并CV≤0.05；host只描述批间中位数稳定性与P95，不默认设5% CV门槛。host每组总时/N仅算一个统计观测，不能按N复制成独立观测。输出拷贝在计时窗内，逐元素比较/hash在窗外。全部夹具、文件哈希、RTL相位和实际PL时钟仍须单独核验；统计PASS不等于硬件全链通过。

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.hardware_measurement_pilot.measurement --csv results/hardware_measurement_pilot_20261012/attempt_new/board/timing_raw.csv --manifest results/hardware_measurement_pilot_20261012/attempt_new/board/timing_manifest.json --output results/hardware_measurement_pilot_20261012/attempt_new/board/summary.json
```

具体协议以生成的`D09_protocol.md`和工作簿D09页为准。phase1/3仅称加载/写回区间；phase2称片上计算区间（含局部控制开销）。当前证据采用生成FSM/AXI adapter静态检查、HWH完整时钟/复位、独立timer RTL和同bit五夹具板端计数；完整动态集成RTL仿真为NOT_RUN，不声称仅算术指令周期。默认接口是CHW完整帧小端序，不复用历史流接口成本。

## 生成工作簿和状态报告

报告依赖openpyxl 3.1.5。可将依赖隔离安装到本次结果目录，不改全局或训练环境：

```powershell
.\.venv_cuda\Scripts\python.exe -m pip install --target results/hardware_measurement_pilot_20261012/_deps openpyxl==3.1.5
```

```powershell
.\.venv_cuda\Scripts\python.exe -m tools.hardware_measurement_pilot.report --attempt results/hardware_measurement_pilot_20261012/attempt05
```

输出：`results/hardware_measurement_pilot_20261012/Experiment3_硬件测量试做.xlsx`、`report.md`、`D09_protocol.md`、`D07_mapping_recommendation.md`、`extrapolation.json`、`report_manifest.json`。报告工作簿为独立新文件，生成报告不改原工作簿。未测数值留空；统计值是Python脚本可复现的导入结果，没有未计算的Excel公式。

另存`report_data.json`便于复核逐case证据、同算子low/high实测和条件排程。异常注释在独立CSV与规则JSON中，按case/batch/预热或正式/scope计算median+6×1.4826×MAD；host每group只计一个观测，MAD0不作稳健异常判断。全部原行保留，剔除数0，标注不改统计或门槛。索引为`attempt05/anomaly_annotations/selected_runs_index.json`。

报告读取指定attempt下的case、native汇总和`remote_evidence/**/status.json`；按当前case哈希校验阶段身份，旧源码的PASS标为STALE_SOURCE_BINDING。板上逐项结果置于`cases/<case_id>/board/<run目录>/`，保留原生`summary.json`、`timing_manifest.json`、`timing_raw.csv`和`batch1..3/correctness_fixtures.json`等所有文件；`board/selected_run.json`显式指定`{"run_directory":"<run目录>","evidence_kind":"BOARD_MEASUREMENT"}`，不递归搜索probe。完整板上PASS要求summary和CSV哈希绑定当前case、3批全部5组夹具通过及`hardware_measurement_eligible=true`；环境条件、RTL/实际时钟另列。

## 外推与映射缩减

当前W8A8/HEAD_W8A16域按结构/精度去重有65种原语；加入待D06确认的HEAD_W16A16才为69种。六个现有reference策略名是选择器，可重复选择同一物理映射；不能据此声称390个或414个不同综合任务。先绑定并去重不同物理配置，再按算子/精度/规模/映射分层测量完整阶段耗时。未覆盖类别和板上阶段未完成时，全量天数保持null。

`buffer_coverage.json`及独立复核显示：当前整帧BRAM模板有18个结构键仅激活存储下界即超过器件624 BRAM18，另4个键缺模板；剩余43个也未因此证明fit。INT8 PW 16×320×320→48×320×320仅输入/输出帧合计6.25MiB。此限制针对当前实现，不是层或网络不可实现结论；分块、流式或URAM实现需重新绑定测量。P4样本平均时长×65×映射数只能作实现覆盖已解决后的条件情景。

`measurement.extrapolate()`需要每项完整的python_reference、csim、hls_synthesis、route、bitstream、board_test秒数及正确性/稳定性通过；失败尝试的已消耗时间也计入。源阶段名须显式归并，不能重复计入父级总耗时和子阶段。串行工作量与理想并行下界分别报告；真实日历超过21天时，才依据实测点考虑缩减到2–3种不同物理映射，保留低/高并行及必要的非支配折中点。

派生报告另用`schedule_scenarios.schedule_pipeline()`记录完整14项P4 mix的条件排程：实际HLS+route进程耗时送入3个worker，之后由单板串行处理probe与三批timing进程占用；不把板端时间除3。明确允许“已验证数值/相位但有CPU条件偏差”的观测参与条件时间，`board_accepted`仍为false，完整全量时间仍为null。8h/24h仅称“已测工具+板端worker阶段条件工作日”，未计桥接网络、人工复核、恢复和额外失败等开销；完整流程日期仍为空，不能根据该条件范围未超过21日就断言全流程不超过3周。建议先2种不同物理配置，必要时加第3个实测非支配点；六策略并不要求六个独立build。

自动driver的事件记录另作真实耗时描述：只导入已完成且绑定当前case、raw CSV及summary哈希的自动案例，包络为probe上传开始至timing导入结束。减去probe+timing worker的残差包括桥接、轮询、本地校验及编排，并非纯网络开销；不把首个手工案例混入均值，不为凑齐14项猜数，不将这些包络添加到已限定的条件排程。
