# 192 v2 分阶段接口

本模块接入冻结的 `sonar_backbone_neck192_v2`：原192个骨干/颈部组合、每尺度6通道质量头、CIoU、无DFL、EMA。原72、192 v1、阶段B及历史硬件产物保持原样。本模块没有搜索、训练、综合或烧板命令。

## 统一身份

`contract.model_contract()`核对冻结配置、192行结构清单、完整模型合同及模型源码依赖。`make_identity()`进一步绑定：

- 完整结构与检测头合同；结构短ID只作为显示字段。
- checkpoint文件、实际EMA张量、seed、epoch、训练执行manifest。
- 量化档位、训练集校准清单及数值合同。
- 具体实现、PE/SIMD、缓冲/分块、数值和接口合同。

所有后续工件带同一`identity_sha256`。诊断身份不能通过改一个PASS字段升级为正式候选。

最终部署权重选择固定为**第80轮EMA**，不再根据结果选择epoch。76–80轮平均值继续作为结构比较统计；正式交接重新计算第80轮该checkpoint的验证指标。`model.load_checkpoint()`严格读取EMA负载，拒绝原始训练权重回退、错误头、错误结构和不完整正式训练回执。

最终部署seed固定为**42**；seed43、44用于配对确认，不根据验证结果挑选seed。正式候选应部署选中结构的`seed_42/ema_final.pt`（epoch80），并以实际文件SHA256和EMA张量SHA256明确是哪一份权重。当前没有为其伪填文件哈希。

正式身份还必须匹配`contract.REGISTERED_FINAL_EMA_ARTIFACTS`中的已审核训练产物：完整模型合同、seed、第80轮、实际EMA张量、checkpoint文件、training result及execution manifest的哈希。本次没有找到已完成的正式训练证据，注册表为空。不能通过CLI提交自签manifest或修改epoch/PASS字段注册新权重；后续真实训练完成后须独立核验并在单独代码修订中固化。

## 模块与证据

|模块|生产/消费的实际工件|边界|
|---|---|---|
|`graph.py`|从真实模型模块生成完整四尺度原语图|包括固定blur、PAN、分组卷积、SiLU及4+2拼接；不是旧21通道线性头|
|`quantization.py`|校准、整数参数、逐节点原始整数输出、独立数值见证|原始图像仅来自绑定train清单；量化精度保持需要另行full-valid评价|
|`hardware.py`|192原语覆盖、具体映射、板测导入、缺测队列和留出误差|试做/不合格记录保留但不能进入正式库；算子汇总不是整网实测|
|`confirmation.py`|三seed FP32/PTQ逐图预测及来源回执|独立重算COCO mAP50–95和配对准入；平均统计不能充当单checkpoint指标|
|`handoff.py`|整数二进制、参数/激活地址布局、加载清单、板端原始结果|软件传输ABI须由实际bitstream实现并提供绑定证据；生成包不等于已经上板|

已实现`W8A8`、`HEAD_W8A16`、`HEAD_W16A16`。混合档位有显式body8到head16边界转换，头部SiLU查找表覆盖65536个输入值。精度支持范围以`quantization.SUPPORTED_PROFILES`及新交付报告为准。旧T6测量使用另一数值/接口合同；只有精确匹配的已验收证据才能导入当前候选。

## 有界运行入口

从仓库根目录运行；任何输出都应使用新的目录。模型构建使用原项目`.venv_cuda`加独立安装的本地验证依赖，未修改原训练环境：

```powershell
$env:PYTHONPATH = 'E:/1/hwnas/hwnas/results/sonar_192_chain_20261015/runtime_overlay;E:/1/hwnas/hwnas;E:/1/hwnas/hwnas/src'
$env:YOLOV5_CONFIG_DIR = 'E:/1/hwnas/hwnas/results/sonar_192_chain_20261015/runtime_cache/author'
& '.venv_cuda/Scripts/python.exe' -B -m tools.sonar_192_chain.run --help
```

`contract`仅核对身份；`prepare`接收一份实际checkpoint，执行校准、整数验证和软件参数包生成；`package`消费已有量化导出；`confirm`重算三seed回执；`verify-board`读取板端返回工件，**不会连接板卡**。

正式`prepare`必须提供80轮EMA、完整训练result、执行manifest、train64校准和冻结映射。量化导出后先执行三seed准入，再用`package`消费同一份导出；未提供确认报告时，`prepare`保存量化工件并明确阻断正式交接。已有一轮smoke只能使用`--diagnostic`。真实训练/量化/板端精度缺失时，不以测试夹具补数。

## 验收层次

1. 软件合同和错误来源拒绝测试。
2. 既有真实EMA权重、真实train输入的完整整数图一致性。
3. 软件参数包回读、地址布局及板端回传解析测试。
4. 实际正式候选：三seed量化准入、物理实现绑定、真实整网板端数值。
5. 同一候选的完整valid精度、端到端延迟、功耗/能量各自实测。

前三级通过不能替代后两级。本次结果和执行源快照保存在`results/sonar_192_chain_20261015/`；正式搜索仍禁用。

当前固定候选诊断为`035462 / seed42 / 既有第1轮EMA / train1校准`。三档分别比对109、113、113个节点（含输入），50,438,400、52,070,400、52,070,400个整数元素均无差异；参数包已经独立回读。该结果只验证软件接口和这一个输入的算术一致性，不能替代train64校准、三seed最终权重准入或真实上板。
