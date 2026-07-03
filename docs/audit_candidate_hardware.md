# 候选表示与硬件代价审计归档

归档时间：2026-05-17

审计状态：已完成静态审计；未修改源码；未运行训练、HLS、Vivado。

前置引用：入口与配置闭环审计确认当前主入口为 `run_search.py`，主 formal LUT 配置样例为 `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:60` 和 `E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:61`。

## 候选表示流转图

```text
config/search_space
  -> SearchSpaceConfig.from_dict()
     E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:300
  -> ArchitectureSpec / StageSpec / BlockSpec
     E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:343
     E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:367
     E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:403
        |
        +-> models.builder.build_model()
        |   BlockSpec -> PyTorch block
        |   E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:542
        |
        +-> SearchSpace.resolve_blocks()
            ArchitectureSpec -> ResolvedBlockSpec
            E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:1066
              -> FPGACostEstimator.estimate()
                 stem + pool + blocks + head
                 E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:138
                 E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:142
                   -> CostEstimate.to_candidate_metrics()
                      E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:82
                        -> SearchCandidate
                           E:\1\hwnas\hwnas\src\hwnas_fpga\interfaces.py:48
                             -> ExperimentTracker artifacts
                                candidates.jsonl / candidates.csv / best_candidate.json
                                E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:190
                                E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:302
```

## 字段一致性矩阵

| 字段 | search_space | models | hardware | search candidate / artifact |
|---|---|---|---|---|
| `op` | `op_choices` 与 `BlockSpec.op`，并在 `available_ops` 过滤非法 `skip`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:157`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:926` | `build_block()` 映射各 op：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:542` | `_block_complexity()` 与 `_block_to_op_spec()` 映射：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:647`, `E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:971` | `ArchitectureSpec.to_dict()` 写入 encoding：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:414`；artifact 写入：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:344` |
| `kernel_size` | `kernel_choices`、`BlockSpec.kernel_size`：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:155`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:345` | 多数 op 使用；`mixconv` 固定 `(3,5,7)`：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:587` | 多数 op 使用；`mixconv` 也固定 `(3,5,7)`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:786` | LUT key 仍包含 `kernel_size`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:993` |
| `expand_ratio` | `mbconv/fused_mbconv` 可变，其他 op 被归一为 1：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:845` | `mbconv/fused_mbconv` 使用：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:562` | `mbconv/fused_mbconv` 代价使用：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:700` | `CandidateMetrics` 不记录 expand，本体在 encoding 中：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:347` |
| `stride/output_resolution` | 第一块继承 stage stride，`resolve_blocks()` 计算 resolution：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:823`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:1081` | 除 `skip` 外传入 block；合法采样会过滤 stride/channel 不匹配的 `skip`：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:548`, `E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:933` | 代价按 `ResolvedBlockSpec.output_resolution` 计算：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:425` | encoding 保存 block stride：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:349` |
| `latency/LUT/DSP/BRAM/power` | 约束进入 `SearchConstraints`：`E:\1\hwnas\hwnas\src\hwnas_fpga\interfaces.py:18` | 模型不直接记录硬件指标 | `CostEstimate` 同时有 peak 与 total；候选指标写 total：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:48`, `E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:82` | CSV 写 `latency_ms/lut/bram/dsp/power_w`：`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:302` |
| `num_classes/head` | `ArchitectureSpec.num_classes` 可为空：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:411` | `build_model()` 使用外部参数 `num_classes`：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:706` | cost 若 `architecture.num_classes` 为空则不估 head：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:325` | 当前 configs 定义 `num_classes: 8`，主配置不触发：`E:\1\hwnas\hwnas\configs\search\formal_lut_compact_4stage_nksid_main_av7k325.yaml:12` |

## op 语义对照表

| op | 模型实现 | 代价实现 | LUT 映射 | 是否一致 |
|---|---|---|---|---|
| `conv` | `ConvBlock` 使用 kernel/stride：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:546` | `_block_complexity` 直接卷积 MAC/DSP：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:648` | `op="conv"`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:978` | 一致 |
| `dw_pw_conv` | depthwise + pointwise：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:554` | DW + PW 参数/MAC：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:668` | `op="dw_pw_conv"`，groups=in_channels：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:979`, `E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:990` | 一致 |
| `mbconv` | expand + DW + project：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:562` | expand/DW/project：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:700` | `mbconv_e{expand}_k{kernel}`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:980` | 一致 |
| `fused_mbconv` | fused conv + project：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:571` | fused + project：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:750` | `op="fused_mbconv"`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:981` | 分析模型一致；当前 formal 主配置未启用 |
| `skip` | 仅 channel mismatch 时 1x1 conv，无 stride 参数：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:217` | 0 MAC，固定小 LUT/1 cycle：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:525` | `op="skip"`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:982` | 正常采样一致，因为非法位置过滤：`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:933` |
| `denoise` | DW + smooth + PW：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:327` | 按 DW + PW 估算：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:829` | `op="denoise"`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:984` | 基本一致 |
| `edge` | 4 个方向 edge conv + fusion：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:402` | 4 方向 edge + fusion：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:863` | `op="edge"`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:985` | 基本一致 |
| `mixconv` | 固定 `(3,5,7)`：`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:586` | 固定 `(3,5,7)`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:786` | LUT key 仍使用 `block.kernel_size`：`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:993` | 不一致 |

## 发现列表

### P1

现象：Proxyless 训练期硬件指标不是 `FPGACostEstimator.estimate()` 的同一口径。
证据：`ProxylessSuperNet.expected_hardware_metrics()` 只返回 stage block 的 `latency_ms/dsp/bram/lut/energy_proxy_mj`，`E:\1\hwnas\hwnas\src\hwnas_fpga\models\proxyless.py:739`；完整 estimator 包含 stem/pool/block/head，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:138`；Proxyless loss 直接用 expected 指标，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:178`。
影响：Proxyless 搜索阶段的硬件惩罚与最终候选落盘指标可能排序不同。
具体修正动作：在 `E:\1\hwnas\hwnas\src\hwnas_fpga\models\proxyless.py::ProxylessSuperNet.expected_hardware_metrics` 加入 stem/post-stem/head 常量代价，或将返回字段改名为 `proxyless_expected_*`，并同步 `E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py::_compute_hardware_penalty`。

### P1

现象：`mixconv` 的 `kernel_size` 在 search/LUT key 中有区分，但模型和分析代价固定使用 `(3,5,7)`。
证据：candidate specs 会为非 mbconv op 遍历 `kernel_choices`，`E:\1\hwnas\hwnas\src\hwnas_fpga\models\proxyless.py:49`；模型固定 `kernel_sizes=(3,5,7)`，`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:587`；代价固定 `(3,5,7)`，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:791`；LUT key 写入 `block.kernel_size`，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:993`。
影响：两个 `mixconv` 候选可能模型相同、分析代价相同，但查到不同 LUT 条目。
具体修正动作：在 `models\builder.py::build_block` 和 `hardware\cost.py::_block_complexity` 统一 `mixconv` 的 kernel 语义；若固定 `(3,5,7)`，则在 `search_space\space.py::_build_block` 和 `hardware\cost.py::_block_to_op_spec` 将 `mixconv.kernel_size` 固定为同一 sentinel。

### P2

现象：`num_classes/head` 有两个来源。
证据：`runtime.create_search_space()` 把启动时 `num_classes` 写入 search space，`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:123`；数据加载后才解析默认类别数，`E:\1\hwnas\hwnas\src\hwnas_fpga\runtime.py:377`；模型用调用参数 `num_classes`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\searcher.py:221`；cost 若 `architecture.num_classes` 为空会跳过 head，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:325`。
影响：当前主配置有 `num_classes: 8`，但无类别数配置时，模型 head 与 cost head 可能不一致。
具体修正动作：在 `run_search.py` 数据加载得到 `resolved_num_classes` 后，对 `search_space.config.num_classes` 做一致性断言；同时让 `models\builder.py::build_model` 在 `architecture.num_classes` 非空时断言等于调用参数。

### P2

现象：`FPGACostEstimator` 与 `BackboneCostEstimator` 是两套并行资源口径。
证据：NAS estimator 同时保存 peak/total，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:55`；候选指标写 total，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:82`；Backbone 只有 peak 字段，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\backbone_cost.py:47`；Backbone 约束用 peak，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\backbone_cost.py:188`；且 hardware spec 检查未包含 `max_dsp`，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\backbone_cost.py:434`。
影响：NAS 候选与 backbone baseline 的 DSP/BRAM/LUT 数值不可直接横向比较。
具体修正动作：在 `hardware\backbone_cost.py::BackboneCostEstimate` 增加 `total_dsp/total_bram/total_lut`，并在 `_check_constraints()` 增加 `hardware_spec.max_dsp` 检查。

### P2

现象：LUT 条目有 `power_w/energy_mj`，但 estimator 命中 LUT 时只消费 latency/DSP/BRAM/LUT。
证据：`LutEntry` 定义 `power_w/energy_mj`，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\lookup_table.py:193`；LUT hit 只写入 `lut_entry.dsp/bram/lut/cycles`，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:445`；全局 power 用 peak resource 公式，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:170`。
影响：`CandidateMetrics.power_w/energy_mj` 不是 formal LUT 的实测功耗/能耗。
具体修正动作：在 `hardware\cost.py::LayerCost` 增加可选 `power_w/energy_mj`，LUT hit 时带入 `LutEntry`，并在 `FPGACostEstimator.estimate()` 明确选择“实测聚合”或“分析公式”。

### P2

现象：`CandidateMetrics.accuracy` 在 Proxyless 路径被写成 selection score，而不是稳定的 top1/accuracy 语义。
证据：接口同时定义 `accuracy/macro_f1/top1`，`E:\1\hwnas\hwnas\src\hwnas_fpga\interfaces.py:31`；Proxyless 写 `accuracy=float(selection_score)`，`E:\1\hwnas\hwnas\src\hwnas_fpga\search\proxyless_searcher.py:431`；artifact CSV 单独输出 `accuracy/macro_f1/top1`，`E:\1\hwnas\hwnas\src\hwnas_fpga\experiment.py:306`。
影响：结果消费脚本若把 `accuracy` 当 top1，会误读 macro_f1 或其他 selection metric。
具体修正动作：在 `interfaces.py::CandidateMetrics` 增加 `selection_score/selection_metric`，并把 `proxyless_searcher.py` 的 `accuracy=float(selection_score)` 改为真实 accuracy 或 `None`。

### P3

现象：`SearchSpaceConfig` 明确允许未知 op，但模型和 cost 仍是固定分支。
证据：op_choices 限制被注释，`E:\1\hwnas\hwnas\src\hwnas_fpga\search_space\space.py:276`；模型未知 op 抛错，`E:\1\hwnas\hwnas\src\hwnas_fpga\models\builder.py:612`；cost 未知 op 抛错，`E:\1\hwnas\hwnas\src\hwnas_fpga\hardware\cost.py:896`。
影响：配置层可接受的 op 不等于可构建/可估计 op，错误会延迟到运行期。
具体修正动作：在 `search_space\space.py::SearchSpaceConfig.__post_init__` 增加 `supported_ops` 校验，或引入显式 extension registry 并让 `builder.py`、`cost.py` 共用。

## 需要动态验证的最小测试清单

| 命令 | 目标 |
|---|---|
| `python -m pytest tests/test_search_space.py::SearchSpaceTests::test_roundtrip_architecture_serialization tests/test_search_space.py::SearchSpaceTests::test_illegal_skip_is_rejected -q` | 验证 `ArchitectureSpec` roundtrip 与非法 `skip` 过滤 |
| `python -m pytest tests/test_hardware.py::HardwareEstimatorTests::test_estimate_returns_positive_metrics tests/test_hardware.py::HardwareEstimatorTests::test_strict_formal_lut_marks_missing_queries_infeasible -q` | 验证 total/peak resource 与 strict formal LUT missing 行为 |
| `python -m pytest tests/test_hardware.py::SonarOpsCostTests::test_mixconv_has_higher_cost_than_dw_pw -q` | 验证 sonar op 代价 smoke；仍需补充 `mixconv kernel_size` 等价性测试 |
| `python -m pytest tests/test_proxyless_supernet.py::ProxylessSuperNetTests::test_expected_hardware_metrics_scale_with_larger_stage_choices -q` | 验证 Proxyless expected metrics 当前只做规模单调性检查 |
| `python -m pytest tests/test_backbone_cost.py::BackboneCostTests::test_estimate_backbone_cost -q` | 验证 backbone cost 基线；需补充 total/peak 口径断言 |
