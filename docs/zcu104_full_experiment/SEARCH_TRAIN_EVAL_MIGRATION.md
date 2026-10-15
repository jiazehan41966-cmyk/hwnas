# ZCU104 CFEM20 搜索—训练—Protocol V2 迁移说明

## 状态

ARCHIVED / RETIRED：本文件记录旧 `fourstage_cfem20_v1` RAW20 迁移方案。
旧 9 行 CFEM20 operator-cost adapter 已从当前运行时删除，不再作为本轮
NKSID RL-HW-NAS / ZCU104 formal LUT 搜索入口。本轮当前 formal LUT 是
`mobilenetA_full_dmdc_cfem_v2_recovered` 的 88 行 ZCU104 HLS/route
operator-cost 表。

## 复用链路与新增边界

原通用链路为：`run_search.py` 读取 YAML，`runtime.build_cost_estimator()`
加载旧通用 LUT，`FPGACostEstimator` 对逐层代价求和；搜索器把
`ArchitectureSpec` 写入 `best_candidate.json`；`run_retrain.py` 从该 JSON
重建模型并写 `final_best_model.pt`；`run_eval_protocol.py --candidate-path`
在 Protocol V2 下重新训练并做 outer 评估。

本次继续复用 SearchSpace、Random/RL/Aging 搜索器、`build_model()`、训练器、
ExperimentTracker 和 Protocol V2 数据/指标代码。新增内容仅为：

1. 旧 RAW20/CFEM20 9 行 exact-shape adapter：已退役；
2. `configs/search/zcu104_fourstage_cfem20_v1.yaml`：固定 Stage1/Stage3，
   Stage2 取 4 种、Stage4 取 5 种，空间恰为 20；
3. candidate、retrain checkpoint 与 Protocol V2 记录之间的 SHA256 校验和传播；其中
   retrain 记录实际 inner split 索引及其 SHA256，而不只记录划分参数；
4. Protocol V2 的可选 `--initial-checkpoint` 接口，用于对已重训权重做单一
   fold/seed 的 checkpoint 评估；入口会核对 candidate、架构、训练配方、数据清单、
   精确 split、fold 和 seed。它不会被表述为完整 5×3 正式结果。

## 旧绑定审计

- `runtime.load_lut_query_engine()` 与各历史 `configs/search/*.yaml` 的
  `hardware.lut_path` 仍服务历史 AV7K325/通用逐层 LUT 实验；它们未删除，
  但新 ZCU104 配置不再走该入口。
- `configs/experiments/zcu104_full_campaign_v1.yaml` 仍登记四个旧网络及其
  checkpoint 作为历史档案，字段已明确为 `historical_excluded_*`，禁止作为
  本轮 search input、control、calibration 或 deployment target。
- `run_search.py` 原仓库级 Stage-3 gate 绑定上一轮 Gate0/G2/G4/AV7K325 证据；
  旧 RAW20/CFEM20 专用门已随 9 行 adapter 退役。历史搜索配置只能作为档案，
  不作为当前 claimable ZCU104 搜索入口。
- 旧 `run_search.py -> best_candidate.json -> run_retrain.py` 文件格式继续兼容；
  新写出的 candidate 增加 `architecture_sha256` 和 search config SHA256。
- `run_eval_protocol.py --candidate-path` 原本总是从头训练；新增
  `--initial-checkpoint` 后可严格绑定已重训权重，同时核对 candidate、架构、
  fold、seed 和 checkpoint SHA256。默认正式 Protocol V2 从头独立重训行为不变。

## 20 个网络到 9 个代价项

每个网络只选择两个新实测行：一个 Stage2 exact-shape 行和一个 Stage4
exact-shape 行。4 个 Stage2 键与 5 个 Stage4 键的笛卡尔积给出 20 个网络，
全体网络共同引用 9 个唯一键。

本轮搜索硬件代价的 aggregation（累加规则）为：

- `latency_board_ms`、`energy_dynamic_mj_per_op`、LUT、DSP、BRAM36：两行求和；
- `p_dynamic_w`：两行最大值，仅作搜索代理，不是完整网络功率实测；
- 参数量与 MAC：由实际完整 PyTorch 模型统计；
- 固定 stem/Stage1/Stage3/head 不得从旧 LUT 或解析估算补成本轮 ZCU104
  实测代价，因此上述硬件数值的 scope 明确是 searchable Stage2+Stage4。

## Fail-closed 条件

正式表只有同时满足以下条件才可进入搜索：表和 coverage JSON 的 canonical
payload SHA256 正确；`data_kind=FORMAL_ZCU104_MEASURED`；`search_ready=true`；
9/9 行、20/20 mapping 完整；搜索空间、ZCU104 part、200 MHz、2022.2、精确
输入输出形状、route gate、measurement status 和关键来源 SHA256 全部一致。
此外，20 个 mapping 必须逐个与代码枚举得到的 frozen RAW20 架构 SHA256、架构 ID
及 Stage2/Stage4 代价键完全一致；仅有 20 个格式正确的条目仍不足以放行。

测试表必须声明 `TEST_ONLY_SYNTHETIC`，且配置显式设置
`allow_test_operator_cost_data: true`。正式配置固定为 false，避免测试数值混入实验。

## 启动方式

当前只允许执行 MobileNetA formal LUT preflight：

```powershell
python scripts/audit_zcu104_mobilenetA_formal_lut_preflight.py
python -m pytest tests/test_zcu104_mobilenetA_lut_preflight.py tests/test_zcu104_mobilenetA_recovery.py -q
```

正式 MobileNetA-bound estimator 接入前，不要启动 claimable RL search。旧
RAW20/CFEM20 adapter 已退役；不要再执行：

```powershell
python run_search.py --config configs/search/zcu104_fourstage_cfem20_v1.yaml
```

搜索结果进入重训：

```powershell
python run_retrain.py --config configs/search/zcu104_fourstage_cfem20_v1.yaml `
  --candidate-path <search_run>/checkpoints/best_candidate.json `
  --fold <fold> --seed <seed> --epochs <preregistered_epochs>
```

将该 fold/seed 的重训 checkpoint 交给 Protocol V2（命令中的训练配方参数必须与
checkpoint 记录一致，否则 fail closed）：

```powershell
python run_eval_protocol.py --candidate-path <best_candidate.json> `
  --initial-checkpoint <retrain_run>/checkpoints/final_best_model.pt `
  --folds <fold> --seeds <seed> --epochs <same_epochs> `
  --batch-size 8 --warmup-epochs 5 --min-lr-ratio 0.01 `
  --label-smoothing 0.1 --logit-adjust-tau 1.0 `
  --gradient-accumulation-steps 4 --amp
```

Protocol V2 的正式推荐路径仍是按每个 fold/seed 独立重训并评估：

```powershell
python run_eval_protocol.py --candidate-path <best_candidate.json> `
  --experiment-contract configs/evaluation/nksid_frozen_protocol_v2.yaml `
  --selection-provenance new_nested --folds 0,1,2,3,4 --seeds 42,43,44 `
  --save-checkpoints --run-name <registered_run_name>
```

只有完成预注册的独立搜索/重训单元及全部来源冻结后，才能使用 `new_nested`
作为正式结果；本迁移的 dummy/synthetic smoke 均为非正式接口证据。
