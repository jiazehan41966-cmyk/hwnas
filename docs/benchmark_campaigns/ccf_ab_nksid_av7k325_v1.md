# CCF A/B 论文对标实验：ccf_ab_nksid_av7k325_v1

## 证据状态

本 campaign 已完成论文源代码固定、许可证审计、统一适配器接口、闭集/开放集逐样本预测 schema、精确三目标 HV、声呐 SNR 合成与图像质量边界、配对统计方法及归档生成器。当前结果不构成论文数值结论。

机器可读正式 readiness 当前为 `2/9 PASS`：五篇主论文源审计和六个 paper-specific CUDA 适配环境已闭合；分类、NAS、HLS、AV7K325、功耗、Gate 和正式图表仍未闭合。readiness 文件为 `artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/manifests/formal_readiness.json`。

正式分类运行还必须绑定 `scripts/freeze_experiment_source.py` 生成的 source freeze（源码冻结）清单。统一入口会在训练前复验当前源码、清单 SHA256 和保留的 `source_snapshot.zip`；缺少或变更任一项时，即使 15 个 fold-seed 单元全部完成，协议和 readiness 也不得标记为可声明。外层 G1 启动器和每个 `run_eval_protocol.py` 子进程会分别失败闭合。

测量优先 Gate 状态为：G1 `PASS`（45/45）、G2 `PENDING`、G3 `FROZEN`、G4 `PENDING`、功耗 `NOT_MEASURED`、G5 `PAUSED`。G1 通过只闭合三个冻结分类基线，不会自动使 NAS、HLS/route、AV7K325、功耗或正式图表可声明；这些层仍须保留各自的 Gate 状态。

## 冻结主对标

| 方向 | paper_id | 角色 | 比较类别 | 当前代码状态 |
|---|---|---|---|---|
| NAS/Pareto 代理排序 | `hw_pr_nas_2023` | 主对标 | B | 固定 commit，MIT；官方仓库缺少 README 所列核心 predictor/base_surrogate 文件且入口不可执行。已完成 10 个项目 evaluator 记录上的 paper-spec ListMLE 数据流 smoke，但它不是作者实现，禁止进入 T5 |
| 分类鲁棒性与校准 | `sure_2024` | 主对标 | B | 固定 commit，无明确许可证，隔离保存；作者 FMFP+CRL+RegMixup+cosine 组件的方法级 smoke 已通过，禁止再分发 |
| HLS/route 代理 | `harp_2023` | 主对标 | B | 固定 commit，BSD-3-Clause；作者 AES 分层 LLVM 程序图契约 smoke 已通过。HARP 输入是 HLS C/C++ 的 LLVM 程序图而非 NAS 网络图，项目候选图仍待生成 |
| FPGA/量化/板测链路 | `esda_2024` | 主对标 | C | 固定 commit，MIT；作者训练—量化—HLS/Vivado—bitstream—PYNQ 板测—功耗归档契约 smoke 已通过。作者材料属于 ZCU102，禁止与 AV7K325 跨板数值排名，也不填入 T7/T8 |
| NKSID 开放集长尾 | `dmcl_sonar_oltr_2025` | 主对标 | A | 固定 archive SHA256，无许可证；隔离解包后已动态加载作者 `DynamicMarginLoss`，1 fold × 1 seed × 1 epoch 方法级 smoke 已通过；README 对应 PLUD 而非 DMCL，论文—代码对应关系仅为 partial |
| NKSID 开放集长尾 | `plud_sonar_oltr_2024` | 补充方法 | A | 作者包 README 与 `plud.py` 对应关系明确；已动态加载作者 `push_logit_loss` 并完成统一协议 1×1×1 smoke。无明确许可证，只允许隔离本地执行 |

## 统一执行入口

闭集与开放集均从 `run_eval_protocol.py` 进入。外部作者脚本不得绕过该入口生成可声明分类数字。

```powershell
# 闭集单 fold/seed smoke；只用于接口验证
.venv_cuda\Scripts\python.exe run_eval_protocol.py --task closed_set --folds 0 --seeds 42 --epochs 1 --run-name smoke_closed

# 开放集 CE+MSP 单 fold/seed smoke；只用于接口验证
.venv_cuda\Scripts\python.exe run_eval_protocol.py --task open_long_tail --folds 0 --seeds 42 --epochs 1 --run-name smoke_open_msp

# SURE 作者配方方法级 smoke；从隔离 checkout 动态加载作者组件并绑定环境卡
.venv_benchmarks\sure_2024_cuda\Scripts\python.exe run_eval_protocol.py --task closed_set --adapter-id sure_author_recipe --arch simplecnn --folds 0 --seeds 42 --epochs 1 --image-size 64 --campaign-id ccf_ab_nksid_av7k325_v1 --run-name sure_dedicated_env_1x1x1

# DMCL 作者损失方法级 smoke；只允许 open_long_tail 任务
.venv_benchmarks\dmcl_sonar_oltr_2025_cuda\Scripts\python.exe run_eval_protocol.py --task open_long_tail --adapter-id dmcl_author_loss --arch simplecnn --folds 0 --seeds 42 --epochs 1 --image-size 64 --campaign-id ccf_ab_nksid_av7k325_v1 --run-name dmcl_dedicated_env_1x1x1

# PLUD 作者 push-logit 方法级 smoke；同样只允许 open_long_tail 任务
.venv_benchmarks\plud_sonar_oltr_2024_cuda\Scripts\python.exe run_eval_protocol.py --task open_long_tail --adapter-id plud_author_loss --arch simplecnn --folds 0 --seeds 42 --epochs 1 --image-size 64 --campaign-id ccf_ab_nksid_av7k325_v1 --run-name plud_dedicated_env_1x1x1

# 完成代码与回归测试后冻结源码；冻结后不得继续改动被纳入清单的文件
.venv_cuda\Scripts\python.exe scripts/freeze_experiment_source.py freeze --output-dir artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/source_freeze/g1_20260715_v2

# G1 正式运行必须使用同一冻结清单、5 folds × 3 seeds，并等待对应 Gate
powershell -ExecutionPolicy Bypass -File scripts/run_g1_clean_after_patch.ps1 -SourceFreezeManifest artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/source_freeze/g1_20260715_v2/source_freeze_manifest.json
```

NAS/HLS 的非声明性契约 smoke 不经过分类入口，但必须读取项目产生的真实候选日志并保留边界：

```powershell
# 10 个 evaluator 调用的结构 smoke 候选池
.venv_cuda\Scripts\python.exe run_search.py --config configs/search/aging_vs_rl_dummy_smoke.yaml --search-method random --num-candidates 10 --train-epochs 1 --device cuda --output-dir results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke --run-name hwpr_random_evaluator_pool_10

# 仅复刻论文 equations 7--8 的 listwise Pareto ranking 行为；不是作者缺失的 GCN+LSTM 实现
.venv_cuda\Scripts\python.exe scripts/run_hwpr_paper_spec_smoke.py --candidates-jsonl results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/hwpr_random_evaluator_pool_10/results/candidates.jsonl --output results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/hwpr_paper_spec_10/paper_spec_smoke.json --limit 10

# 只验证作者 HLS C 源码到 LLVM 分层 GEXF 的输入契约
.venv_cuda\Scripts\python.exe scripts/run_harp_graph_contract_smoke.py

# 只核验 ESDA 作者 ZCU102 分层证据材料；不执行跨板比较
.venv_cuda\Scripts\python.exe scripts/run_esda_artifact_contract_smoke.py
```

逐样本输出至少记录 `campaign_id, paper_id, method, fold, seed, sample_id, target, prediction, confidence, checkpoint_sha, config_sha, data_sha, split_sha, code_commit, code_state_sha, claimability_status`，同时保存 logits。开放集 MSP 阈值只使用 inner-known 数据校准，外层 unknown 不参与阈值选择。

## 搜索比较与精确 HV

```powershell
python scripts/compare_search_methods.py `
  --run <random_run> --run <rl_run> --run <aging_run> --run <hwpr_run> `
  --latency-limit-ms <frozen_limit> `
  --output-dir results/benchmarks/ccf_ab_nksid_av7k325_v1/search_comparison
```

正式 HV 定义为可行候选上的 `[1-f_clean, 1-f_robust, latency/latency_limit]`，参考点 `(1,1,1)`。未显式给出 `--latency-limit-ms` 时，精确 HV 标为不可用；旧 `reported_hypervolume` 仅保留为 legacy 字段。

## 统计与归档

统计单位为配对的 fold-seed 单元。正式分析使用 10,000 次分层配对 bootstrap 95% CI、配对置换检验、Holm 校正和 Cohen's dz，并保留原始配对分布。

```powershell
python scripts/audit_benchmark_sources.py
python scripts/run_benchmark_source_smoke.py
python scripts/capture_benchmark_environments.py
python scripts/verify_benchmark_environments.py
python scripts/audit_benchmark_readiness.py
python scripts/build_benchmark_artifacts.py --campaign-id ccf_ab_nksid_av7k325_v1
```

归档生成器不会为缺数据项目创建空表或空图。`artifact_status.json` 中只有 CSV/Markdown/LaTeX 三种表格格式或 PNG/PDF/source CSV/meta JSON 四种图片格式全部存在时才标记 `AVAILABLE`；这不等价于科学 Gate 已通过。

## 尚未完成及正式启动条件

- SURE、DMCL 与补充方法 PLUD 已在各自 CUDA 适配环境中完成 1 fold × 1 seed × 1 epoch 方法级集成 smoke，运行 manifest 会校验环境卡、解释器目录和 lock SHA；数值仍不可用于 T2/T3。HW-PR-NAS 完成的是 paper-spec 非声明性数据流 smoke；因作者仓库不完整，尚无可称为作者方法的运行。HARP 完成的是作者数据图契约 smoke，尚无项目候选预测。ESDA 完成的是作者 ZCU102 证据链材料契约 smoke，不构成 AV7K325 板测或功耗结果。
- 五篇主论文加一篇补充 PLUD 的六个 paper-specific 统一协议适配环境均为 `READY_DEDICATED_ENVIRONMENT`，固定为 Python 3.13.3、PyTorch 2.7.0+cu126，并有独立路径、probe 和 freeze。它们不是作者旧版 Python/PyTorch 环境的逐字节复刻；作者依赖声明仍在环境卡中单独留档。
- Sonar-OLTR 作者包缺少明确许可证；RAR 已在忽略跟踪的隔离目录完成可审计解包，可本地动态加载，但不得再分发。包内 README 对应 2024 PLUD，不能据此声称完整复现 2025 DMCL。
- HW-PR-NAS 官方仓库的源文件可编译，但 README 所列 `base_surrogate.py` 及 accuracy/latency/energy predictor 均缺失，`valid_loss()` 未定义且 `test.py` 调用签名不匹配。正式 T5 必须等待可验证作者实现，或经书面协议把独立复刻改列为单独方法。
- HARP 项目适配必须从实际候选导出的 HLS C/C++ 经 LLVM 13 生成程序图；禁止把 NAS architecture JSON 直接改名为图输入。
- HARP 正式比较需要至少 100 个语义安全、完整网络的 HLS/route 样本；少于 30 仅描述，30–99 仅 exploratory。
- ESDA checkout 含 8 个作者 ZCU102 bitstream 和 8 份归档功率数组；它们仅用于学习证据组织。任何跨平台数值排名会被适配层拒绝，项目功耗仍为 `NOT_MEASURED`。
- AV7K325 三类候选的同工具链 route、COM5、外部功率仪器时间序列尚未产生，功耗继续为 `NOT_MEASURED`。
- 所有正式结论以 `audit_measurement_first_gates.py` 复审结果为准，搜索代理、retrain、HLS/route、COM5、图像质量和功耗必须分层陈述。
