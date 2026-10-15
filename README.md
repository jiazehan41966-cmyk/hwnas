# Sonar HW-NAS

当前主线为 **E0/E1 → elastic_neck72_v1** 声呐检测：固定 B1 八阶段骨干、640×640 灰度输入、P2/P3/P4/P5、48 通道融合与直接 1×1 检测头。搜索 8 个 E0 与 64 个 E1 结构；3 种量化 × 6 种映射，共 1296 个联合决策。

这是独立的源码迁移分支，保留原网络、语义初始化、训练损失和验证协议。历史数据与研究结果仍在原工作区。本分支没有因为重构而获得新的精度、硬件可行性或部署结论。

```text
configs/   OmegaConf 实验配置及原始冻结协议
space/     Genotype、严格校验、JSON、SHA256、编码/解码
models/    Genotype → 独立静态检测网络
train/     原损失、YOLO 数据接口、训练/恢复与冻结 valid 评估
quant/     Q(x) 纯规划；executor/runtime 单独执行 PTQ/整数参考
hw/        分层证据导入、报告解析与上下文 SQLite 缓存
search/    原 Pareto 晋升、40/60/80 调度、弃选抽查与三种子确认
results/   每次运行及每候选不可变记录；不纳入源码提交
scripts/   search / retrain / quantize / deploy / collect / tables
```

依赖由上层消费下层；`scripts/` 负责组合执行器。`space` 不依赖 PyTorch，`models` 不依赖训练或搜索，搜索中没有模型构建代码。静态测试守住这些边界。

每个候选唯一交换格式为 `Genotype`。新 `arch_hash` 同时覆盖空间版本、模型合同、输入和类别数；历史 choice-only hash 保留为 `legacy_arch_hash`，仅用于溯源。种子和训练预算不改变架构身份，分别写入运行及工件记录。

硬件缓存键为 `(arch_hash, q_hash, context_hash)`。第三项覆盖映射、器件、时钟、工具版本、证据范围及适用工件摘要；板测还绑定 checkpoint、量化工件和 bitstream。命中时重新核验原文件，缺测返回 `NEEDS_MEASUREMENT`。接口合同见 [hw/README.md](hw/README.md)。

在源码目录建立环境并安装：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

先生成无数据访问、无训练的预检：

```powershell
python -m scripts.search --output results/plan_001
python -m scripts.tables --run results/plan_001 --output results/plan_001_tables
```

预检会记录 1296 个缺测决策，精度留空、Pareto 表为空。配置通过 OmegaConf 解析并冻结插值；运行记录包含 resolved YAML、Git commit、源码摘要、Python 与依赖版本。实际执行前必须提交源码和配置，避免把脏工作区归给某个 commit。

需要执行研究任务时，先在新的 YAML 中填写真实 `dataset.root`。单候选阶段入口如下；这些命令会产生实际计算，重构验证没有执行它们：

```powershell
python -m scripts.retrain --config configs/my_experiment.yaml --index 0 --budget 40 --seed 42 --output results/train_001
python -m scripts.quantize --config configs/my_experiment.yaml --training-run results/train_001 --profile W8A8 --output results/ptq_001
python -m scripts.deploy --config configs/my_experiment.yaml --quantized-run results/ptq_001 --output results/bundle_001
```

`--genotype candidate.json` 可代替 `--index`；所有后续记录都以完整 hash 为准。续训在同一训练 run 使用 `--resume --budget 60`，再增加到 80，保留每个预算的 checkpoint 和回执。多回执目录的量化用 `--receipt` 指定实际 JSON。校准固定为前 64 张 train、无增强；选择与量化精度评估只用 valid，test 封闭。当前 QAT 无对应主线执行器，会明确报未实现。

`deploy` 导出整数参数包和证据准入报告。它不会实现尚缺的整网板端执行器，也不会自动烧板。`--context context.json --evidence report.xml` 可导入真实硬件证据；原始 XML 需要身份旁车。未提供测量时继续保留缺测状态。

多个同源码、同配置的 bundle 可用 `scripts.collect --input RUN_A --input RUN_B --output results/collected` 聚合；`scripts.search --records results/collected --stage 0 --output results/selection` 消费真实候选记录。后续阶段需要 `--previous` 指定上一阶段的 hash 列表。`scripts.tables` 从记录导出 table1_accuracy、table2_hardware、table3_pareto，种子与预算分别计算，不补造缺值。

完整原流程由 `search.engine.run_staged` 调度：72→54→36→12，40/60/80 累计训练、两次 E0/E1 弃选抽查、12 架构三种子确认与配对量化准入。CLI 用 `scripts.search --execute` 显式启动；启动前必须配置并通过完整 1296 决策硬件证据索引、索引 SHA 和工具版本，缺任一项即在读取数据前退出。索引是 JSON 列表，每项含 `arch_hash`、`q_hash`、完整 `context` 及相对索引的 `evidence` 路径。默认配置保留原主线缺测状态，不能直接启动完整研究流程。三种子确认不等于三次独立搜索，跨方法比较的原门槛也不会因重构自动通过。

训练器和 COCO 风格评估沿用原实现。当前评估是固定阈值、IoU .50:.05:.95 的 bbox all-area AP，不宣称等同官方 COCO 全指标；没有切换 ultralytics/mmdet，避免迁移同时改变实验协议。

验证命令：

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp results/pytest_new
```

迁移范围、存档与已验证边界见 [docs/MIGRATION.md](docs/MIGRATION.md)。
