# T6 结构候选池冻结（中文伴随档案，2026-07-17）

- 英文原件：`t6_structural_candidate_pool_patch_20260717.md`；SHA256：`0af3b72c93d652571402c130cba2cfe565f156d293caff72766b421484efb345`。

## 决定

排除暂停的声呐算子后，早期 v3/v4 deployment pool 只有 4 个 semantic-safe operator signature，不能支持预声明的五架构族 grouped comparison。较广的 200 行 `mobile_anchor` probe 含 200 个唯一 encoding，仅使用 `dw_pw_conv`、`mbconv`、`fused_mbconv` 与合法 `skip`，因此只作为结构 DOE（实验设计）来源。

## 冻结池

- 100 个唯一 encoding，5 个架构族，每族恰好 20 个。
- `skip_heavy`：skip block 占全部 block 至少 25%。
- 三个 dominant family：一种非 skip 算子具有严格多数；纯网络映射到对应 dominant family。
- `mixed_balanced`：没有任何非 skip 算子具有严格多数。
- Grouped CV：5 个 leave-one-family-out fold，每个 fold 80 个训练候选、20 个 held-out 候选。
- 采样平衡：每族最多保留 5 个历史解析 infeasible 行，再按 salted encoding-hash 顺序用 feasible 行补齐。历史 flag 只作采样 stratum，不是 HLS/route 真值。

## 绑定

- Builder：`f7016edd07c3b9ccdc20eb81496e6d4f99758e1bfda3c47f17570b773da61f32`。
- Auditor：`2da583d12e6313445a2f0c6c6df9ab3c8ee06ecc3741f84b616b166d9b562d44`。
- Manifest：`046e1ecba4e73306beb46e381070b5c251ef9d59b08b8045c635b1a905d48f65`。
- 独立审计：`f93b129a6249e6e4c75496b26dbeccf8b10cf41ea6605624029674cbe93ec43d`；200 个来源行/唯一 encoding、100 个选中 encoding、5 个平衡 fold、0 错误。

## 边界与执行 Gate

状态为 `STRUCTURAL_POOL_FROZEN_TRUTH_NOT_COLLECTED`。不验证旧搜索 accuracy、解析资源、HLS performance、route feasibility 或 HARP prediction；T6 真值仍为 0/100，F4/T6 PENDING。

先为每族导出 1 个 source-linked complete-network HLS top 并要求 semantic equivalence，再允许 csynth/route。任何架构族无法忠实表示时必须停止并修生成器，禁止用其他族重复 encoding 替代。五族 pilot 全通过后才可排队剩余 95 个，并保留每个 synthesis/route failure 的阶段与类别。
