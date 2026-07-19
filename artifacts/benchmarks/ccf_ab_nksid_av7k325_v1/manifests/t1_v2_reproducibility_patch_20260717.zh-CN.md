# T1 v2 可复现性审计补丁（中文伴随档案，2026-07-17）

- 英文原件：`t1_v2_reproducibility_patch_20260717.md`；SHA256：`fd6824a287fe28ef7c8f4fe9fca2139fb3ef799b761b36e3b1bbf2d319e52cf6`。

## 结果

Staged T1 replacement 已按 `paper_id` 联结 paper registry/source audit、source smoke、integration smoke 或受限方法合同，以及六张独立环境卡。包含 5 篇主论文和补充 PLUD，26 列，显式本地统一协议状态；当前所有 non-eligible 行均有非空 blocker。

该替换只存在于 runtime artifact，不修改 canonical Python builder 或当前 G1 source freeze。Canonical promotion 必须在新的 source freeze 下进行。

## 已验证 fail-closed 规则

- source/environment/smoke PASS 不会自动变成正式数值 eligible；
- 每个 `formal_numeric_eligible=False` 行至少有一个显式 blocker；
- HW-PR-NAS 保持“不完整官方发布”，其本地 smoke 标为 paper-spec/non-claimable；
- SURE、DMCL、PLUD 在 redistribution license 未核验时继续隔离；
- HARP 仍受项目完整网络 LLVM graph 与 grouped proxy evaluation 阻断；
- ESDA 仍为 C 类并标记 `no_cross_platform_ranking`；
- CSV、Markdown、LaTeX 来自相同有序 6×26 dataset。

## 独立审计

`t1_v2_independent_audit_20260717.json.txt`：status PASS；6 行、26 列；5 篇主论文、1 篇补充论文；正式数值 eligible 0；显式 blocker 6；输入/环境绑定 16；输出绑定 3；跨格式一致性 PASS；errors `[]`。

## Artifact SHA256

| Artifact | SHA256 |
|---|---|
| `runtime/build_t1_reproducibility_v2.txt` | `d66ea11314e09c2bc3d0ac46d1130d3ff781381faf7c1915d642c81c1f1d46d4` |
| `runtime/audit_t1_reproducibility_v2.txt` | `9f2ef5b2b2c85891eed86b59c5e56f81b7fc6ceec26a9d563caf87e79b75950f` |
| `tables/t1_v2.csv` | `5c9c3e7935e137410113b1fabd40148e37cb9b9a6827941bca2cc5aec78d10d4` |
| `tables/t1_v2.md` | `26a49ebf7393a31884dc182abbf19a27cec72a314ae55046bcc971c4bdf4979c` |
| `tables/t1_v2.tex` | `e0f9c4b2f26ff3135944d98c8297a9cd569698b7bba5afec10b7988e995132be` |
| `tables/t1_v2_meta.json.txt` | `8aadc5038540003968e6da4bdc369da6742e9479fedd0563a554c2ff8d31bdfd` |
| `t1_v2_independent_audit_20260717.json.txt` | `cfa037e5af40c9006dcf50a020d07e276856ec69098f35f9840b700e0ecac8fa` |

## 证据边界

T1 v2 证明来源、许可证、环境与可复现性记账，不证明模型优越性或论文结果复现。各方向完成本地统一协议前禁止数值比较；canonical builder 更新、新 source freeze 与 readiness 重建前，禁止把 staged artifact 提升到正式 `t1.*`。
