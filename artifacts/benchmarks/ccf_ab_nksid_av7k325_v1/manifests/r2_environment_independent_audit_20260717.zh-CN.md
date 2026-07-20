# R2 独立环境审计（中文伴随档案，2026-07-17）

- 英文原件：`r2_environment_independent_audit_20260717.md`；SHA256：`6f998277d2c3de7c7d5123d1147002b678dbce009d4192c4d2e59d9462681f7f`。

## 结果与检查

六个已登记主/补充论文 runtime 的文件系统与 manifest 一致性审计均 PASS。活动 G1 时没有重新执行 CUDA probe；只复核归档 probe、path、hash 与 count，避免竞争 GPU。

每张环境卡均检查：在 `environment/index.json` 登记；`pinned_commit==observed_commit`；state 为 `READY_DEDICATED_ENVIRONMENT`；归档 probe PASS 且 observed commit 正确；独立 interpreter 与 lock file 存在；重算 lock SHA 与 package count 一致；verification fingerprint 是 64 字符 SHA。六张卡全部通过，并有六个不同 venv path。

| Paper ID | Probe | Packages | Lock SHA256 | Verification fingerprint |
|---|---:|---:|---|---|
| `hw_pr_nas_2023` | PASS | 16 | `e136caaa1776bef95b235119d73d883c69caebbf34a2ddda1befe3b17249be2a` | `bc048cbf8435b5ba294ef664663e28c9034705b40b5152576dc85b1b51615030` |
| `sure_2024` | PASS | 16 | 同上 | `4b24934310c5b58734725a500c242256d53193364b8ce325c10dfbc80f695e99` |
| `harp_2023` | PASS | 16 | 同上 | `020a7a423c7055820c9f4955c4d00384e810ee5b946f2d6a4f60f7f9757665dc` |
| `esda_2024` | PASS | 16 | 同上 | `84e116545379c9827f7f27a781457dd515212d728371ffae84217885b5779813` |
| `dmcl_sonar_oltr_2025` | PASS | 16 | 同上 | `a1fb5acc73d7a4c2ef16700dd3887a1ebecddfc738bcd6cdd8bb0e9778d571cf` |
| `plud_sonar_oltr_2024` | PASS | 16 | 同上 | `b92827b87b0e59adfcb8864855e2a88769fedc33042bf1b0c3f7d69089b3a74b` |

相同 lock SHA 是预期行为：六个 adapter 使用同一冻结 minimal CUDA dependency set，但路径与 method-specific probe fingerprint 不同。

## 边界

R2 只证明 runtime 存在、隔离、固定 commit 且依赖锁定；不证明 author-environment equivalence、作者表复现、许可证清除或本地结果正式 eligible。HW-PR-NAS 仍是 paper-spec adapter；SURE/DMCL/PLUD 因再分发许可证未核验继续隔离。
