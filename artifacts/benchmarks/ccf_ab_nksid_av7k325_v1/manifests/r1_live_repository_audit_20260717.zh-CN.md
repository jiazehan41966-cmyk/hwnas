# R1 外部仓库与许可证实时审计（中文伴随档案，2026-07-17）

- 英文原件：`r1_live_repository_audit_20260717.md`；SHA256：`90e7873b1264df214863bab55998c315f2d69d1990e1058eab6d61b8fd02a293`。

## 结果与边界

五个唯一外部 checkout 均现场核验，而非只信任归档 JSON。所有 checkout 的 HEAD 与登记 commit、official remote 一致，tracked author source 无 staged/unstaged 修改。这只证明源码版本完整性，不证明论文结果复现。

| Checkout / paper | HEAD | Remote | Tracked 修改 | Untracked 状态 |
|---|---|---|---:|---|
| `reference/HW-PR-NAS` | `296c6576fbae2b277e56c704ff3b6e648ec4c2be` | IHIaadj/HW-PR-NAS | 0 | 3 个 pycache |
| `reference/_local/SURE` | `5ce0193bc93e73b1c7f1f53aeda8854e997011e2` | Intellindust-AI-Lab/SURE | 0 | 7 个 pycache |
| `reference/HARP` | `c8bffd9411917b125846429b4d6be4f21c7a7165` | UCLA-VAST/HARP | 0 | 3 个 pycache |
| `reference/ESDA` | `b75c8c93ca258158c06a6434f5f0f084add02ee5` | CASR-HKU/ESDA | 0 | clean |
| `reference/_local/Sonar-OLTR` | `eea8dc07ce007988150ac208cd09e00daedba2ca` | gmgslinyu/Sonar-OLTR | 0 | 隔离解压，82 个 untracked |

Untracked 是 runtime/extraction 副产物，不是 tracked author code 修改；保持不动。不能把外部 worktree 笼统描述为全 clean，也不能误报成作者源码变化。

## License

- HW-PR-NAS：LICENSE SHA `5f89424986edb716ba4040af41626a8471120e1a625c6a463ea1b271d685fa98`，MIT verified。
- HARP：`2d633a2c625be312afeb6f660bfa12b9dd4ab8b051ee96c04f38ab91abc21912`，BSD-3-Clause verified。
- ESDA：`07fd61b75f13681e7b46355e8f70314f53ffca846fd1f02f7829cd735b2cdded`，MIT verified。
- SURE 与 Sonar-OLTR：无 tracked license，redistribution unverified，只能隔离使用。
- Sonar archive：6,359,379 bytes，SHA `4bd5158c491821bb1de3138856344949ca3ce1747f033601809d30774e7d5a61`；`reference/_local/` 被顶层 `.gitignore` 忽略，不由项目仓库再分发。

## Superproject integration 缺口

`.gitmodules` 声明 HW-PR-NAS/HARP/ESDA，但顶层 index 尚无对应 gitlink；三个目录在 superproject 仍 untracked，`git submodule status` 无法报告。准确表述是：仓库已下载、固定版本、remote 验证、tracked source clean，可用于审计与本地执行；但 submodule integration 尚未完成。

后续需在新的 source freeze 下登记 gitlink、验证 `git submodule status`、重跑 R1/T1。不得称为“submodules fully integrated”。

## 方法边界

HW-PR-NAS 官方代码不完整；SURE/DMCL/PLUD 许可证未核验；DMCL paper-code correspondence partial，PLUD verified；HARP/ESDA 源码完整不替代本地 HLS、route、board、power 测量。
