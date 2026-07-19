# T6 五架构族导出 Pilot 冻结（中文伴随档案）

- 英文原件：`t6_five_family_pilot_patch_20260717.md`；SHA256：`d79e07d64dffa75048d696bfef2503781dfba7d8939d191855c2b513c3323929`。

## 决定与输入

100 候选 DOE 在进入完整网络 export 前，必须先验证生成器覆盖每个声明架构族，因此每族冻结一个确定性 pilot。该子集只是执行预检，不是性能子样本，也不替代 100 条完整网络 HLS/route 真值。

| 架构族 | Pool sample | 来源候选 | Blocks | Encoding SHA256 |
|---|---|---|---:|---|
| `dw_pw_dominant` | `t6_dw_pw_dominant_04` | `probe_0000` | 4 | `47a1c40c725a6c71066b26cdf1aad23ea65c5ad7177c74d0822648083b9df407` |
| `mbconv_dominant` | `t6_mbconv_dominant_05` | `probe_0060` | 4 | `990070c9cf249f1eaeecf8d5e275e963690aab068707a15bb62207dbfe5e94c4` |
| `fused_mbconv_dominant` | `t6_fused_mbconv_dominant_09` | `probe_0048` | 4 | `f1335b4d5f45eba6b8a87c5e3e7add7a8f0f967c881a87c83de48b70cb63e662` |
| `skip_heavy` | `t6_skip_heavy_00` | `probe_0136` | 5 | `56749442d6ee3eaccf3735dc56fb769d47ca9be2fae8ac2d7395f4eb78febf89` |
| `mixed_balanced` | `t6_mixed_balanced_15` | `probe_0188` | 4 | `4a7ae1e28481b8252847d58f998ab1ded800a07c221fa0cc46b4b0491572e686` |

选择先要求冻结的 source analytic flag 为 true，再最小化 block count，以 salted selection key 破平局；该 flag 只能用于早期生成器覆盖排序，不能报告为 HLS/route feasible。

## Provenance 与测试

- Builder：`cdf82b5074e4ad92f011b938376b9f8a88c3e5aecf7a3a4efb07d5aea2249077`。
- Auditor：`2d52e7b0ff2347e759009ac0ec56aa97f8ad20073871be5fa8942246b5e3f3f6`。
- Manifest：`ae9aad20b0c10f5958a530f2860272d5455543970de19ba91e261b08efa7c17c`。
- Audit：`88010ac17644747aa9c7dc92ec1ddf1d4fd9a24468050a60def607dd88991117`，PASS、五族、0 错误。
- Contract test：`f794783038943f94de0529518501d1926d6b38b67d3c590cbc8a20feec4e5ab5`；合法输入通过，过早要求板卡以及重复 family/encoding 被拒绝。

## 现状与边界

五个 pilot 均为 `COMPLETE_NETWORK_PENDING_EXPORT`，尚未收集 export、semantic equivalence、csynth 或 route，T6 仍 0/100。上述四阶段不需要物理板卡；只有五个 export 全部通过语义等价后才可排队剩余 95 个，bitstream/COM5 只在 route-feasible 选择后开始。

现有 isolated full-network planner 的 plan-only probe 对五个候选均返回 `not_generated_mapping_incomplete`，每个至少缺 6 条 component row，没有 candidate-HLS mapping，arch-84 identity gate 失败。Gap audit SHA `f2b9a28f7311ac5adc355aedc82ca578ae6bdff6f0dd98317005f58d61680d64`；结果 manifest `0d514c94d840a93ce2da57ed632f50092ef949fbc659e78eec06b67f638c8066`，状态 `PASS_EXPECTED_GAP_CONFIRMED`。

这证明五个候选不能复用历史 bitstream，当前 planner 不能直接产生 T6 真值。修复必须在新的 source freeze 下实现 source-linked candidate-specific HLS mapping 与语义等价路径。Planning 输出是 smoke，T6 增量为 0。
