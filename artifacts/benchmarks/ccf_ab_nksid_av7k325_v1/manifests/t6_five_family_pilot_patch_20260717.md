# T6 Five-Family Export-Pilot Freeze

## Decision

The 100-candidate structural DOE is too large to enter complete-network export without first validating generator coverage for every declared architecture family. One deterministic pilot is therefore frozen per family. This pilot subset is an execution precheck, not a performance subsample and not a replacement for the required 100 complete-network HLS/route truth rows.

## Frozen inputs

| Family | Pool sample | Source candidate | Blocks | Encoding SHA256 |
|---|---|---|---:|---|
| `dw_pw_dominant` | `t6_dw_pw_dominant_04` | `probe_0000` | 4 | `47a1c40c725a6c71066b26cdf1aad23ea65c5ad7177c74d0822648083b9df407` |
| `mbconv_dominant` | `t6_mbconv_dominant_05` | `probe_0060` | 4 | `990070c9cf249f1eaeecf8d5e275e963690aab068707a15bb62207dbfe5e94c4` |
| `fused_mbconv_dominant` | `t6_fused_mbconv_dominant_09` | `probe_0048` | 4 | `f1335b4d5f45eba6b8a87c5e3e7add7a8f0f967c881a87c83de48b70cb63e662` |
| `skip_heavy` | `t6_skip_heavy_00` | `probe_0136` | 5 | `56749442d6ee3eaccf3735dc56fb769d47ca9be2fae8ac2d7395f4eb78febf89` |
| `mixed_balanced` | `t6_mixed_balanced_15` | `probe_0188` | 4 | `4a7ae1e28481b8252847d58f998ab1ded800a07c221fa0cc46b4b0491572e686` |

Selection requires the frozen source analytic flag to be true, then minimizes block count and uses the frozen salted selection key as the tie-break. This optimizes early generator coverage only. The source analytic flag remains an ordering hint and cannot be reported as HLS or route feasibility.

## Provenance

- Builder SHA256: `cdf82b5074e4ad92f011b938376b9f8a88c3e5aecf7a3a4efb07d5aea2249077`.
- Auditor SHA256: `2d52e7b0ff2347e759009ac0ec56aa97f8ad20073871be5fa8942246b5e3f3f6`.
- Manifest SHA256: `ae9aad20b0c10f5958a530f2860272d5455543970de19ba91e261b08efa7c17c`.
- Audit SHA256: `88010ac17644747aa9c7dc92ec1ddf1d4fd9a24468050a60def607dd88991117`; status `PASS`, five families, zero errors.
- Contract-test SHA256: `f794783038943f94de0529518501d1926d6b38b67d3c590cbc8a20feec4e5ab5`; valid input passes, while premature board requirement and duplicated family/encoding are rejected.

## Evidence boundary and release condition

All five pilots remain `COMPLETE_NETWORK_PENDING_EXPORT`; export, semantic equivalence, csynth and route are not yet collected. Formal T6 truth therefore remains `0/100`, and no T6/F4 result is available. A physical board is not required for these four stages. Only after all five exports pass semantic equivalence may the remaining 95 candidates be queued; route failures are retained, and bitstream/COM5 board work starts only after route-feasible selection.

## Existing-generator planning probe

The repository's isolated full-network planner was executed in `plan` mode for all five frozen candidates. The independent gap auditor, SHA256 `f2b9a28f7311ac5adc355aedc82ca578ae6bdff6f0dd98317005f58d61680d64`, produces `t6_five_family_mapping_gap_audit_v1.json.txt`, SHA256 `0d514c94d840a93ce2da57ed632f50092ef949fbc659e78eec06b67f638c8066`, with status `PASS_EXPECTED_GAP_CONFIRMED`.

All five plans are `not_generated_mapping_incomplete`; each has at least six missing component rows, no candidate-HLS mapping and a failed arch-84 identity gate. This proves that the frozen candidates cannot reuse the historical bitstream and that the current planner cannot directly supply T6 truth. After the running software chain reaches a new source-freeze boundary, the required repair is a source-linked candidate-specific operator/HLS mapping and semantic-equivalence path. Planning outputs remain smoke evidence and increment T6 by zero.
