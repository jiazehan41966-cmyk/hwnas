# T6 structural candidate-pool freeze: 2026-07-17

## Decision closed

The earlier v3/v4 deployment pools contain only four semantic-safe operator signatures after paused sonar operators are excluded. They cannot support the predeclared five-family grouped comparison. A broader 200-row `mobile_anchor` search-space probe contains 200 unique encodings using only `dw_pw_conv`, `mbconv`, `fused_mbconv` and legal `skip` blocks, so it is used strictly as a structural design-of-experiments source.

## Frozen pool

- Selected candidates: 100 unique encodings.
- Families: five, exactly 20 candidates each.
- Family definitions:
  - `skip_heavy`: skip blocks are at least 25% of all blocks;
  - the three dominant families: one non-skip operator has a strict majority; pure networks map to the corresponding dominant family;
  - `mixed_balanced`: no non-skip operator has a strict majority.
- Grouped CV: five leave-one-family-out folds, each with 80 training and 20 held-out candidates.
- Sampling balance: up to five legacy analytically infeasible rows per family are retained, then filled with feasible rows using a salted encoding-hash order. The legacy flag is a sampling stratum only.

## Evidence bindings

- Builder SHA256: `f7016edd07c3b9ccdc20eb81496e6d4f99758e1bfda3c47f17570b773da61f32`.
- Auditor SHA256: `2da583d12e6313445a2f0c6c6df9ab3c8ee06ecc3741f84b616b166d9b562d44`.
- Manifest SHA256: `046e1ecba4e73306beb46e381070b5c251ef9d59b08b8045c635b1a905d48f65`.
- Independent audit SHA256: `f93b129a6249e6e4c75496b26dbeccf8b10cf41ea6605624029674cbe93ec43d`.
- Independent audit: 200 source rows, 200 unique source encodings, 100 selected unique encodings, five balanced folds, zero errors.

## Evidence boundary

The manifest is `STRUCTURAL_POOL_FROZEN_TRUTH_NOT_COLLECTED`. It does not validate old search accuracy, analytic resources, HLS performance, route feasibility or HARP prediction quality. Formal T6 truth remains `0/100`; F4 and T6 stay `PENDING`.

## Execution gate

First export one candidate per family as a source-linked complete-network HLS top and require semantic equivalence before csynth/route. If any family cannot be represented faithfully by the current generator, stop that family and repair the generator; do not substitute repeated encodings from another family. After five family pilots pass, queue the remaining 95 candidates and retain every synthesis/route failure with its stage and category.
