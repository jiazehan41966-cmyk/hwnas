# Strict40 Next RL Expansion Boundary

Date: 2026-05-09

## Current Coverage Finding

The 40-entry strict LUT is sufficient for the compact plumbing search, but it is
not sufficient for a larger strict full-block NAS search under the current cost
estimator query contract.

The current estimator queries NAS blocks as whole blocks, for example:

- `conv`
- `mbconv_e3_k3`
- `mbconv_e3_k5`
- `mbconv_e6_k3`
- `mbconv_e6_k5`
- `skip`

Although the strict LUT contains many measured internal `pw_conv` and `dw_conv`
entries, those are not currently used to assemble a full `mbconv` cost query.
Therefore they do not safely expand the current full-block strict search space.

## Full-Block Entries Usable By The Current Compact Search

The current four-candidate strict search uses:

- stem: `conv`, 1 -> 32, k3, stride 2, resolution 224
- stage0: `conv`, 32 -> 16, k1, stride 1, resolution 112
- stage1 choices, 16 -> 24, stride 2, resolution 112:
  - `mbconv_e3_k3`
  - `mbconv_e3_k5`
  - `mbconv_e6_k3`
  - `mbconv_e6_k5`
- stage2 fixed: `mbconv_e6_k3`, 24 -> 32, stride 2, resolution 56
- stage3 fixed: `skip`, 32 -> 32, stride 1, resolution 28

This gives 4 exact-covered architectures.

## Why A Larger Strict RL Run Is Not Started Yet

A larger RL run should have more candidate combinations and 50-200 episodes.
Under strict formal LUT, expanding `stage_block_choices` without additional
measured full-block entries would create true misses or deferred hits, which
would invalidate the strict40 chain-validation contract.

Running 50-200 episodes on the same 4-candidate space would still be a small
sampling exercise, not a meaningful next RL experiment.

## Recommended Measurement Additions

To expand strict RL while keeping full-block lookup semantics, measure more
whole NAS blocks at the exact shapes used by the search space.

Minimum useful additions:

- stage2 alternatives, 24 -> 32, stride 2, resolution 56:
  - `mbconv_e3_k3`
  - `mbconv_e3_k5`
  - `mbconv_e6_k5`
- stage3 alternatives, 32 -> 32, stride 1, resolution 28:
  - `mbconv_e3_k3`
  - `mbconv_e3_k5`
  - `mbconv_e6_k3`
  - `mbconv_e6_k5`
- optional stage0 alternatives, 32 -> 16, stride 1, resolution 112:
  - `mbconv_e3_k3`
  - `mbconv_e3_k5`
  - `mbconv_e6_k3`
  - `mbconv_e6_k5`

With only the stage2 and stage3 additions, the compact strict space can expand
from 4 to roughly 4 x 4 x 5 = 80 combinations, including the current stage3
`skip` option. That is large enough for a first 50-episode RL signal while
remaining strict-LUT clean.

## Alternative Engineering Path

Instead of measuring more whole blocks, the estimator could be extended to
compose measured `pw_conv` and `dw_conv` sub-operators into a full `mbconv`
estimate. That is a separate modeling change and must be validated carefully,
because it changes the LUT contract from whole-block measured latency/resource
to composed sub-operator latency/resource.
