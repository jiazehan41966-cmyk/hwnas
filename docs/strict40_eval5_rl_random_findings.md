# Strict40 Eval5 RL vs Random Findings

Date: 2026-05-10

## Runs

- RL: `results/formal_lut_strict40_nksid_rl_50ep_eval5`
- Random: `results/formal_lut_strict40_nksid_random_50c_eval5`
- Aggregated analysis: `results/strict40_rl_vs_random_eval5_analysis.json`

Both runs used the strict40 NKSID search space with `eval_epochs=5`.

## LUT And Feasibility

| run | evaluated | feasible | infeasible | strict LUT hits | true_miss | deferred_hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RL | 50 | 50 | 0 | 250 | 0 | 0 |
| Random | 50 | 50 | 0 | 500 | 0 | 0 |

Both runs have `hit_rate=1.0` and `fallback_rate=0.0`.

## Exploration

RL sampled only one candidate:

- `mbconv_k3_e3_s2`: 50 / 50

Random sampled all four candidates:

- `mbconv_k3_e3_s2`: 18 / 50
- `mbconv_k3_e6_s2`: 11 / 50
- `mbconv_k5_e3_s2`: 9 / 50
- `mbconv_k5_e6_s2`: 12 / 50

Decision: RL exploration is not normal. Do not scale this controller to 200
episodes before fixing exploration.

## Proxy Metrics

Top macro-F1:

- RL top-1: 0.5788598053 (`mbconv_k3_e3_s2`)
- Random top-1: 0.5614129406 (`mbconv_k3_e3_s2`)
- RL top-3 mean: 0.5720396167
- Random top-3 mean: 0.5599490060

This apparent RL top-K advantage is not useful evidence, because RL repeated the
same architecture 50 times and selected the best stochastic training instance.

Random group means:

| stage1 | n | mean macro_f1 | mean top1 |
| --- | ---: | ---: | ---: |
| `mbconv_k3_e3_s2` | 18 | 0.4797936018 | 0.6974358974 |
| `mbconv_k3_e6_s2` | 11 | 0.4687944130 | 0.7038461538 |
| `mbconv_k5_e3_s2` | 9 | 0.4945717804 | 0.7378205128 |
| `mbconv_k5_e6_s2` | 12 | 0.4529633336 | 0.6934294872 |

The mean macro-F1 range is 0.0416084468 and the mean top1 range is
0.0443910256, both above 1%. However, within-candidate variance is still large
under 5 epochs, so this is a weak proxy signal rather than final architecture
evidence.

## Controller Check

The controller has no explicit sampling temperature or entropy bonus. Initial
stage1 choice probabilities were already biased toward `mbconv_k3_e3_s2`
(`0.4452`), and after 50 episodes the latest checkpoint collapsed to:

- `mbconv_k3_e3_s2`: ~1.0
- other three candidates: ~0

Kernel and expand entropy both collapsed to approximately `1.19e-7`.

Next action: add temperature/entropy-controlled exploration or an explicit
exploration bonus before any 200-episode strict40 RL run.
