# Strict40 Plumbing Baseline

Date: 2026-05-09

This record freezes the short strict40 LUT plumbing run as a chain-validation
baseline. It must not be used as evidence of RL learning quality or architecture
accuracy quality.

## Inputs

- Status-authoritative CSV: `hls_lut_builder/board_harness/results/board_measure_status_current_impl.csv`
- Strict measured LUT: `hls_lut_builder/results/formal_lut_strict40_v1.json`
- Strict status file: `hls_lut_builder/results/formal_lut_status_strict40_v1.json`
- Short NKSID config: `configs/search/formal_lut_strict40_nksid_short_av7k325.yaml`

The strict LUT was generated with:

```powershell
python hls_lut_builder/scripts/generate_formal_lut.py --status-authoritative --output-lut-json hls_lut_builder/results/formal_lut_strict40_v1.json --output-status-json hls_lut_builder/results/formal_lut_status_strict40_v1.json
```

Generation integrity:

- `measured_entries`: 40
- `board_result_status_overrides`: 0
- status counts: 40 `measured`, 44 `defer_current_impl`

## Short RL Plumbing Run

Command:

```powershell
python run_search.py --config configs/search/formal_lut_strict40_nksid_short_av7k325.yaml
```

Result directory:

- `results/formal_lut_strict40_nksid_short_8ep`

Run status:

- completed
- feasible ratio: 8 / 8 episodes
- LUT hits: 40
- LUT misses: 0
- true_miss: 0
- deferred_hit: 0
- hit_rate: 1.0
- fallback_rate: 0.0

Trace metrics only:

- best candidate: `rl_arch_7`
- macro_f1: 0.47124796257717794
- top1: 0.7019230769230769
- latency_ms: 86.24667
- DSP: 363
- BRAM: 133
- LUT: 12870
- power_w: 9.0312

Interpretation boundary:

- This validates strict LUT generation, strict LUT query, constraint checking,
  reward calculation, NKSID dataloading, and artifact logging.
- This does not validate RL learning. The run used only 8 episodes and 1 eval
  epoch.
- macro_f1 and top1 are recorded only for traceability.

## Deterministic Four-Candidate Smoke

Command:

```powershell
python scripts/enumerate_strict40_candidates.py
```

Result directory:

- `results/strict40_deterministic_4candidate_smoke`

Summary:

- candidate_count: 4
- feasible: 4
- infeasible: 0
- strict_lut_ok: true
- each candidate: 5 LUT hits, 0 LUT misses, 0 true_miss, 0 deferred_hit

Enumerated stage1 choices:

- `mbconv_k3_e3_s2`
- `mbconv_k5_e3_s2`
- `mbconv_k3_e6_s2`
- `mbconv_k5_e6_s2`

Candidate cost trace:

| candidate | stage1 | latency_ms | DSP | BRAM | LUT | power_w |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| strict40_enum_0 | mbconv_k3_e3_s2 | 86.24667 | 363 | 133 | 12870 | 9.0312 |
| strict40_enum_1 | mbconv_k5_e3_s2 | 86.679175 | 363 | 169 | 13959 | 9.2472 |
| strict40_enum_2 | mbconv_k3_e6_s2 | 111.00411 | 363 | 150 | 13645 | 9.0312 |
| strict40_enum_3 | mbconv_k5_e6_s2 | 111.836215 | 363 | 190 | 14732 | 9.6252 |

## Next RL Boundary

For a real RL signal:

- increase candidate combinations through more measured-covered
  `stage_block_choices`
- increase episodes to 50-200
- increase eval_epochs beyond 1 before using macro_f1 or top1 for architecture
  comparison
- if strict LUT remains enabled, the expanded search space must stay within
  the measured LUT coverage, or new board measurements must be added first
