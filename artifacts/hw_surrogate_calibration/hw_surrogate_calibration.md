# Hardware surrogate calibration

Paired routed runs: 10 (skipped: 5)

| run | arch | est lat ms | meas lat ms | ratio | est DSP | meas DSP | est LUT | meas LUT | est BRAM | meas BRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_rank001_rl_arch_185_d34e207f23 | rl_arch_185 | 6.85 | 49.03 | 7.16 | 900 | 524 | 92362 | 15380 | 135 | 101 |
| full_rank001_rl_arch_60_5ce4a50d25 | rl_arch_60 | 5.32 | 24.84 | 4.67 | 900 | 524 | 92544 | 15354 | 135 | 101 |
| full_rank002_rl_arch_186_b58ae648c3 | rl_arch_186 | 7.29 | 49.06 | 6.73 | 1172 | 612 | 128445 | 18598 | 188 | 128 |
| full_rank003_rl_arch_135_b58ae648c3 | rl_arch_135 | 7.29 | 49.06 | 6.73 | 1172 | 612 | 128445 | 18598 | 188 | 128 |
| full_rank003_rl_arch_242_f22bc81934 | rl_arch_242 | 5.76 | 24.87 | 4.32 | 1172 | 612 | 128627 | 18554 | 188 | 128 |
| full_rank004_rl_arch_154_0d84539938 | rl_arch_154 | 38.06 | 39.01 | 1.02 | 904 | 528 | 93177 | 16368 | 138 | 108 |
| full_rank004_rl_arch_276_5ce4a50d25 | rl_arch_276 | 5.32 | 24.84 | 4.67 | 900 | 524 | 92544 | 15354 | 135 | 101 |
| full_rank005_rl_arch_169_0f984469ad | rl_arch_169 | 39.66 | 50.55 | 1.27 | 904 | 528 | 92940 | 16393 | 138 | 108 |
| full_rank006_rl_arch_175_d34e207f23 | rl_arch_175 | 6.85 | 49.03 | 7.16 | 900 | 524 | 92362 | 15380 | 135 | 101 |
| full_rank007_rl_arch_193_f22bc81934 | rl_arch_193 | 5.76 | 24.87 | 4.32 | 1172 | 612 | 128627 | 18554 | 188 | 128 |

## Ratio summary (measured / estimated)

| metric | n | geomean | min | max | MAPE of estimate |
|---|---:|---:|---:|---:|---:|
| latency_ms | 10 | 4.065 | 1.025 | 7.155 | 67.7% |
| dsp | 10 | 0.558 | 0.522 | 0.584 | 79.5% |
| lut | 10 | 0.159 | 0.144 | 0.176 | 531.1% |
| bram | 10 | 0.725 | 0.678 | 0.779 | 38.1% |

## Post-calibration residual (measured / calibrated estimate)

| metric | n | geomean | min | max | worst abs error |
|---|---:|---:|---:|---:|---:|
| latency_ms | 10 | 0.568 | 0.143 | 1.000 | 85.7% |
| dsp | 10 | 1.000 | 0.936 | 1.047 | 6.4% |
| lut | 10 | 1.000 | 0.907 | 1.109 | 10.9% |
| bram | 10 | 1.000 | 0.935 | 1.074 | 7.4% |

## Recommended analytic calibration

```json
{
  "latency_scale": 7.155491108861801,
  "dsp_scale": 0.5577738296422039,
  "lut_scale": 0.15898363117309786,
  "bram_scale": 0.7251733774191728
}
```

Apply via `hardware.analytic_calibration_path` in a search config;
factors touch only analytic-fallback layers, never LUT-measured entries.
