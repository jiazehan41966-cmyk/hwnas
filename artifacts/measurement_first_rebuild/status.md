# Measurement-first rebuild status

- overall: `IN_PROGRESS`
- boundary: Implemented software gates are not experimental results. Missing training, csynth, route, COM5, or meter evidence remains PENDING.

| gate | status |
|---|---|
| G0_protocol | PASS |
| G1_accuracy_baselines | PENDING |
| G2_hardware_measurement | PENDING |
| G3_search | FROZEN |
| G4_int8_board | PENDING |
| power | NOT_MEASURED |
| G5_sonar_ablation | PAUSED |

## Current blockers

- G2: four frozen independent full-network probes are incomplete
- G2: fewer than eight unique semantic-safe full-network samples
- G2: independent interval-screening quality gates are not all PASS
- G2: candidate HLS shortlist coverage is not 100%
- G4: ptq_or_qat_accuracy
- G4: hls_bit_exact_parity
- G4: full_outer_validation_board
- G4: zero_board_numeric_mismatch
- G4: no_missing_board_samples
- power: external meter CSV acceptance has not passed
- G5: denoise/edge remain paused
