# Hardware calibration v2

- G2 pass: `False`
- strict40 measured rows: `40`
- unique network rows: `6`
- unique mainline network rows: `4`
- denoise/edge/mixconv are excluded from the semantic-safe mainline.

| tier | unique n | metric | MAPE | P90 APE | Spearman | hard screen |
|---|---:|---|---:|---:|---:|---|
| analytic_to_hls_operator | 34 | latency_ms | 3127.2% | 151.9% | 0.8042515650626251 | False |
| analytic_to_hls_operator | 34 | dsp | 24.9% | 65.0% | 0.44729308754394503 | False |
| analytic_to_hls_operator | 34 | lut | 12.9% | 24.5% | 0.9409787692728553 | True |
| analytic_to_hls_operator | 34 | bram | 54.8% | 111.2% | 0.7119495390255699 | False |
| hls_to_post_route_operator | 43 | latency_ms | 4.5% | 14.9% | 0.9915556389566523 | True |
| hls_to_post_route_operator | 43 | dsp | 3.3% | 9.7% | 0.9976536148995454 | True |
| hls_to_post_route_operator | 43 | lut | 39.5% | 93.2% | 0.8939897311990336 | False |
| hls_to_post_route_operator | 43 | bram | 44.9% | 83.0% | 0.875916694630822 | False |
| legacy_analytic_to_route_com5_network_diagnostic | 4 | latency_ms | 41.4% | 52.4% | -0.6 | False |
| legacy_analytic_to_route_com5_network_diagnostic | 4 | dsp | 10.9% | 11.5% | 1.0 | True |
| legacy_analytic_to_route_com5_network_diagnostic | 4 | lut | 13.8% | 14.9% | 0.6 | True |
| legacy_analytic_to_route_com5_network_diagnostic | 4 | bram | 9.8% | 10.3% | 1.0 | True |

## G2 blockers

- four frozen independent full-network probes are not route/COM5 complete
- candidate-level HLS evidence completeness has not passed for a shortlist
