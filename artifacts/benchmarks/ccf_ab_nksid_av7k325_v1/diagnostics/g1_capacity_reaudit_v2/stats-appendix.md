# Statistical appendix

## Analysis unit and pairing

- Unit: one `(outer fold, seed)` run.
- Primary methods: `rl_arch_135` and the formal scratch-v2 MobileNetV2 rerun.
- Included pairs: 15 complete 150-epoch curves with the same fold/seed key.
- Late window: arithmetic mean over epochs 140-150 within each run; inference is across runs,
  not across epoch lines.

## Primary result

| Quantity | Estimate |
|---|---:|
| NAS mean | 0.682396 |
| Scratch MNV2 mean | 0.987815 |
| Paired mean delta (scratch − NAS) | 0.305419 |
| Paired bootstrap 95% CI | [0.300971, 0.310248] |
| Paired Cohen's dz | 32.081 |
| Exact two-sided sign-flip p | 0.00006104 |

The exact p-value is descriptive mechanism-triage evidence, not a license to claim that capacity
has been isolated. The bootstrap uses 50,000 paired resamples with seed 20260722.

## Coverage and exclusions

The primary NAS and scratch-v2 logs each contain 15 complete × 150-epoch curves. The historical
root pretrained log is retained as context only because it does not contain a balanced complete
15-run curve set. It is excluded from primary curve inference.
