# F1 visual and archive QA: 2026-07-17

## Result

`F1: Evidence-aligned Benchmark Workflow / 分层证据与对标实验工作流`
passes the current visual and archival-format check. It is a conceptual workflow
figure only and supports no model-performance, hardware-performance or power
claim.

## Required files and hashes

| File | SHA256 | Check |
|---|---|---|
| `figures/f1.png` | `119c29589433c44b075b69e6693b017e3da38e331be0afa3050e50031363c98f` | present; 3600 x 1080 RGBA; 299.9994 dpi |
| `figures/f1.pdf` | `4a8d5100be14155c556718fb9f4bac552f4ca7ea278e28ee061835b7d8f75c87` | present; one 864 x 259.2 point page; selectable vector text; zero embedded raster images |
| `figures/f1_source.csv` | `92a0bb1f5d4d7dcb652ddbda26e3a63546fb1b87f31c35f6f56416133f8be29a` | present; hash matches `f1_meta.json` |
| `figures/f1_meta.json` | `1543512a4031a31526876698d4de717c95eb6bc7b0ec67488cd1df299e9d963d` | present; title, caption, supported claim, limitation, input and generator hashes recorded |

The metadata generator SHA256 is
`7dd9f5b6ec7a9d5b7c2661ddc67ef7839ecb95e540691671f5719323bcdc6ba6`,
which matches the current `scripts/build_benchmark_artifacts.py` file. The campaign
configuration input SHA256 is
`0bd44e881b7080c2cc155ac2930e6168044e0d9e3bcafecbaf3e18752d647260`.

## Visual inspection

- The five evidence layers are ordered left to right and connected without
  ambiguous branching: search proxy, frozen-protocol retrain, HLS/route, COM5
  board inference, and external power meter.
- English and Chinese stage labels are legible. Status labels are separated from
  evidence descriptions, and no title, node, arrow or caption is clipped or
  overlapping in the 3600 x 1080 render.
- The current conservative states remain visibly separate: `PENDING`,
  `G1 PENDING`, `G3 FROZEN`, `G2 PENDING`, and `NOT_MEASURED`.
- The figure correctly keeps GPU/runtime power diagnostics outside the external
  power-meter evidence layer.

## Encoding finding

The UTF-8 source CSV, metadata JSON, status JSON and rendered figure contain
correct Chinese text. Mojibake observed through Windows PowerShell 5.1 without an
explicit encoding was a terminal-decoding artifact. All future text QA commands
must use `Get-Content -Encoding UTF8` or a UTF-8-aware reader before declaring
file corruption.

## Acceptance boundary

F1 is accepted as an archival workflow figure. Its statuses are a snapshot, so a
future Gate transition requires regeneration from the canonical audited ledger;
the image must not be hand-edited. Passing this QA does not advance any Gate or
increase the count of completed empirical tables/figures.
