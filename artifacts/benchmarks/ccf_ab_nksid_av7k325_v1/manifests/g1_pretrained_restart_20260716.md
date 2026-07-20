# G1 pretrained full restart: 2026-07-16

- Status: `INTERRUPTED_BY_SYSTEM_REBOOT`.
- Method: `imagenet_pretrained_mobilenet_v2`, planned 15 fold-seed units.
- New run fingerprint: `ff1d84701305b45488e321213320adef3f54cfd2483f38b1d8fa1719befd42b0`.
- Source freeze: `PASS`, 556/556 files, manifest SHA256 `8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc`.
- The four pre-shutdown records are retained under `results/protocol/g1_clean_20260711/interrupted_20260716_0407/` and are not mixed into the restarted formal run.
- No `--force` override was used. Formal results remain pending until all 15 new atomic records exist and pass independent hash/metric/protocol checks.
- Windows recorded a Kernel API reboot transition at 07:45:36 and OS restart at 07:45:49. The run produced 0/15 atomic records and no Python traceback or CUDA OOM.
- This attempt is retained only as interruption evidence and is superseded by the persistent scheduled execution.
