# G1 persistent execution: 2026-07-16

- Temporary Windows scheduled task: `Codex_HWNAS_G1_20260716`.
- Triggers: current-user logon plus a 30-minute repetition window for 30 days (extended from seven days at 17:52 CST); multiple instances are ignored, so a live training process is never duplicated. Execution time is unlimited; battery execution and wake-to-run are enabled.
- Runtime wrapper: `artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/runtime/run_g1_persistent_20260716.ps1`.
- Pretrained formal run started at 14:34:12 CST with fingerprint `e9cc0c17cb17773396673c06cfa15157b7f8d8559454134b2da9a7ab76a47f01`.
- Source freeze at launch: `PASS`, 556/556 files.
- `powercfg /requests` confirms both Task Scheduler and the PowerShell orchestrator hold active `SYSTEM` power requests; idle sleep is therefore suppressed while training runs.
- Runtime status is intentionally stored in `results/protocol/g1_clean_20260711/g1_persistent_runtime_20260716.txt`, not JSON, so reboot/resume does not alter `git_code_provenance`.
- On logon after a reboot, the wrapper invokes the same formal entrypoint with `--resume`. Completed atomic pairs are skipped only if the run fingerprint remains compatible; `--force` is never used.
- After pretrained reaches 15/15, `audit_g1_run.py` must independently pass checkpoint/prediction hashes, row bindings, metric recomputation, fold disjointness and the 2617-sample union before NAS champion starts.
- After NAS champion reaches 15/15, the same independent audit is required. The scheduled task must be removed only after all generated evidence is archived and checked.
- The original scratch run remains scientifically intact but its generated `code_patch.diff` was overwritten by a failed resume. The wrapper therefore creates a new `g1_mobilenet_v2_scratch_v2` 15-unit run and requires a clean independent audit; it never weakens the audit or relabels the old artifact as passing.
- Once all four closed-set methods have 60 audited units, the wrapper builds T2/F6, executes the frozen 21-condition sonar-corruption protocol for each method, and builds T4/T9/F7/F8. These outputs remain `PENDING_G1_LEDGER`; `f_robust` is not redefined. The plotting backend is a separate system Python runtime with matplotlib, so the active CUDA training environment is not mutated.
- The updated wrapper then runs `sure_same_backbone`: pinned SURE FMFP+CRL+RegMixup+cosine components, MobileNetV2, the same 5 folds × 3 seeds, 150 epochs, dedicated `sure_2024_cuda` environment and canonical `run_eval_protocol.py` entrypoint. Its 15 records require the same independent audit before R3 may advance.
- After the four closed-set methods pass, the reloaded wrapper continues the frozen 5-known/3-unknown experiment in order: CE+MSP, DMCL author loss, and PLUD author loss. Each uses MobileNetV2, 5 folds × 3 seeds, 150 epochs, the canonical entrypoint, source-freeze verification, and its registered dedicated environment where applicable.
- `audit_open_run.txt` independently checks all 15 checkpoint/prediction hashes, row provenance and threshold decisions, the 5-known/3-unknown manifest, fold disjointness and 2617-sample union, then recomputes known macro-F1, NMA, OSFM, OSCRmac, unknown AUROC and FPR95. Tests on the CE+MSP and DMCL method smokes reproduce all six reported values exactly; they remain non-claimable because those smoke runs lack formal 15-unit/source-freeze completeness.
- After all 45 open-set units pass, the wrapper builds T3 and the combined closed/open F5, appends open-set paired statistics to T9, refreshes artifact availability, reruns benchmark readiness, and regenerates the conservative measurement-first ledger. Hardware and power states are not promoted by this software-only chain.

This file documents execution control and does not support a model-performance claim.

## First atomic records observed at 16:14 CST

- The restarted pretrained run has atomically written fold 0 / seeds 42 and 43; the aggregate manifest remains unfinalized while later pairs train.
- Seed 42: macro-F1 `0.9108403453238872`, top1 `0.9730769230769231`, best epoch `80`, 520 unique outer samples.
- Seed 43: macro-F1 `0.91885352145244`, top1 `0.975`, best epoch `62`, 520 unique outer samples.
- Both checkpoint and prediction-file SHA256 values match their run records; every prediction row binds the new run fingerprint, code state, checkpoint, split, dataset and method, and both seeds use the same fold-0 outer sample IDs.
- After removing only run-specific provenance fields (`config_sha`, `code_state_sha`, `checkpoint_sha`, `claimability_status`), both new prediction files are byte-for-structure identical to the corresponding pre-shutdown scientific rows. Their metrics and best epochs are exactly equal; differing full-file/checkpoint hashes correctly reflect the new provenance cohort.
- These are partial execution checks only. No aggregate pretrained claim is allowed before 15/15 and the independent full-run audit.
- The 15:44 repetition trigger returned `0x800710E0` while the 14:34 instance remained `Running`; with `MultipleInstances=IgnoreNew`, this is the expected refusal of a duplicate instance, not a training failure.

## Additional atomic records observed by 17:26 CST

- Fold 0 / seed 44: macro-F1 `0.9688367188367187`, top1 `0.9865384615384616`, best epoch `149`; checkpoint/prediction hashes and recomputed metrics pass.
- Fold 1 / seed 42: macro-F1 `0.9213366533030154`, top1 `0.975`, best epoch `79`; its scientific prediction fields, metrics and best epoch exactly match the pre-shutdown archive.
- Fold 1 / seed 43 was atomically written at 17:26 CST. The run is therefore 5/15; aggregate claimability remains unavailable until the full independent audit passes.

## Evidence-binding hardening at 18:12 CST

- A read-only partial audit revalidated all five completed pretrained units: checkpoint and prediction SHA256, unique sample IDs, row provenance, fold/seed consistency, and independently recomputed macro-F1, top1 and weighted-F1 all pass.
- The persistent wrapper now reuses an audit only when its run directory, method, run fingerprint, auditor version and all 15 checkpoint/prediction hashes still match the current run. For closed-set audits that predate record-file hashing, the audit must also be newer than every record; the next formal audit is regenerated if any binding fails.
- `audit_open_run.txt` now records the run-record, prediction and checkpoint SHA256 values for every fold-seed unit. This closes the stale-PASS reuse gap without changing the frozen training source or the active G1 code-state fingerprint.
- The closed/open artifact builders and the sonar-corruption evaluator independently enforce the same run/method/fingerprint/file binding. A synthetic contract test confirms that matching bindings pass and a changed open-set fingerprint is rejected before any formal artifact is built.
- Source-freeze verification after these runtime-only changes remains `PASS`, 556/556.

## Sixth atomic record and figure-pipeline contract

- Fold 1 / seed 44 was atomically written with macro-F1 `0.9328381379846484`, top1 `0.9788461538461538`, weighted-F1 `0.9796292735122349`, and best epoch `100`.
- Its checkpoint/prediction SHA256, 520 unique rows, row-level run/checkpoint/split/method bindings, source-freeze binding and independently recomputed metrics all pass. The pretrained run is 6/15 and G1 remains `PENDING`.
- Synthetic-data-only smoke exports for F5 and F7 produced SVG, PDF, 300 dpi PNG, source CSV and meta JSON with generator/contract/input/output hashes. These smoke values are explicitly non-scientific and are stored only under `results/benchmarks/ccf_ab_nksid_av7k325_v1/smoke/artifact_builder_contract_20260716/`.
- Visual QA caught and repaired an F7 legend/title collision and F5 panel-label/whitespace issues. The second render has no observed overlap; the same corrected generators will be used only after audited formal inputs exist.

## Resume boundary and 14/15 checkpoint at 08:19 CST, 2026-07-17

- Windows entered an explicit `Button or Lid` sleep at 22:28:56 CST and resumed through the power button at 08:11:08 CST. The scheduled task and Python process remained the same process cohort; this interval is excluded from any interpretation of training throughput or active GPU time.
- `SetThreadExecutionState` and Task Scheduler wake controls prevent idle sleep but cannot override an explicit lid/button sleep. No system lid policy was changed.
- After wake, the active child process continued accumulating CPU time and using the GPU. At 08:19 CST it used approximately 28% GPU utilization, 929 MiB device memory and 25.65 W board-reported GPU power; these are execution-health diagnostics only, not experiment power evidence.
- Fourteen atomic units now exist with matching run record, prediction JSONL and checkpoint files. The latest complete unit is fold 4 / seed 43; fold 4 / seed 44 is still training. Aggregate pretrained status therefore remains `PENDING_15_OF_15` and no summary metric is claimable yet.
- A read-only partial audit had already verified all 14 completed units for checkpoint/prediction hashes, unique sample rows, row-level provenance, source-freeze binding and independently recomputed classification metrics. The full 15-unit audit remains mandatory after the final atomic unit is written.
- Targeted tests for exact hypervolume, frozen three-objective anytime-HV handling, search-method comparison contracts and the evidence-bounded HW-PR paper-spec helper pass: `15 passed`. This proves the software contracts only; G3 remains `FROZEN`, no formal NAS run was started, and T5/F2/F3 remain unavailable.

## Pretrained 15/15 acceptance and NAS-champion transition at 08:56 CST, 2026-07-17

- The final pretrained atomic unit, fold 4 / seed 44, completed at 08:56 CST. It contains 537 unique outer samples, macro-F1 `0.9533467761289156`, top1 `0.9851024208566108`, weighted-F1 `0.9854032482756044`, and best epoch `52`.
- Its prediction SHA256 is `1e9248046d5210fdb544d4aad77b665f11d8d6ad5ac65936b80207b7660e4564`; its checkpoint SHA256 is `fe839a51a063db972b2df4cbd686a868aa679cc5aa84a195339992fe5c2d60ab`.
- The completed 15-unit method summary reports macro-F1 `0.9316318800867744 +/- 0.024188436767340244`, top1 `0.9778262426586449 +/- 0.008171489678581206`, and weighted-F1 `0.9785838641890703 +/- 0.008001145628411041` (mean +/- standard deviation over 5 folds x 3 seeds). These are method-level frozen-protocol summaries, not a cross-method significance result.
- The independent audit at `artifacts/benchmarks/ccf_ab_nksid_av7k325_v1/manifests/g1_pretrained_independent_audit_20260716.json` passed with 15 records and no errors. Its SHA256 is `f7e5c6c1acb020fa3eaf38449ba93c2ea66df4e8fcf34fa1638e0538fcca5b70`; auditor source SHA256 is `021355fb92dd4bcfa9443a8166c9a9c84e1c5cf333fb907d0f4c0a197c316399`.
- The audit recomputes checkpoint and prediction hashes and classification metrics, checks row-level provenance, verifies the outer sample IDs are identical across seeds within each fold, rejects fold overlap, and verifies the five-fold union contains exactly 2,617 samples. The audit was generated after the final bound output.
- The source-freeze verification then remained `PASS`, 556/556 files, with manifest SHA256 `8b4de1d5bf8931c7a175cf913abd95b7a0a63848b2eaea3b2a87bc09ea2665dc` and retained archive SHA256 `376e9e162240a2e15dd8456882c9f209ad16bfc8b8da5244fac51342c8d4410d`.
- The persistent wrapper accepted pretrained as 15/15 and started `frozen_nas_champion` (`g1_rl_arch_135_legacy_selected`) at 08:56:13 CST through the canonical `run_eval_protocol.py` entrypoint. At the first post-launch check, its manifest and code patch existed but no atomic fold-seed result had yet been written; the training process remained alive.
- G1 remains `PENDING`: pretrained is one accepted 15-unit method, while the NAS champion, the clean scratch-v2 rerun, and same-backbone SURE must each independently reach and pass 15/15 before T2/F6 or paired significance claims are released.

## First NAS-champion atomic-unit audit at 09:24 CST, 2026-07-17

- `frozen_nas_champion` fold 0 / seed 42 completed with 520 unique outer samples, macro-F1 `0.6847830910932077`, top1 `0.801923076923077`, weighted-F1 `0.8455041709594839`, and best epoch `150`.
- Independent recomputation from `outer_predictions_fold0_seed42.jsonl` returned macro-F1 `0.6847830910932076`, top1 `0.801923076923077`, and weighted-F1 `0.8455041709594839`; the macro-F1 difference is floating-point rounding below `1e-12`.
- Prediction SHA256 `31f5705e981f17e91b8b7b8e8d204c94c202de37435e91f2ff53a17123fefb84` and checkpoint SHA256 `b864ced54858d83a880e68cb6e7c2256c45e6c260a4dfe88e897c71ec94d0014` both match the bound run record.
- All 520 sample IDs and outer positions are unique. Targets match dataset labels, stored correctness flags match predictions, predictions match logit argmax, and confidence differs from independently recomputed softmax maximum by at most `1.06e-7`.
- Every row binds fold `0`, seed `42`, split `outer_val`, method `frozen_nas_champion`, checkpoint, run fingerprint, code-state SHA and `PENDING` claimability with zero mismatches. The run fingerprint is `1ce4ba03c9a29a9dc7c8cdd6e751944f6b8de4f9f98a581617e1c6ca6e1e0ea9`; candidate SHA256 is `3e0f0d943ca31f659ee545a04d47bac985e6eebe508a3a1ba2d41460c818543a`.
- This establishes only one accepted atomic unit (`1/15`) for the NAS champion. No method aggregate, cross-method comparison, T2/F6 table or G1 promotion is supported until all 15 units and the independent full-run audit pass.
- A second recomputation using the canonical calibration implementation confirms the stored confusion matrix and all eight per-class F1 values exactly. AURC, failure AUROC and FPR95 are also exact; NLL, Brier and ECE differ by at most `3.38e-9` after reconstructing probabilities from persisted logits, below the frozen `1e-6` numerical tolerance. This calibration check therefore passes without changing the `1/15` claim boundary.

## Second NAS-champion atomic-unit audit at 09:50 CST, 2026-07-17

- `frozen_nas_champion` fold 0 / seed 43 completed with 520 unique outer samples, macro-F1 `0.6741714542567324`, top1 `0.7807692307692308`, weighted-F1 `0.8291172437324108`, and best epoch `143`.
- Prediction SHA256 `131979ef2317bca6dedf4d49c9f44c4058939cedf4f3a634cb1a1a6c56b6c5cc` and checkpoint SHA256 `8fc0eb8cedd3ac79ccf0ee99a56eb1b7dc4348b7774bd79dd9cef60d83317c1b` both match the run record.
- Classification metrics recomputed exactly from 520 unique rows. The confusion matrix and all per-class F1 values match exactly; calibration metrics reconstructed from persisted logits differ by at most `1.46e-9`, below the frozen `1e-6` tolerance. Prediction/logit argmax, confidence, correctness, dataset label and all row-level run/checkpoint/code/method bindings pass.
- Fold 0 seeds 42 and 43 have the same ordered outer sample IDs and targets. Their full split SHA values correctly differ because the canonical split payload binds the seed-specific inner train/validation partition: both 1,783/314/520 partitions are disjoint and cover all 2,617 samples, their outer indices match, and their inner/train indices differ. Seed 43's recomputed split SHA is `badb981353af3ead2e6bfae3780533679ab91f0e01dcc47f50bd1afc41e94f4d`, matching its record and all prediction rows.
- The NAS champion now has two independently accepted atomic units (`2/15`). No method aggregate, cross-method comparison, T2/F6 table or G1 promotion is supported yet.

## Third NAS-champion atomic-unit audit at 10:23 CST, 2026-07-17

- `frozen_nas_champion` fold 0 / seed 44 completed with 520 unique outer samples, macro-F1 `0.6437996520760838`, top1 `0.7769230769230769`, weighted-F1 `0.8177937310247814`, and best epoch `78`.
- Prediction SHA256 `20ed865db0f474eece3680bbae1c4142725ee6b0afe249401ce1f5439a271e26` and checkpoint SHA256 `764ba93194e27651de2375828dfea78e977dc72eb252bf1b19fa9b8c35829213` match the run record. Classification metrics and confusion matrix recompute exactly; calibration differs by at most `2.65e-9`, and confidence reconstructed from logits differs by at most `1.46e-7`.
- All row-level fold/seed/split/method/checkpoint/run/split/code-state/claimability bindings pass. Its seed-specific partition has 1,783 train, 314 inner-validation and 520 outer-validation indices, is disjoint, and covers 2,617 samples. Its split SHA is `a7f111c6085c79cf2e3438a340e69e5f5c93c1450232c80fec37f470c341fc00`.
- All three fold-0 seeds share the same ordered outer sample IDs and targets while retaining distinct seed-specific inner splits. The NAS champion now has three independently accepted atomic units (`3/15`); this completes fold 0 only and still supports no method aggregate or G1 promotion.

## Fourth NAS-champion atomic-unit audit at 10:38 CST, 2026-07-17

- `frozen_nas_champion` fold 1 / seed 42 completed with 520 unique outer samples, macro-F1 `0.7049382928198581`, top1 `0.8442307692307692`, weighted-F1 `0.8727932731273752`, and best epoch `87`.
- Prediction SHA256 `083cfd74e4c1fd68cc6acfb27960011046296ff89bc5590d1a60b225a47709c1` and checkpoint SHA256 `e8b4282b9bc563403e2687e0d562cf046ede6acbbf286dec736725ae429355dc` match the record. Classification/confusion metrics recompute exactly; calibration differs by at most `1.36e-9`, and all row-level split/checkpoint/run/code bindings pass.
- The fold-1 seed-42 partition is disjoint and covers all 2,617 samples; its split SHA is `ed750eeb9de54aee17c05ba0d9ea2d6b4a798f648b04656340b043396bb63724`. Its 520 outer samples are completely disjoint from the fold-0 outer set, and the two-fold union contains exactly 1,040 samples.
- The NAS champion now has four independently accepted atomic units (`4/15`). Fold 1 is incomplete, so no method aggregate, T2/F6 output or G1 promotion is supported.

## Fifth NAS-champion atomic-unit audit at 10:53 CST, 2026-07-17

- `frozen_nas_champion` fold 1 / seed 43 completed with 520 unique outer samples, macro-F1 `0.6551299124560107`, top1 `0.8096153846153846`, weighted-F1 `0.832827565584705`, ECE `0.1608834275545982`, AURC `0.06386511506988465`, and best epoch `99`.
- Prediction SHA256 `bd705a0f87a0a57d09d8c7eba2e25bf448e0ac3db2255a861de6e4d576f3558f` and checkpoint SHA256 `7f8e1fd3027c6d1445e3238e6161e37e0b76046e0ff1b181f915159dd3123b96` match the bound record. Its split SHA is `709c2a0f9ec6df25e35bdbb10df71d028144bff002c82150044a55d2d03f719b`.
- The reusable partial auditor independently rechecked all five available NAS units: file hashes, 520-row uniqueness and positions, float32 logit argmax/confidence, row-level checkpoint/run/split/data/code provenance, classification metrics, all eight per-class F1 values, confusion matrices, calibration/risk metrics, within-fold sample/target order and completed-fold disjointness. It returns `PASS`, 5 accepted / 5 observed, zero errors.
- Retained audit: `results/benchmarks/ccf_ab_nksid_av7k325_v1/g1_partial_audits/nas_champion_partial.json`, SHA256 `76fc749e93afe81559a3ccacea2fb8b57c5e6941488b8929d58e9b199d817d62`. Auditor SHA256 is `668c99b75dd6441fd0fcc47f88ff7783b9988851b9d325b5449a8f77055bd26f`.
- The same partial auditor cross-checked the already completed pretrained directory and accepted all 15/15 units with zero errors, consistent with the separate full-run audit. Cross-check SHA256: `f221d93eb22799a6bcf88b4746c6f33ffd52458682e11edf4cb566814ff43f42`.
- This is an explicitly partial audit and never grants aggregate claimability. The NAS champion is now `5/15`; fold 1 seed 44 and all fold 2–4 units remain, followed by the mandatory independent 15-unit full-run audit.

## Sixth NAS-champion atomic-unit audit at 11:34 CST, 2026-07-17

- `frozen_nas_champion` fold 1 / seed 44 completed with 520 unique outer samples, macro-F1 `0.6734461530449827`, top1 `0.7692307692307693`, weighted-F1 `0.8233374491327963`, ECE `0.14412848201508704`, AURC `0.08911950979022955`, and best epoch `138`.
- Prediction SHA256 `ad033ae9d79b3645372760bc861ab48993fd13053adc835850113eff52b25e7f` and checkpoint SHA256 `bc59006de570c63995b1dce1023853c3c9a05f86d972f43c4cbd0ad58f476f52` match their files and bound record. Its split SHA256 is `e05314b547bca5ddd2f5d2db4e570af038847acb53e320717f9d2ebb0d431fa0`.
- The reusable partial auditor independently accepted all six observed NAS units with zero errors. Retained audit SHA256 is `486eee68875987f6bd5414309bc8dfeddcd5d2a751fe3bbbad20ed7e1ad8bde6`.
- Folds 0 and 1 are now complete across seeds 42/43/44. The NAS champion is `6/15`; no aggregate, cross-method, T2/F6 or G1 claim is released before 15/15 plus the full independent audit.

## Seventh NAS-champion atomic-unit audit at 11:53 CST, 2026-07-17

- `frozen_nas_champion` fold 2 / seed 42 completed with 520 unique outer samples, macro-F1 `0.6896732793596113`, top1 `0.8134615384615385`, weighted-F1 `0.8552309116812645`, ECE `0.1888490591484767`, AURC `0.07369162331872947`, and best epoch `84`.
- Prediction SHA256 `e57ae9e5e400b1dbfdb2ab2342cdb06d783fdbf81f0715d674f93a817ceb7f0a` and checkpoint SHA256 `b28db80cc811c6e26231c5ebf3f4e7197e922a52c46359ffebfab76cdd8904a5` match their files and bound record. Its split SHA256 is `6b77845044dbe97c2c8406539c5e309f49bb35dc89383aafe435f18801f73978`.
- The reusable partial auditor independently accepted all seven observed NAS units with zero errors. Retained audit SHA256 is `29fb007730c93eaca76a192f7d570d4a4e853f0f4b624edd9509983921c56c97`.
- The NAS champion is now `7/15`; fold 2 remains incomplete. No aggregate, cross-method, T2/F6 or G1 claim is released before 15/15 plus the full independent audit.

## Eighth through thirteenth NAS-champion atomic-unit audit at 14:16 CST, 2026-07-17

Six additional atomic units were observed and then independently recomputed together with the previous seven:

| Fold | Seed | Samples | macro-F1 | top1 | weighted-F1 | ECE | AURC | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 43 | 520 | 0.7169423734421112 | 0.8519230769230769 | 0.8752139838885870 | 0.2319700289231080 | 0.06036178899141582 | 75 |
| 2 | 44 | 520 | 0.6767176191741419 | 0.8076923076923077 | 0.8456112892130125 | 0.20236247889697556 | 0.07513117125680542 | 93 |
| 3 | 42 | 520 | 0.7307331919130048 | 0.8192307692307692 | 0.8566289503049418 | 0.18407412125514105 | 0.06361807119466913 | 97 |
| 3 | 43 | 520 | 0.7420635546249515 | 0.8423076923076923 | 0.8788305335591990 | 0.1997013930231333 | 0.053866134040337604 | 123 |
| 3 | 44 | 520 | 0.6995334498193501 | 0.8096153846153846 | 0.8514615756835606 | 0.16996272212037672 | 0.06230684962204441 | 94 |
| 4 | 42 | 537 | 0.6737691725267392 | 0.7839851024208566 | 0.8250994037938130 | 0.1507281878307545 | 0.0920986980883426 | 86 |

Exact bindings for the six newly accepted units are:

- Fold 2 / seed 43: split `e95ad7c594188c8590a2c337db7b062ae48c7406a9b8f951e276adfeef240dad`, checkpoint `63157d1bad642e18989a2147a2538ad5ab9adbc2555c0dbd86b6acfb1cc8b852`, prediction `0f51c8966440bf22fed60be05183726b6aa6fa909fc9b1f6b3d176e97af640c1`.
- Fold 2 / seed 44: split `168a5d1029420a6f0b72f77e25d906687a82eaecbdc3a1dd7648e289a1ccde27`, checkpoint `55e604e1c06ecd93950d7dd0111611c9956265c4ee567084137890803dfa1b7f`, prediction `580851fb1f834ab8d6afcd9eb9a9693e0575307e5e564c6cae26c76d056eaf2d`.
- Fold 3 / seed 42: split `c275c9629c323911cbbec2155cd61d16e5ec2c4db7835ec972154b7a314cfc13`, checkpoint `0e0ce0c819827ca738b3fc9356cb8e859bec9d48647700b9a7f5b7f80bf79baf`, prediction `0cecf81edc0fad4e979bb4b9c3ed521345468adcaaa4e6de9d16ee6cdd5bbe49`.
- Fold 3 / seed 43: split `f0fbd8d68c9d123361c9290c3cbace2c58df708321cec3b872880ecfe90d46ca`, checkpoint `7a7f8138e2a08bf389f210676b503f420be9486bbc68e8470b8de3eebb284dc3`, prediction `aedb59ddd982841977731fa783dd515b16e46f1f81dd9fda62be1f76c4d41d54`.
- Fold 3 / seed 44: split `c634bbf0c04f2cd48d77467730d47e19b8d53674aee5d478f8cd7e94e1eed6ca`, checkpoint `7d182cfe783cfcb2604692f121057e21cd6c8a876b27b5893f8cbe243d08f736`, prediction `9f9c0e264ffc30a4e604e365fd7d7be7888b6fad27e1c6443fc95d0be17b0258`.
- Fold 4 / seed 42: split `84ebd0c1e6dcd2af062f978f03c698b753af808a5701cda6e6e7cbe6760c5b1b`, checkpoint `1084308856992c8eb11f657b844ef163d54fe3fe4a378c4cc8b50373ecdee21a`, prediction `3e0251130273e0564d476905752352d37188c9f8fa3b84be9f77426795438c7f`.

The partial auditor accepted all `13/13` observed units with zero errors and retains SHA256 `b2b474412f89f06be7ab555f6eabf94f4e696cdae97c4b14f3a42bed47b856e4`. Its scope remains `PARTIAL_ATOMIC_UNITS_ONLY`, `aggregate_claimable=false`, and two NAS units remain. Together with the independently accepted pretrained run, the closed-set campaign has `28/60` accepted atomic units; no T2/F6 aggregate or G1 promotion is supported yet.

## Fourteenth NAS unit and user-decision pause at 14:24 CST, 2026-07-17

- Fold 4 / seed 43 completed atomically with 537 samples, macro-F1 `0.7006401504235198`, top1 `0.8268156424581006`, weighted-F1 `0.8545149423506884`, ECE `0.17477996147877678`, AURC `0.04699735192663537`, and best epoch `109`.
- Its split, checkpoint and prediction SHA256 values are respectively `56e38b9fb478593a16b3a21b04e58bf27622fb31c023e0018683d475c33f1fb4`, `f2493868d56a7680186c0525501f4c0c97ab1c0b224c8a197ca61a82a4fbaa7d`, and `0de0e1ce5ff3fe7dc6e5e34af835d9c6d69821bc28c0135d76ba07ea69776632`.
- The independent partial audit accepted all `14/14` observed units with zero errors; retained audit SHA256 is `0e8da5427b19fb8ddf2aefb871ee4965839c21f17e294887b2668b50dc6e507c`.
- Across the 14 paired units, NAS mean macro-F1 is `0.6904529534` versus pretrained `0.9300808161`; mean difference is `-0.2396278627`, all 14 differences are negative, and the interim paired-bootstrap 95% interval is `[-0.2612163648, -0.2169519822]` using 10,000 replicates and seed `20260717`.
- Following the user-mandated anomaly rule, execution was stopped after the atomic write. PIDs 23088 and 19108 are stopped; no fold 4 / seed 44 artifact exists. Scheduled task `Codex_HWNAS_G1_20260716` is `Ready`, last result `2`, and no downstream SURE/scratch-v2 run was launched.
- State is `PAUSED_ATOMIC_BOUNDARY_PENDING_USER_DECISION`. The interim statistics are decision support, not a formal 15-unit aggregate. The user must decide whether to finish the last NAS unit, continue downstream baselines, or authorize a separately frozen diagnosis/protocol change.
