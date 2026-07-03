# Phase0 v4 Sonar-Op Search Precheck

Last updated: 2026-06-07

Status: historical pre-launch admission snapshot. The precheck evidence remains
valid for its stated scope, but its `not_launched` statements do not describe
the later Phase0 v4 execution state. Use
`docs/PHASE0_V4_SONAR_RESULTS.md` for current search/retrain/route/COM5 and
image-quality status.

## Scope

This precheck evaluates whether sonar-specific operators can be admitted into
the next Phase0 search round without weakening the Phase0 v3 low-DSP board
claim. It does not run server training, Vivado full route, or COM5
measurement, and it does not overwrite Phase0 v3 `rl_arch_186`, `rl_arch_242`,
or `rl_arch_276` artifacts.

Inputs:

- `docs/PROJECT_MEMORY.md`
- `docs/MOBILENETV2_OPERATOR_LUT_SPEC.md`
- `configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3_lowdsp_av7k325.yaml`
- `src/hwnas_fpga/models/builder.py`
- `src/hwnas_fpga/hardware/cost.py`
- `hls_lut_builder/results/nas_board_lut_strict_current84_arch84/nas_board_lut_strict.json`
- `hls_lut_builder/results/nas_board_lut_strict_current84_arch84/nas_board_lut_status.json`
- `hls_lut_builder/board_harness/configs/phase0_v3_lowdsp_override_catalog.yaml`
- `hls_lut_builder/configs/operator_manifest.yaml`

Generated evidence:

- `results/phase0_v4_sonar_op_precheck/sonar_op_coverage.json`
- `results/phase0_v4_sonar_op_precheck/sonar_op_coverage.md`
- `results/phase0_v4_sonar_op_precheck/diversity_audit_v3_lowdsp_guardrail.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/sonar_stage3_k3_timing_gate.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/formal_lut_phase0_v4_sonar_stage3_k3_pilot.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/formal_lut_status_phase0_v4_sonar_stage3_k3_pilot.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/merged_current84_arch84_plus_sonar_stage3_k3_lut.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/merged_current84_arch84_plus_sonar_stage3_k3_status.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/precheck_merged_with_lowdsp/sonar_op_coverage.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/diversity_audit_v4_sonar_stage3_k3_lowdsp_draft.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/launch_readiness/phase0_v4_sonar_stage3_k3_launch_readiness.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/launch_readiness/phase0_v4_sonar_stage3_k3_launch_readiness.md`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/pipeline_poll/phase0_v4_sonar_stage3_k3_pipeline_poll_latest.json`
- `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/pipeline_poll/phase0_v4_sonar_stage3_k3_pipeline_poll_latest.md`

## Conclusion

Do not automatically launch server Phase0 v4 RL300. The local admission
materials for the minimal stage3 k3 sonar pilot are now complete, but the
created search config is explicitly inactive and local-only until a separate
manual launch decision is made.

Final local admission result for the minimal pilot:

- admitted rows: `denoise k3 e1 stage3` and `edge k3 e1 stage3`;
- operator-level HLS/OOC downstream: pass for both rows at 200 MHz;
- isolated pilot strict LUT/status: pass for both rows;
- merged v3+pilot strict LUT/status: `60` measured LUT rows and `90` formal
  status rows, written only under the pilot directory;
- targeted precheck with low-DSP overrides: `PASS`, strict LUT hits `2/2`,
  true misses `0`, deferred hits `0`, low-DSP override coverage `2/2`;
- inactive v4 draft config created:
  `configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml`;
- diversity dry-run on the inactive draft config: `PASS`,
  `unique_encoding_count=8`, `feasible_unique_encoding_count=8`.

The broader sonar-op audit remains blocked for stage1/stage2, k5, and
`mixconv`. Those rows must not be added to formal search until they have their
own strict LUT/status and route evidence.

## 中文 Handoff 摘要

本轮目标不是启动 Phase0 v4 RL300，而是把最小声呐试点
`denoise/edge k3 e1 stage3` 从“可综合”推进到“具备 200 MHz
operator-level post-route 证据，并可被 strict LUT 和 low-DSP override
准入检查命中”。

当前最小试点已经完成本地准入闭环：

- `denoise`: HLS DSP `4`，post-route WNS `1.238 ns`，Fmax
  `265.816 MHz`，post-route DSP `4`；
- `edge`: HLS DSP `4`，post-route WNS `1.246 ns`，Fmax `266.383 MHz`，
  post-route DSP `4`；
- merged strict LUT/status 只写入 pilot 隔离目录，不修改 Phase0 v3
  canonical LUT/status；
- targeted precheck 结果为 `PASS`：strict hit `2/2`，true miss `0`，
  deferred `0`，low-DSP override `2/2`；
- inactive v4 draft config 已生成，但 `server_launch_status` 仍是
  `not_launched`；
- diversity dry-run 结果为 `PASS`：`unique_encoding_count=8`，
  `feasible_unique_encoding_count=8`。
- launch-readiness 包已生成，状态为
  `READY_FOR_MANUAL_SERVER_LAUNCH`，但其中 `runs_server=false`、
  `runs_training=false`、`runs_vivado=false`、`runs_com_measurement=false`。

结论：stage3 k3 的 `denoise` 和 `edge` 两个声呐候选已经具备本地
Phase0 v4 搜索准入材料；但是否真正开服务器跑 RL300 是下一步单独决策。
stage1/stage2、k5、`mixconv` 仍然不能进正式搜索。

## Coverage Matrix

Stage 0 remains the fixed Phase0 v3 low-DSP `conv k1 e1 32->16` stage. The
candidate sonar rows below cover the currently relevant stage 1-3 insertion
points.

| operator | kernel | expand | stage | shape | strict LUT | formal status | low-DSP override |
|---|---:|---:|---:|---|---|---|---|
| `denoise` | 3 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `denoise` | 3 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `denoise` | 3 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |
| `denoise` | 5 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `denoise` | 5 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `denoise` | 5 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |
| `edge` | 3 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `edge` | 3 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `edge` | 3 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |
| `edge` | 5 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `edge` | 5 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `edge` | 5 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |
| `mixconv` | 3 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `mixconv` | 3 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `mixconv` | 3 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |
| `mixconv` | 5 | 1 | 1 | `112:16->56:24 s2` | miss | missing | missing |
| `mixconv` | 5 | 1 | 2 | `56:24->28:32 s2` | miss | missing | missing |
| `mixconv` | 5 | 1 | 3 | `28:32->28:32 s1` | miss | missing | missing |

## Phase0 v4 Search-Space Draft

For the full sonar-op program, the formal v4 search space remains conservative.
Only the two measured stage3 k3 rows are admitted in the inactive local draft:

- fixed stage0: `conv k1 e1`;
- stage1: `mbconv k3 e3/e6`;
- stage2: `mbconv k3 e3`;
- stage3: `mbconv k3 e3`, `skip`, `denoise k3 e1`, or `edge k3 e1`.

After additional strict evidence is added, expand rows in this order:

1. Keep the current `denoise k3 e1` and `edge k3 e1` at stage3 as the only
   admitted sonar rows.
2. Stage1/stage2 stride-2 sonar rows only after dedicated route evidence.
3. `k5` rows only after separate timing-clean strict LUT evidence.
4. `mixconv` only in a separate ablation lane after its search/LUT kernel-size
   semantics are fixed and measured.

The hardware gate policy remains unchanged: actual Vivado DSP must be `<=700`,
`physical_risk` and `early_expand_pressure` must not directly enter the main
reward, and final claimability still requires full route gate PASS.

## Remaining Blocking Items

- Server Phase0 v4 RL300 has not been launched and requires an explicit manual
  decision.
- Stage1/stage2 sonar rows remain blocked by missing strict LUT/status and
  low-DSP route evidence.
- k5 sonar rows remain blocked because the current denoise/edge template line is
  k3-only and has no timing-clean strict evidence.
- `mixconv` remains blocked until its kernel-size semantics are fixed, frozen,
  and measured.
- Full claimability still requires a later full-network route gate PASS, actual
  Vivado DSP `<=700`, and only then COM5 measurement.

## Minimal Stage3 k3 Pilot Implementation

Implemented a scoped pilot config for the first admissible sonar rows:

- config: `hls_lut_builder/configs/candidate_kernels_phase0_v4_sonar_stage3_k3_pilot.yaml`;
- buffered templates:
  - `hls_lut_builder/templates/denoise_buffered_stage3_k3.cpp.tmpl`;
  - `hls_lut_builder/templates/edge_buffered_stage3_k3.cpp.tmpl`;
- cases:
  - `denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns`;
  - `edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns`;
- shape: `28:32->28:32 s1`, `kernel=3`, `expand=1`;
- output root: `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/`;
- HLS physical project dirs are shortened to `p/d_s3k3` and `p/e_s3k3`
  while preserving the formal case names above, to avoid Vitis path-length
  failures;
- no canonical `nas_board_lut_strict_current84_arch84` files were modified.

Local validation performed:

- generated the two isolated HLS project directories with `gen_project.py`;
- ran denoise-only `run_synthesis.py --downstream-check --dry-run`, producing
  `denoise_hls_synth_summary_dry_run.json`;
- extended `scripts/audit_phase0_v4_sonar_stage3_k3_pilot_timing.py` to gate
  HLS status, HLS log `HLS 200-885`, maximum observed II, downstream
  `post_route_setup_WNS_ns >= 0`, and downstream
  `post_route_Fmax_est >= 199.9`;
- changed buffered preload from nested loops to flat single-read loops after an
  initial attempt showed Vitis auto-pipelining the preload loops and re-emitting
  `HLS 200-885` on BRAM `weights`;
- added optional `workspace.case_dir_names` support in
  `hls_lut_builder/scripts/common.py`, used only by this pilot config, so the
  official case names and op specs remain unchanged while physical HLS dirs are
  short enough for Vitis RTL export;
- reran the real denoise operator-level HLS/OOC pilot:
  - HLS status: `success`;
  - HLS report:
    `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/d_s3k3/project/solution1/syn/report/denoise_kernel_csynth.xml`;
  - HLS `HLS 200-885` count at final gate: `10`, now from line-buffer
    scheduling rather than BRAM `weights` preload;
  - maximum observed II at final gate: `8`, below the `>=64` hard block
    threshold;
  - downstream OOC: manually stopped at `route_design` after the log stalled
    with large failed-net count and negative timing; no post-route report was
    produced, so downstream CSV has `0` parsed rows;
  - blocker evidence:
    `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/denoise_shortcasedir_downstream_manual_stop_reason.json`,
    `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/d_s3k3/logs/vitis_hls.log`,
    `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/d_s3k3/vivado_downstream/vivado_downstream.log`,
    and
    `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/sonar_stage3_k3_timing_gate.json`;
- edge was not run, because the sequential gate requires denoise HLS and
  downstream timing to pass first.

This buffered implementation record is superseded by the low-DSP recovery
record below. It remains useful as evidence that whole-pixel pipelining is not
an acceptable sonar-op implementation style for this board target.

## Low-DSP Recovery Update

Implemented a serial low-DSP recovery profile while preserving the two formal
case names and op specs:

- config:
  `hls_lut_builder/configs/candidate_kernels_phase0_v4_sonar_stage3_k3_pilot.yaml`;
- serial templates:
  - `hls_lut_builder/templates/denoise_serial_lowdsp_stage3_k3.cpp.tmpl`;
  - `hls_lut_builder/templates/edge_serial_lowdsp_stage3_k3.cpp.tmpl`;
- implementation profile: `input_parallelism=1`, `output_parallelism=1`,
  `unroll_factor=1`, `array_partition_factor=1`, `pipeline_ii=8`;
- timing/log gate:
  `scripts/audit_phase0_v4_sonar_stage3_k3_pilot_timing.py`;
- isolated evidence root:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/`.

Current timing gate result:

| case | HLS | HLS DSP | max II | downstream | post-route WNS | post-route Fmax | post-route DSP | status |
|---|---|---:|---:|---|---:|---:|---:|---|
| `denoise_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns` | success | 4 | 8 | success | 1.238 ns | 265.816 MHz | 4 | PASS |
| `edge_sonar_stage3_28_32_32_s1_baseline_pi1_po1_u1_main_5ns` | success | 4 | 8 | success | 1.246 ns | 266.383 MHz | 4 | PASS |

Important evidence:

- denoise HLS report:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/d_s3k3/project/solution1/syn/report/denoise_kernel_csynth.xml`;
- denoise downstream status:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/d_s3k3/vivado_downstream_status.json`;
- edge HLS report:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/e_s3k3/project/solution1/syn/report/edge_kernel_csynth.xml`;
- edge downstream status:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/p/e_s3k3/vivado_downstream_status.json`;
- final gate:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/sonar_stage3_k3_timing_gate.json`.

The earlier edge downstream failure was resolved by rerunning only edge
operator-level downstream with a clean Vivado runtime environment and short
isolated work root. No denoise rerun, training, COM5, or full-network route was
performed during that recovery.

The pilot formal LUT/status and merged draft LUT/status were generated only
after both rows passed the timing/log gate. The merged status normalizes the
two deployable `measured_implementation` pilot rows to `measured` for runtime
strict-LUT compatibility while preserving the original status per row.

## Sonar-Aware Additions Roadmap

The near-term sonar-aware search-space rule is conservative: add only
domain-aware operators with strict LUT/status and route evidence. Software-only
sonar preprocessing and evaluation improvements can proceed independently.

Phase0-admissible candidates after evidence:

- `denoise k3 e1` and `edge k3 e1` at stage3 only, because they are now proven
  HLS-low-DSP and denoise is post-route deployable;
- a small `sonar_stem` lane only after exact LUT rows exist, prioritizing
  `conv3`, `conv5`, `dwconv3_pw`, and fixed edge/DoG-like filters followed by
  `1x1` projection;
- conservative early stride policy and low-level skip preservation, expressed
  as search constraints rather than new hardware primitives.

Software/training additions that do not need FPGA operator admission:

- single-channel intensity normalization and log compression;
- CLAHE or contrast enhancement if clearly reported as CPU preprocessing;
- speckle noise, brightness, blur, and low-contrast augmentation;
- robustness tests across noise level, scene/batch, target distance, and
  acquisition split.

Phase1 or separate ablation only:

- dilation, because it has no current strict HLS/LUT coverage;
- SE/ECA/spatial attention, because pooling and channel scaling need separate
  LUT and route evidence;
- `mixconv`, after kernel-size semantics are fixed and frozen;
- `k5` sonar rows, after separate HLS and post-route measurement.

Reporting additions for sonar classification:

- keep `macro_f1` as the primary classification metric;
- add per-class Precision/Recall/F1, balanced accuracy, confusion matrix, and
  error-case visualization;
- audit data split by scene, mission, target instance, or acquisition batch to
  avoid adjacent-frame leakage;
- report latency with an explicit statement of whether CPU preprocessing is
  included.

## Diversity Dry-Run

The initial local diversity dry-run was executed on the v3 low-DSP guardrail
space only:

- command: `python scripts/audit_phase0_search_diversity.py --config configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v3_lowdsp_av7k325.yaml --sample-count 64 --min-unique 3 --output results/phase0_v4_sonar_op_precheck/diversity_audit_v3_lowdsp_guardrail.json`
- result: `PASS`;
- `unique_encoding_count=4`;
- `feasible_unique_encoding_count=4`;
- no server, Vivado, or COM measurement was run.

After the stage3 k3 sonar admission materials were complete, a second
diversity dry-run was executed on the inactive v4 draft config:

- command: `python scripts/audit_phase0_search_diversity.py --config configs/search/nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml --sample-count 64 --min-unique 3 --output hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/diversity_audit_v4_sonar_stage3_k3_lowdsp_draft.json`
- result: `PASS`;
- `unique_encoding_count=8`;
- `feasible_unique_encoding_count=8`;
- no server, Vivado, or COM measurement was run.

## Launch-Readiness Package

The next-step package is generated by:

```bash
python scripts/prepare_phase0_v4_sonar_launch_readiness.py --rerun-local-audits --require-ready
```

Latest result:

- status: `READY_FOR_MANUAL_SERVER_LAUNCH`;
- local precheck rerun return code: `0`;
- diversity rerun return code: `0`;
- server/training/Vivado/COM5 flags: all `false`;
- JSON:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/launch_readiness/phase0_v4_sonar_stage3_k3_launch_readiness.json`;
- Markdown:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/launch_readiness/phase0_v4_sonar_stage3_k3_launch_readiness.md`.

The readiness package includes copyable commands for a later manual RL300
launch decision and for post-search route-gate planning. It intentionally does
not execute those commands.

Important route-gate detail: Phase0 v4 sonar route-gate planning must layer
both low-DSP catalogs:

```bash
--lowdsp-override-catalog hls_lut_builder/board_harness/configs/phase0_v3_lowdsp_override_catalog.yaml \
--lowdsp-override-catalog hls_lut_builder/board_harness/configs/phase0_v4_sonar_stage3_lowdsp_override_catalog.yaml
```

`scripts/prepare_phase0_full_route_gate.py` now supports repeated
`--lowdsp-override-catalog` arguments for this exact use case. Existing v3
behavior is unchanged when no explicit catalog is supplied.

## Pipeline Poller

The next local status poll is:

```bash
python scripts/poll_phase0_v4_sonar_pipeline.py
```

Latest result:

- server/training/Vivado/COM5 flags: all `false`;
- readiness status: `READY_FOR_MANUAL_SERVER_LAUNCH`;
- local server-search artifacts are not present yet:
  `summary=false`, `pareto_selection=false`,
  `pareto_ranked_candidates=false`, `lut_stats=false`;
- next action:
  `manual_server_rl300_launch_or_fetch_results`.

The poller writes:

- JSON:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/pipeline_poll/phase0_v4_sonar_stage3_k3_pipeline_poll_latest.json`;
- Markdown:
  `hls_lut_builder/results/phase0_v4_sonar_stage3_k3_pilot/pipeline_poll/phase0_v4_sonar_stage3_k3_pipeline_poll_latest.md`.

It can optionally refresh readiness with `--refresh-readiness`. It can also run
only the local route-gate prepare step with `--prepare-route-gate` after search
artifacts exist. It never launches server RL300, Vivado implementation, or
COM5 measurement.
