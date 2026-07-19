# Canonical Search Configs After Fair Macro Rerun

This note records the historical fair scratch-only macro backbone configs from
`2026-04-17` and the semantic-safe default adopted on `2026-07-03`.

Macro anchor source:

- [selected_backbone_pool.json](../results/backbone_nksid_av7k325_fair_scratch_seed42_gpu_20260417/results/selected_backbone_pool.json)

Resolved anchors:

| Profile | Source role | Source backbone | Source arch id |
| --- | --- | --- | --- |
| `mobile_anchor` | `search_anchor` | `mobilenet_v2` | `mobilenet_v2_scratch` |
| `accuracy_biased` | `accuracy_anchor` | `mobilenet_v2` | `mobilenet_v2_scratch` |
| `lightweight_sonar` | `lightweight_anchor` | `shufflenet_v2` | `shufflenet_v2_scratch` |

Canonical config entry points:

| Purpose | Config |
| --- | --- |
| Current semantic-safe default | [nksid_fpga_search_mobile_anchor_av7k325.yaml](../configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml) |
| Generic accuracy-biased search space | [nksid_fpga_search_accuracy_biased_av7k325.yaml](../configs/search/nksid_fpga_search_accuracy_biased_av7k325.yaml) |
| Generic lightweight search space | [nksid_fpga_search_lightweight_sonar_av7k325.yaml](../configs/search/nksid_fpga_search_lightweight_sonar_av7k325.yaml) |
| Random baseline on the new mobile anchor | [nksid_random_baseline_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_random_baseline_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| RL search on the new mobile anchor | [nksid_rl_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_rl_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| Three-objective RL comparison protocol | [nksid_rl_pareto3_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_rl_pareto3_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| Multi-objective aging-evolution comparison protocol | [nksid_aging_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_aging_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| ProxylessNAS on the new mobile anchor | [nksid_proxyless_mobile_anchor_mobilenet_v2_av7k325.yaml](../configs/search/nksid_proxyless_mobile_anchor_mobilenet_v2_av7k325.yaml) |

Notes:

- The older `*_shufflenet_*` mobile-anchor experiment configs are preserved for
  historical reproducibility only.
- The older generic `nksid_fpga_search*.yaml` entry points have been moved under
  `configs/search/legacy/` so the top-level config directory only exposes the
  current formal paths.
- New experiments should start from the semantic-safe default. The dated
  random/RL/Proxyless comparison configs are historical protocols until they
  are explicitly refreshed.
- The RL/aging comparison configs are preregistered execution inputs, not
  completed results. G3 remains `FROZEN`; both requested methods require Gate 0,
  G2, G4, any applicable G5 admission, and explicit manual Stage-3 approval.
- The default config declares `mbconv`, `denoise`, `edge`, and `skip`, then
  loads `hls_lut_builder/configs/operator_manifest_semantic_safe.yaml`.
  Runtime policy filtering keeps `mbconv` and `skip`; `denoise` and `edge`
  remain paused until matching computation, weight export, and numeric parity
  are verified.
- The XC7K325T physical capacity is `203800` LUTs. Historical configs that
  explicitly contain `50950` preserve the old slice-count constraint and must
  not be silently relabeled as the corrected protocol.
- Search-time power/energy estimates are not measured board objectives. The
  current default disables the power hard cap and energy reward.
- `dw_pw_conv` has been removed from the formal MobileNetV2 search profiles.
- `mixconv` remains an optional ablation operator rather than part of the main
  formal MobileNetV2 search space.
- The canonical `mobile_anchor` profile is a board-feasible, MobileNetV2-inspired
  compressed space aligned to the AV7K325 total-resource estimator rather than
  the original macro backbone widths.
