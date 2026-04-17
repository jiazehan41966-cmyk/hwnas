# Canonical Search Configs After Fair Macro Rerun

This note records the canonical search configs after the fair scratch-only macro
backbone rerun on `2026-04-17`.

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
| Generic mobile-anchor search space | [nksid_fpga_search_mobile_anchor_av7k325.yaml](../configs/search/nksid_fpga_search_mobile_anchor_av7k325.yaml) |
| Generic accuracy-biased search space | [nksid_fpga_search_accuracy_biased_av7k325.yaml](../configs/search/nksid_fpga_search_accuracy_biased_av7k325.yaml) |
| Generic lightweight search space | [nksid_fpga_search_lightweight_sonar_av7k325.yaml](../configs/search/nksid_fpga_search_lightweight_sonar_av7k325.yaml) |
| Random baseline on the new mobile anchor | [nksid_random_baseline_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_random_baseline_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| RL search on the new mobile anchor | [nksid_rl_mobile_anchor_mobilenet_v2_av7k325_200.yaml](../configs/search/nksid_rl_mobile_anchor_mobilenet_v2_av7k325_200.yaml) |
| ProxylessNAS on the new mobile anchor | [nksid_proxyless_mobile_anchor_mobilenet_v2_av7k325.yaml](../configs/search/nksid_proxyless_mobile_anchor_mobilenet_v2_av7k325.yaml) |

Notes:

- The older `*_shufflenet_*` mobile-anchor experiment configs are preserved for
  historical reproducibility only.
- New formal mobile-anchor experiments should use the `*_mobilenet_v2_*`
  configs above.
