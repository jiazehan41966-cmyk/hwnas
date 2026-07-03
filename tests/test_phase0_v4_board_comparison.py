from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_phase0_v4_vs_v3_board_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase0_v4_board_comparison", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_uses_current_evidence_classes_without_stale_candidate_claims():
    module = load_module()
    payload = {
        "summary": {
            "v4_pareto_count": 3,
            "v4_route_clean_count": 2,
            "v4_board_claimable_count": 1,
            "v4_route_fail_count": 1,
        },
        "findings": [],
        "rows": [
            {
                "phase": "v4",
                "arch_id": "board_arch",
                "role": "screen",
                "stage3_op": "edge k3 e1",
                "evidence_class": "board-claimable",
                "macro_f1": 0.6,
                "top1": 0.8,
                "com5_latency_ms": 10.0,
                "com5_cycles": 100,
                "actual_wns_ns": 0.1,
                "actual_dsp": 10,
                "actual_lut": 20,
                "measurement_runs": 5,
            },
            {
                "phase": "v4",
                "arch_id": "route_arch",
                "role": "screen",
                "stage3_op": "skip k1 e1",
                "evidence_class": "route-clean",
                "macro_f1": 0.5,
                "top1": 0.7,
                "com5_latency_ms": "",
                "com5_cycles": "",
                "actual_wns_ns": 0.2,
                "actual_dsp": 8,
                "actual_lut": 15,
                "measurement_runs": "",
            },
            {
                "phase": "v4",
                "arch_id": "failed_arch",
                "role": "screen",
                "stage3_op": "denoise k3 e1",
                "evidence_class": "route-fail",
                "macro_f1": 0.55,
                "top1": 0.72,
                "com5_latency_ms": "",
                "com5_cycles": "",
                "actual_wns_ns": -1.0,
                "actual_dsp": 900,
                "actual_lut": 30,
                "measurement_runs": "",
            },
        ],
    }

    markdown = module.render_markdown(payload)

    assert "Current manifest board-claimable rows: `board_arch`." in markdown
    assert "Current manifest route-clean-only rows: `route_arch`." in markdown
    assert "Current manifest route-fail rows: `failed_arch`" in markdown
    assert "macro_f1/top1 columns are search-proxy metrics" in markdown
    assert "COM5 slots were not spent" not in markdown
