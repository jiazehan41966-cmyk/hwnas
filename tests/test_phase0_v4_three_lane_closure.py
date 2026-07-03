from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "phase0_v4_three_lane_closure.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phase0_v4_three_lane_closure", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_boundary_rows_keep_missing_v4_layers_explicit():
    mod = load_module()
    manifest = {
        "pareto_route_screen_candidates": [
            {
                "arch_id": "rl_arch_154",
                "role": "pareto_route_screen",
                "stage3_op": "edge k3 e1",
                "search_proxy": {"macro_f1": 0.1, "top1": 0.2},
                "vivado_actual": {"status": "route-clean", "wns_ns": 0.1, "actual_dsp": 528},
                "com5_board": {"status": "board-claimable", "status_code": 0, "latency_ms": 39.0},
            }
        ],
        "v3_board_claimable_baseline": [],
    }
    retrain_payload = {
        "rows": [
            {
                "phase": "v4",
                "arch_id": "rl_arch_154",
                "retrain_status": "not run",
            }
        ]
    }

    rows = mod.build_evidence_boundary_rows(
        manifest=manifest,
        route_index={},
        retrain_payload=retrain_payload,
        v3_retrain_index={},
    )

    assert rows[0]["macro_f1_source"] == "search proxy eval10"
    assert rows[0]["board_val_accuracy_status"] == "not run"
    assert rows[0]["power_energy_status"] == "not measured"
    assert "COM5 deterministic harness" in rows[0]["claim_boundary"]
    for field in mod.EVIDENCE_FIELDS:
        assert field in rows[0]


def test_ablation_configs_change_only_stage3_admission(tmp_path):
    mod = load_module()
    base_config = REPO_ROOT / "configs" / "search" / (
        "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_"
        "phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml"
    )
    output_root = tmp_path / "ablation_results"
    config_dir = tmp_path / "ablation_configs"

    payload = mod.build_ablation_artifacts(
        base_config_path=base_config,
        output_root=output_root,
        config_dir=config_dir,
        python_path=Path(".venv_cuda/Scripts/python.exe"),
    )

    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    generated = {
        row["variant"]: yaml.safe_load((REPO_ROOT / row["config"]).read_text(encoding="utf-8"))
        if not Path(row["config"]).is_absolute()
        else yaml.safe_load(Path(row["config"]).read_text(encoding="utf-8"))
        for row in payload["rows"]
    }
    expected_stage3_ops = {
        "no_sonar": ["mbconv", "skip"],
        "denoise_only": ["mbconv", "skip", "denoise"],
        "edge_only": ["mbconv", "skip", "edge"],
        "denoise_edge": ["mbconv", "skip", "denoise", "edge"],
    }
    for variant, cfg in generated.items():
        assert cfg["search_space"]["stage_block_choices"][:3] == base["search_space"]["stage_block_choices"][:3]
        stage3_ops = [row["op"] for row in cfg["search_space"]["stage_block_choices"][3]]
        assert stage3_ops == expected_stage3_ops[variant]
        assert cfg["search"]["num_candidates"] == 300
        assert cfg["search"]["eval_epochs"] == 10


def test_ablation_summary_marks_partial_search_state_not_comparison_ready(tmp_path):
    mod = load_module()
    base_config = REPO_ROOT / "configs" / "search" / (
        "nas_board_lut_strict_current84_arch84_nksid_full_rl300_eval10_cuda_physical_"
        "phase0_v4_sonar_stage3_k3_lowdsp_draft_av7k325.yaml"
    )
    output_root = tmp_path / "ablation_results"
    config_dir = tmp_path / "ablation_configs"
    run_dir = output_root / "phase0_v4_sonar_ablation_no_sonar_rl300_eval10_seed42"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "search_state.json").write_text(
        json.dumps(
            {
                "total_evaluated": 3,
                "feasible": 3,
                "best_candidate": {
                    "arch_id": "rl_arch_2",
                    "metrics": {
                        "macro_f1": 0.596884,
                        "top1": 0.771154,
                        "latency_ms": 6.851485,
                        "lut": 92362,
                        "dsp": 900,
                        "bram": 135,
                    },
                },
                "extra": {"episode": 2},
            }
        ),
        encoding="utf-8",
    )

    payload = mod.build_ablation_artifacts(
        base_config_path=base_config,
        output_root=output_root,
        config_dir=config_dir,
        python_path=Path(".venv_cuda/Scripts/python.exe"),
    )

    row = next(item for item in payload["rows"] if item["variant"] == "no_sonar")
    assert row["status"] == "partial_started"
    assert row["state_source"] == "checkpoints/search_state.json"
    assert row["total_evaluated"] == 3
    assert row["latest_episode"] == 2
    assert row["comparison_ready"] is False
    assert row["best_arch_id"] == "rl_arch_2"
    assert row["macro_f1"] == 0.596884


def test_measurement_audit_upgrades_route_clean_candidate(tmp_path):
    mod = load_module()
    bitstream = tmp_path / "candidate.bit"
    bitstream.write_bytes(b"deterministic-bitstream")
    expected_sha = hashlib.sha256(b"deterministic-bitstream").hexdigest()
    measurement = tmp_path / "candidate.measurement.json"
    measurement.write_text(
        json.dumps(
            {
                "bitstream": str(bitstream),
                "frame": {"status_code": 0, "cycles": 123, "checksum": 456},
                "latency_ms": 1.23,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "stability_runs.json").write_text(
        json.dumps({"run_count": 5, "status_codes": ["0"], "stable": True}),
        encoding="utf-8",
    )
    manifest = {
        "pareto_route_screen_candidates": [
            {
                "arch_id": "rl_arch_60",
                "role": "pareto_route_screen",
                "stage3_op": "skip k1 e1",
                "search_proxy": {"macro_f1": 0.1, "top1": 0.2},
                "vivado_actual": {
                    "status": "route-clean",
                    "wns_ns": 0.1,
                    "actual_dsp": 524,
                    "bitstream_sha256": expected_sha,
                },
                "com5_board": {"status": "ready_for_com5"},
            }
        ],
        "v3_board_claimable_baseline": [],
    }

    rows = mod.build_evidence_boundary_rows(
        manifest=manifest,
        route_index={"rl_arch_60": {"measurement_json": str(measurement)}},
        retrain_payload={"rows": []},
        v3_retrain_index={},
    )

    assert rows[0]["com5_status"] == "board-claimable"
    assert rows[0]["com5_status_code"] == 0
    assert rows[0]["com5_latency_ms"] == 1.23
    assert rows[0]["measurement_runs"] == 5


def test_retrain_metric_reader_uses_best_eval_macro_f1_without_accuracy_alias():
    mod = load_module()
    summary = {
        "metrics": {
            "best_val_acc": 0.9,
            "best_eval": {
                "macro_f1": 0.8123,
                "top1": 0.9012,
                "weighted_f1": 0.9234,
            },
        }
    }

    assert mod.metric_from_retrain_summary(summary, "macro_f1") == 0.8123
    assert mod.metric_from_retrain_summary(summary, "top1") == 0.9012
    assert mod.metric_from_retrain_summary(summary, "weighted_f1") == 0.9234
