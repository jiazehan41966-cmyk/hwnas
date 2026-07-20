from pathlib import Path
import sys
import json

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_aging_vs_rl_benchmark as benchmark


def test_parse_nvidia_compute_processes_ignores_malformed_rows() -> None:
    parsed = benchmark.parse_nvidia_compute_processes(
        "123, C:\\Python\\python.exe\nmalformed\nabc, ignored.exe\n456, trainer\n"
    )
    assert parsed == [
        {"pid": 123, "process_name": "C:\\Python\\python.exe"},
        {"pid": 456, "process_name": "trainer"},
    ]


def test_formal_cuda_run_rejects_external_python_process(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "find_external_python_gpu_processes",
        lambda: [{"pid": 987, "process_name": "python.exe"}],
    )
    with pytest.raises(RuntimeError, match="exclusive Python/CUDA"):
        benchmark.assert_gpu_is_exclusive(device="cuda", allow_shared_gpu=False)


def test_non_cuda_or_explicit_shared_mode_skips_exclusivity_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "find_external_python_gpu_processes",
        lambda: pytest.fail("probe should not run"),
    )
    benchmark.assert_gpu_is_exclusive(device="cpu", allow_shared_gpu=False)
    benchmark.assert_gpu_is_exclusive(device="cuda", allow_shared_gpu=True)


def test_counterbalanced_method_order_alternates_across_seeds() -> None:
    assert benchmark.methods_for_seed(0, "counterbalanced") == (
        "rl",
        "aging_evolution",
    )
    assert benchmark.methods_for_seed(1, "counterbalanced") == (
        "aging_evolution",
        "rl",
    )


def test_benchmark_manifest_is_machine_readable(tmp_path: Path) -> None:
    path = tmp_path / "comparison" / "benchmark_manifest.json"
    benchmark.write_manifest(
        path,
        {"schema_version": 1, "status": "running", "seeds": [42, 43]},
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "running",
        "seeds": [42, 43],
    }


def test_cuda_child_python_prefers_project_cuda_environment() -> None:
    expected = PROJECT_ROOT / ".venv_cuda" / "Scripts" / "python.exe"
    if not expected.exists():
        pytest.skip("project CUDA environment is not installed")
    assert benchmark.resolve_python_executable(None, "cuda") == expected


def test_source_protocol_match_ignores_method_specific_search_blocks() -> None:
    common = {
        "dataset": {"name": "dummy", "image_size": 32},
        "search_space": {"op_choices": ["conv", "mbconv"]},
        "constraints": {"max_lut": 1000},
        "hardware": {"board": "test"},
        "training": {"batch_size": 8},
    }
    rl = {
        **common,
        "search": {
            "method": "rl",
            "controller_hidden": 16,
            "selection_metric": "macro_f1",
            "pareto": {"objectives": ["macro_f1", "latency_ms"]},
        },
    }
    aging = {
        **common,
        "search": {
            "method": "aging_evolution",
            "aging": {"population_size": 4},
            "selection_metric": "macro_f1",
            "pareto": {"objectives": ["macro_f1", "latency_ms"]},
        },
    }
    benchmark.assert_matched_source_protocols({"rl": rl, "aging_evolution": aging})


def test_source_protocol_mismatch_is_rejected_before_compute() -> None:
    rl = {
        "dataset": {"name": "dummy", "image_size": 32},
        "search": {"selection_metric": "macro_f1"},
    }
    aging = {
        "dataset": {"name": "dummy", "image_size": 64},
        "search": {"selection_metric": "macro_f1"},
    }
    with pytest.raises(RuntimeError, match="source protocols differ"):
        benchmark.assert_matched_source_protocols(
            {"rl": rl, "aging_evolution": aging}
        )


def test_gate_statuses_report_each_blocked_method_before_compute() -> None:
    statuses = {
        "rl": {
            "may_start_new_claimable_search": False,
            "gates": {"g2_pass": False, "stage3_method_selected": False},
        },
        "aging_evolution": {
            "may_start_new_claimable_search": False,
            "gates": {"g2_pass": False, "stage3_method_selected": False},
        },
    }
    with pytest.raises(RuntimeError, match="Failed gates by method") as exc_info:
        benchmark.assert_search_gate_statuses(statuses)
    assert "aging_evolution" in str(exc_info.value)
    assert "rl" in str(exc_info.value)


def test_dummy_smoke_gate_status_is_permitted() -> None:
    benchmark.assert_search_gate_statuses(
        {
            "rl": {
                "may_start_new_claimable_search": False,
                "dummy_smoke_permitted": True,
                "gates": {"g2_pass": False},
            }
        }
    )


def test_gate_preflight_resolves_method_from_comparison_key(monkeypatch) -> None:
    requested = []

    def fake_gate(_root, *, dataset_name, config):
        requested.append((dataset_name, config["search"]["method"]))
        return {"dummy_smoke_permitted": True}

    monkeypatch.setattr(benchmark, "require_stage3_search_gate", fake_gate)
    shared = {"dataset": {"name": "dummy"}, "search": {"method": "rl"}}
    benchmark.evaluate_search_gates({"rl": shared, "aging_evolution": shared})
    assert requested == [("dummy", "rl"), ("dummy", "aging_evolution")]


def test_full_job_efficiency_accumulates_resume_segments() -> None:
    first = benchmark.merge_job_efficiency(
        None,
        {
            "device": "cuda",
            "cuda_used": True,
            "exclusive_gpu_required": True,
            "wall_clock_seconds": 10.0,
            "gpu_reserved_wall_seconds": 10.0,
            "returncode": 0,
        },
    )
    merged = benchmark.merge_job_efficiency(
        first,
        {
            "device": "cuda",
            "cuda_used": True,
            "exclusive_gpu_required": True,
            "wall_clock_seconds": 5.0,
            "gpu_reserved_wall_seconds": 5.0,
            "returncode": 0,
        },
    )
    assert merged["segment_count"] == 2
    assert merged["wall_clock_seconds"] == 15.0
    assert merged["gpu_reserved_wall_seconds"] == 15.0
    assert merged["gpu_reserved_hours"] == pytest.approx(15.0 / 3600.0)


def test_resume_skips_only_completed_run_with_primary_timing(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    results = run_root / "results"
    results.mkdir(parents=True)
    (results / "summary.json").write_text(
        json.dumps({"status": "completed", "total_evaluated": 4}),
        encoding="utf-8",
    )
    assert benchmark.classify_resume_run(run_root, 4) == "completed_missing_job_timing"
    (results / "job_efficiency.json").write_text(
        json.dumps({"gpu_reserved_hours": 1.0}), encoding="utf-8"
    )
    assert benchmark.classify_resume_run(run_root, 4) == "completed_with_timing"
    assert benchmark.classify_resume_run(run_root, 5) == "needs_run"
