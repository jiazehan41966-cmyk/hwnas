#!/usr/bin/env python3
"""Run matched RL and aging-evolution searches, then package comparison evidence.

Runs are deliberately sequential so the two methods do not contend for the
same GPU.  Existing run directories are refused unless ``--resume`` is given.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwnas_fpga.research_gates import require_stage3_search_gate, stage3_gate_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-config",
        default="configs/search/nksid_rl_pareto3_mobile_anchor_mobilenet_v2_av7k325_200.yaml",
    )
    parser.add_argument(
        "--aging-config",
        default="configs/search/nksid_aging_mobile_anchor_mobilenet_v2_av7k325_200.yaml",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable for child searches; CUDA defaults to project .venv_cuda when present.",
    )
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-prefix", default="aging_vs_rl_equal_budget")
    parser.add_argument("--comparison-dir", default=None)
    parser.add_argument(
        "--method-order",
        choices=("counterbalanced", "rl-first", "aging-first"),
        default="counterbalanced",
        help="Per-seed execution order; counterbalanced alternates AB/BA across seeds.",
    )
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Allow another Python CUDA process to coexist (invalid for formal GPU-hour comparison).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_from_project(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_python_executable(requested: str | None, device: str) -> Path:
    if requested:
        executable = Path(requested).expanduser().resolve()
    elif str(device).lower().startswith("cuda"):
        candidates = (
            PROJECT_ROOT / ".venv_cuda" / "Scripts" / "python.exe",
            PROJECT_ROOT / ".venv_cuda" / "bin" / "python",
        )
        executable = next((path for path in candidates if path.exists()), Path(sys.executable))
    else:
        executable = Path(sys.executable).resolve()
    if not executable.exists():
        raise FileNotFoundError(f"Python executable not found: {executable}")
    return executable


def parse_nvidia_compute_processes(output: str) -> list[dict[str, object]]:
    """Parse the stable two-column nvidia-smi compute-process query."""

    processes: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        processes.append({"pid": pid, "process_name": fields[1]})
    return processes


def find_external_python_gpu_processes() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot verify exclusive GPU use with nvidia-smi; use --allow-shared-gpu "
            "only for explicitly non-formal runs."
        ) from exc
    return [
        process
        for process in parse_nvidia_compute_processes(result.stdout)
        if int(process["pid"]) != os.getpid()
        and "python" in str(process["process_name"]).lower()
    ]


def assert_gpu_is_exclusive(*, device: str, allow_shared_gpu: bool) -> None:
    if not str(device).lower().startswith("cuda") or allow_shared_gpu:
        return
    conflicting = find_external_python_gpu_processes()
    if conflicting:
        details = ", ".join(
            f"pid={row['pid']} ({row['process_name']})" for row in conflicting
        )
        raise RuntimeError(
            "Formal GPU-hour comparison requires exclusive Python/CUDA use; "
            f"found {details}. Wait for those jobs to finish or use --allow-shared-gpu "
            "only for a non-formal smoke run."
        )


def methods_for_seed(seed_index: int, method_order: str) -> tuple[str, str]:
    if method_order == "rl-first":
        return "rl", "aging_evolution"
    if method_order == "aging-first":
        return "aging_evolution", "rl"
    if method_order != "counterbalanced":
        raise ValueError(f"Unsupported method order: {method_order}")
    return (
        ("rl", "aging_evolution")
        if int(seed_index) % 2 == 0
        else ("aging_evolution", "rl")
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_job_efficiency(
    previous: dict[str, object] | None,
    segment: dict[str, object],
) -> dict[str, object]:
    """Accumulate full child-process timing across resumed benchmark jobs."""

    previous_segments = []
    if previous:
        existing = previous.get("segments")
        previous_segments = list(existing) if isinstance(existing, list) else []
    segments = [*previous_segments, dict(segment)]
    wall_seconds = sum(float(row.get("wall_clock_seconds") or 0.0) for row in segments)
    gpu_seconds = sum(
        float(row.get("gpu_reserved_wall_seconds") or 0.0) for row in segments
    )
    return {
        "schema_version": 1,
        "measurement_scope": "cumulative_full_child_process",
        "device": segment.get("device"),
        "cuda_used": any(bool(row.get("cuda_used")) for row in segments),
        "exclusive_gpu_required": bool(segment.get("exclusive_gpu_required")),
        "wall_clock_seconds": wall_seconds,
        "gpu_reserved_wall_seconds": gpu_seconds,
        "gpu_reserved_hours": gpu_seconds / 3600.0,
        "segment_count": len(segments),
        "segments": segments,
        "definition": (
            "Parent-observed child-process wall time, including interpreter startup, "
            "data/model/search setup, the search call, and artifact finalization. On an "
            "exclusive CUDA run this is the primary allocated GPU-slot cost."
        ),
    }


def write_job_efficiency(
    run_root: Path,
    segment: dict[str, object],
    *,
    resume: bool,
) -> Path:
    path = run_root / "results" / "job_efficiency.json"
    previous = None
    if resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            previous = payload
    merged = merge_job_efficiency(previous, segment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def classify_resume_run(run_root: Path, required_budget: int) -> str:
    """Classify whether a resumed job needs work without fabricating timing."""

    summary_path = run_root / "results" / "summary.json"
    if not summary_path.exists():
        return "needs_run"
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected summary mapping: {summary_path}")
    completed = payload.get("status") == "completed"
    evaluated = int(payload.get("total_evaluated") or 0)
    if not completed or evaluated < int(required_budget):
        return "needs_run"
    timing_path = run_root / "results" / "job_efficiency.json"
    if timing_path.exists():
        return "completed_with_timing"
    return "completed_missing_job_timing"


def read_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def comparison_protocol_payload(config: dict[str, object]) -> dict[str, object]:
    search = dict(config.get("search") or {})
    return {
        "anchor_source": config.get("anchor_source"),
        "dataset": config.get("dataset"),
        "search_space": config.get("search_space"),
        "constraints": config.get("constraints"),
        "hardware": config.get("hardware"),
        "training_batch_size": (config.get("training") or {}).get("batch_size"),
        "selection_metric": search.get("selection_metric", "macro_f1"),
        "objective_weights": search.get("objective_weights"),
        "pareto": search.get("pareto"),
        "robustness": search.get("robustness"),
        "eval_early_stopping_patience": search.get("eval_early_stopping_patience"),
    }


def assert_matched_source_protocols(configs: dict[str, dict[str, object]]) -> None:
    payloads = {
        method: comparison_protocol_payload(config)
        for method, config in configs.items()
    }
    fingerprints = {
        method: json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for method, payload in payloads.items()
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError(
            "RL and aging source protocols differ before CLI budget/seed overrides; "
            f"refusing to spend compute. Protocols: {payloads}"
        )


def evaluate_search_gates(configs: dict[str, dict[str, object]]) -> dict[str, object]:
    statuses: dict[str, object] = {}
    for method, config in configs.items():
        resolved_config = dict(config)
        resolved_config["search"] = {
            **dict(config.get("search") or {}),
            "method": method,
        }
        dataset = dict(config.get("dataset") or {})
        dataset_name = str(dataset.get("name", "dummy"))
        statuses[method] = (
            require_stage3_search_gate(
                PROJECT_ROOT,
                dataset_name=dataset_name,
                config=resolved_config,
            )
            if dataset_name == "dummy"
            else stage3_gate_status(PROJECT_ROOT, config=config)
        )
    return statuses


def assert_search_gate_statuses(statuses: dict[str, object]) -> None:
    blocked: dict[str, list[str]] = {}
    for method, payload in statuses.items():
        status = dict(payload or {})
        if status.get("may_start_new_claimable_search") is True:
            continue
        if status.get("dummy_smoke_permitted") is True:
            continue
        blocked[method] = [
            str(name)
            for name, passed in dict(status.get("gates") or {}).items()
            if not passed
        ]
    if blocked:
        raise RuntimeError(
            "Stage 3 preflight rejected the formal search before compute. "
            f"Failed gates by method: {blocked}"
        )


def main() -> int:
    args = parse_args()
    seeds = args.seeds or [42]
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.train_epochs < 0:
        raise ValueError("--train-epochs must be non-negative")

    output_root = resolved_from_project(args.output_root)
    child_python = resolve_python_executable(args.python, args.device)
    run_roots: list[Path] = []
    commands: list[list[str]] = []
    jobs: list[dict[str, object]] = []
    methods = {
        "rl": resolved_from_project(args.rl_config),
        "aging_evolution": resolved_from_project(args.aging_config),
    }
    for config in methods.values():
        if not config.exists():
            raise FileNotFoundError(config)
    loaded_configs = {
        method: read_yaml_mapping(path) for method, path in methods.items()
    }
    for seed_index, seed in enumerate(seeds):
        ordered_methods = methods_for_seed(seed_index, args.method_order)
        for method in ordered_methods:
            config = methods[method]
            short_method = "aging" if method == "aging_evolution" else method
            run_name = f"{args.run_prefix}_{short_method}_seed{seed}"
            run_root = output_root / run_name
            if run_root.exists() and not args.resume:
                raise FileExistsError(
                    f"Refusing to reuse existing run without --resume: {run_root}"
                )
            command = [
                str(child_python),
                str(PROJECT_ROOT / "run_search.py"),
                "--config",
                str(config),
                "--search-method",
                method,
                "--num-candidates",
                str(args.budget),
                "--episodes",
                str(args.budget),
                "--train-epochs",
                str(args.train_epochs),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--output-dir",
                str(output_root),
                "--run-name",
                run_name,
            ]
            if args.resume:
                command.append("--resume")
            commands.append(command)
            run_roots.append(run_root)
            jobs.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "run_root": str(run_root),
                    "command": command,
                }
            )

    comparison_dir = (
        resolved_from_project(args.comparison_dir)
        if args.comparison_dir
        else output_root / f"{args.run_prefix}_comparison"
    )
    compare_command = [
        str(child_python),
        str(PROJECT_ROOT / "scripts" / "compare_search_methods.py"),
    ]
    for run_root in run_roots:
        compare_command.extend(("--run", str(run_root)))
    compare_command.extend(("--output-dir", str(comparison_dir)))

    if args.dry_run:
        for command in [*commands, compare_command]:
            print(subprocess.list2cmdline(command))
        return 0

    manifest_path = comparison_dir / "benchmark_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_timestamp(),
        "project_root": str(PROJECT_ROOT),
        "python_executable": str(child_python),
        "device": args.device,
        "exclusive_gpu_required": bool(
            str(args.device).lower().startswith("cuda") and not args.allow_shared_gpu
        ),
        "method_order": args.method_order,
        "seeds": [int(seed) for seed in seeds],
        "budget_per_method_seed": int(args.budget),
        "train_epochs_per_candidate": int(args.train_epochs),
        "resume": bool(args.resume),
        "jobs": jobs,
        "comparison_command": compare_command,
        "completed_jobs": 0,
        "preflight": {
            "matched_source_protocols": False,
            "search_gates_checked": False,
        },
    }
    write_manifest(manifest_path, manifest)
    try:
        assert_matched_source_protocols(loaded_configs)
        manifest["preflight"] = {
            "matched_source_protocols": True,
            "search_gates_checked": False,
        }
        manifest["updated_at"] = utc_timestamp()
        write_manifest(manifest_path, manifest)
        search_gate_statuses = evaluate_search_gates(loaded_configs)
        manifest["preflight"] = {
            "matched_source_protocols": True,
            "search_gates_checked": True,
            "search_gate_statuses": search_gate_statuses,
        }
        manifest["updated_at"] = utc_timestamp()
        write_manifest(manifest_path, manifest)
        assert_search_gate_statuses(search_gate_statuses)
        for index, (command, job) in enumerate(zip(commands, jobs), start=1):
            if args.resume:
                resume_state = classify_resume_run(
                    Path(str(job["run_root"])),
                    args.budget,
                )
                job["resume_state"] = resume_state
                if resume_state == "completed_missing_job_timing":
                    raise RuntimeError(
                        "A completed run lacks results/job_efficiency.json, so its primary "
                        f"full-job GPU cost cannot be reconstructed: {job['run_root']}. "
                        "Use a new run prefix for a formal comparison."
                    )
                if resume_state == "completed_with_timing":
                    job["status"] = "already_completed"
                    job["skipped_at"] = utc_timestamp()
                    manifest["completed_jobs"] = index
                    manifest["updated_at"] = utc_timestamp()
                    write_manifest(manifest_path, manifest)
                    continue
            assert_gpu_is_exclusive(
                device=args.device,
                allow_shared_gpu=args.allow_shared_gpu,
            )
            job["status"] = "running"
            job["started_at"] = utc_timestamp()
            manifest["updated_at"] = utc_timestamp()
            write_manifest(manifest_path, manifest)
            started = perf_counter()
            returncode: int | None = 0
            try:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            except BaseException as exc:
                raw_returncode = getattr(exc, "returncode", None)
                returncode = (
                    int(raw_returncode) if raw_returncode is not None else None
                )
                job["status"] = "failed"
                raise
            finally:
                elapsed = perf_counter() - started
                is_cuda = str(args.device).lower().startswith("cuda")
                segment = {
                    "started_at": job["started_at"],
                    "finished_at": utc_timestamp(),
                    "wall_clock_seconds": elapsed,
                    "gpu_reserved_wall_seconds": elapsed if is_cuda else 0.0,
                    "cuda_used": is_cuda,
                    "device": args.device,
                    "exclusive_gpu_required": bool(is_cuda and not args.allow_shared_gpu),
                    "returncode": returncode,
                }
                timing_path = write_job_efficiency(
                    Path(str(job["run_root"])),
                    segment,
                    resume=bool(args.resume),
                )
                job["finished_at"] = segment["finished_at"]
                job["job_efficiency"] = str(timing_path)
                job["wall_clock_seconds"] = elapsed
                job["returncode"] = returncode
                manifest["updated_at"] = utc_timestamp()
                write_manifest(manifest_path, manifest)
            job["status"] = "completed"
            manifest["completed_jobs"] = index
            manifest["updated_at"] = utc_timestamp()
            write_manifest(manifest_path, manifest)
        subprocess.run(compare_command, cwd=PROJECT_ROOT, check=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_timestamp()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_timestamp()
    write_manifest(manifest_path, manifest)
    print(f"Comparison artifacts: {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
