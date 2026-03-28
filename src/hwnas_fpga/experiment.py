"""Run artifact management for search experiments."""

from __future__ import annotations

import csv
import json
import sys
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml

from hwnas_fpga.hardware import CostEstimate
from hwnas_fpga.interfaces import SearchCandidate


def _utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slugify(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isalnum():
            chars.append(char.lower())
        elif char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        primary = self._streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())


class ConsoleCapture(AbstractContextManager["ConsoleCapture"]):
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._handle: Optional[Any] = None
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> "ConsoleCapture":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(self._stdout, self._handle)
        sys.stderr = _TeeStream(self._stderr, self._handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        return None


class ExperimentTracker:
    """Persist logs, intermediate results, and checkpoints for one run."""

    def __init__(
        self,
        output_root: Path,
        *,
        project_name: str,
        script_name: str,
        search_method: str,
        dataset_name: str,
        run_name: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            folder_name = _slugify(run_name)
        else:
            folder_name = "_".join(
                [
                    _slugify(project_name),
                    _slugify(script_name),
                    _slugify(search_method),
                    _slugify(dataset_name),
                    timestamp,
                ]
            )

        self.root_dir = output_root / folder_name
        self.logs_dir = self.root_dir / "logs"
        self.results_dir = self.root_dir / "results"
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.candidates_dir = self.results_dir / "candidates"
        for path in (
            self.root_dir,
            self.logs_dir,
            self.results_dir,
            self.checkpoints_dir,
            self.candidates_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.console_log_path = self.logs_dir / "console.log"
        self.candidates_jsonl_path = self.results_dir / "candidates.jsonl"
        self._run_state: dict[str, Any] = {
            "project_name": project_name,
            "script_name": script_name,
            "search_method": search_method,
            "dataset_name": dataset_name,
            "run_name": folder_name,
            "created_at": _utc_timestamp(),
            "status": "running",
        }
        self._write_json(self.root_dir / "run_info.json", self._run_state)

    def capture_console(self) -> ConsoleCapture:
        return ConsoleCapture(self.console_log_path)

    def save_config(self, config: Mapping[str, Any], cli_args: Mapping[str, Any]) -> None:
        with (self.root_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, allow_unicode=True, sort_keys=False)
        self._write_json(self.root_dir / "cli_args.json", dict(cli_args))

    def write_dataset_summary(self, payload: Mapping[str, Any]) -> None:
        self._write_json(self.results_dir / "dataset_summary.json", payload)

    def write_search_space_summary(self, payload: Mapping[str, Any]) -> None:
        self._write_json(self.results_dir / "search_space_summary.json", payload)

    def write_pruned_search_space_summary(self, payload: Mapping[str, Any]) -> None:
        self._write_json(self.results_dir / "search_space_pruned_summary.json", payload)

    def write_baseline(
        self,
        *,
        architecture: Mapping[str, Any],
        cost_estimate: CostEstimate,
    ) -> None:
        self._write_json(
            self.results_dir / "baseline.json",
            {
                "recorded_at": _utc_timestamp(),
                "architecture": architecture,
                "cost_estimate": self._cost_to_dict(cost_estimate),
            },
        )

    def record_candidate(
        self,
        candidate: SearchCandidate,
        *,
        feasible: bool,
        cost_estimate: Optional[CostEstimate] = None,
        history: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        record = {
            "recorded_at": _utc_timestamp(),
            "feasible": feasible,
            "candidate": self._candidate_to_dict(candidate),
            "cost_estimate": self._cost_to_dict(cost_estimate) if cost_estimate else None,
            "history": dict(history) if history is not None else None,
            "extra": dict(extra) if extra is not None else {},
        }
        self._append_jsonl(self.candidates_jsonl_path, record)
        self._write_json(self.candidates_dir / f"{candidate.arch_id}.json", record)

    def update_search_state(
        self,
        *,
        total_evaluated: int,
        feasible: int,
        infeasible: int,
        best_candidate: Optional[SearchCandidate],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        state = {
            "updated_at": _utc_timestamp(),
            "total_evaluated": total_evaluated,
            "feasible": feasible,
            "infeasible": infeasible,
            "best_candidate": self._candidate_to_dict(best_candidate) if best_candidate else None,
            "extra": dict(extra) if extra is not None else {},
        }
        self._write_json(self.checkpoints_dir / "search_state.json", state)

    def save_best_candidate(
        self,
        candidate: SearchCandidate,
        *,
        model_state_dict: Optional[Mapping[str, Any]] = None,
        history: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {
            "saved_at": _utc_timestamp(),
            "candidate": self._candidate_to_dict(candidate),
            "history": dict(history) if history is not None else None,
            "extra": dict(extra) if extra is not None else {},
        }
        self._write_json(self.checkpoints_dir / "best_candidate.json", payload)
        self._write_json(self.results_dir / "best_candidate.json", payload)
        if model_state_dict is not None:
            checkpoint = {
                "saved_at": _utc_timestamp(),
                "candidate": payload["candidate"],
                "history": payload["history"],
                "extra": payload["extra"],
                "model_state_dict": model_state_dict,
            }
            torch.save(checkpoint, self.checkpoints_dir / "best_model.pt")

    def save_named_checkpoint(
        self,
        filename: str,
        payload: Mapping[str, Any],
    ) -> None:
        torch.save(dict(payload), self.checkpoints_dir / filename)

    def finalize(
        self,
        *,
        status: str,
        candidates: Sequence[SearchCandidate],
        feasible_candidates: Sequence[SearchCandidate],
        infeasible_candidates: Sequence[SearchCandidate],
        best_candidate: Optional[SearchCandidate],
        pareto_candidates: Optional[Sequence[SearchCandidate]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        serialized_candidates = [self._candidate_to_dict(candidate) for candidate in candidates]
        self._write_json(self.results_dir / "candidates.json", serialized_candidates)
        self._write_candidates_csv(serialized_candidates)
        if pareto_candidates is not None:
            self._write_json(
                self.results_dir / "pareto_front.json",
                [self._candidate_to_dict(candidate) for candidate in pareto_candidates],
            )

        summary = {
            "finished_at": _utc_timestamp(),
            "status": status,
            "total_evaluated": len(candidates),
            "feasible": len(feasible_candidates),
            "infeasible": len(infeasible_candidates),
            "best_candidate": self._candidate_to_dict(best_candidate) if best_candidate else None,
            "extra": dict(extra) if extra is not None else {},
        }
        self._run_state["status"] = status
        self._run_state["finished_at"] = summary["finished_at"]
        self._write_json(self.results_dir / "summary.json", summary)
        self._write_json(self.root_dir / "run_info.json", self._run_state)

    def mark_failed(self, error_message: str) -> None:
        self._run_state["status"] = "failed"
        self._run_state["finished_at"] = _utc_timestamp()
        self._run_state["error"] = error_message
        self._write_json(self.root_dir / "run_info.json", self._run_state)

    def _write_candidates_csv(self, rows: Sequence[Mapping[str, Any]]) -> None:
        csv_path = self.results_dir / "candidates.csv"
        fieldnames = [
            "arch_id",
            "accuracy",
            "latency_ms",
            "energy_mj",
            "lut",
            "bram",
            "dsp",
            "power_w",
            "memory_bandwidth_gbps",
            "offchip_mem_mb",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                metrics = row["metrics"]
                writer.writerow(
                    {
                        "arch_id": row["arch_id"],
                        "accuracy": metrics.get("accuracy"),
                        "latency_ms": metrics.get("latency_ms"),
                        "energy_mj": metrics.get("energy_mj"),
                        "lut": metrics.get("lut"),
                        "bram": metrics.get("bram"),
                        "dsp": metrics.get("dsp"),
                        "power_w": metrics.get("power_w"),
                        "memory_bandwidth_gbps": metrics.get("memory_bandwidth_gbps"),
                        "offchip_mem_mb": metrics.get("offchip_mem_mb"),
                    }
                )

    def _candidate_to_dict(self, candidate: Optional[SearchCandidate]) -> Optional[dict[str, Any]]:
        if candidate is None:
            return None
        return {
            "arch_id": candidate.arch_id,
            "encoding": candidate.encoding,
            "metrics": asdict(candidate.metrics),
        }

    def _cost_to_dict(self, estimate: CostEstimate) -> dict[str, Any]:
        return asdict(estimate)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload: Any) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
