"""Search-time accounting shared by all search methods.

The project compares search methods using both result quality and compute cost.
This module records two deliberately distinct GPU-time views:

* ``gpu_reserved_wall_seconds``: wall time while a CUDA device is assigned to
  the search call.  This is the quantity used for GPU-hour accounting.
* ``gpu_event_seconds``: elapsed CUDA stream time between two synchronized
  events.  It approximates active accelerator work and excludes most host-side
  input and orchestration stalls.

Keeping both avoids presenting host waiting time as kernel execution time or,
conversely, hiding the real GPU allocation cost of a search experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, process_time
from typing import Any, Optional

import torch


def merge_search_efficiency(
    previous: Optional[dict[str, Any]],
    segment: dict[str, Any],
) -> dict[str, Any]:
    """Accumulate resumed search segments without under-counting GPU-hours."""

    if not previous:
        merged = dict(segment)
        merged["segment_count"] = 1
        merged["segments"] = [dict(segment)]
        return merged

    if previous.get("search_method") != segment.get("search_method"):
        raise ValueError("cannot merge efficiency ledgers from different search methods")

    def summed(key: str) -> float:
        return float(previous.get(key) or 0.0) + float(segment.get(key) or 0.0)

    previous_gpu_event = previous.get("gpu_event_seconds")
    segment_gpu_event = segment.get("gpu_event_seconds")
    if previous_gpu_event is None and segment_gpu_event is None:
        gpu_event_seconds = None
    else:
        gpu_event_seconds = float(previous_gpu_event or 0.0) + float(segment_gpu_event or 0.0)
    previous_peak = previous.get("peak_cuda_memory_bytes")
    segment_peak = segment.get("peak_cuda_memory_bytes")
    peak_values = [int(value) for value in (previous_peak, segment_peak) if value is not None]
    candidate_count = int(previous.get("candidate_count") or 0) + int(
        segment.get("candidate_count") or 0
    )
    feasible_count = int(previous.get("feasible_candidate_count") or 0) + int(
        segment.get("feasible_candidate_count") or 0
    )
    wall_seconds = summed("wall_clock_seconds")
    gpu_reserved_seconds = summed("gpu_reserved_wall_seconds")
    existing_segments = previous.get("segments")
    if not isinstance(existing_segments, list):
        existing_segments = [
            {
                key: previous.get(key)
                for key in (
                    "wall_clock_seconds",
                    "host_process_seconds",
                    "gpu_reserved_wall_seconds",
                    "gpu_event_seconds",
                    "candidate_count",
                    "feasible_candidate_count",
                )
            }
        ]
    merged = dict(segment)
    merged.update(
        {
            "measurement_scope": "cumulative_whole_search_calls",
            "wall_clock_seconds": wall_seconds,
            "host_process_seconds": summed("host_process_seconds"),
            "gpu_reserved_wall_seconds": gpu_reserved_seconds,
            "gpu_reserved_hours": gpu_reserved_seconds / 3600.0,
            "gpu_event_seconds": gpu_event_seconds,
            "peak_cuda_memory_bytes": max(peak_values) if peak_values else None,
            "candidate_count": candidate_count,
            "feasible_candidate_count": feasible_count,
            "wall_seconds_per_candidate": (
                wall_seconds / candidate_count if candidate_count else None
            ),
            "gpu_reserved_seconds_per_candidate": (
                gpu_reserved_seconds / candidate_count
                if candidate_count and bool(previous.get("cuda_used") or segment.get("cuda_used"))
                else None
            ),
            "segment_count": int(previous.get("segment_count") or len(existing_segments)) + 1,
            "segments": [*existing_segments, dict(segment)],
            "cuda_used": bool(previous.get("cuda_used") or segment.get("cuda_used")),
        }
    )
    return merged


@dataclass
class SearchEfficiencyMonitor:
    """Measure one complete search call without changing the search method."""

    device: str
    _wall_start: Optional[float] = field(default=None, init=False, repr=False)
    _cpu_start: Optional[float] = field(default=None, init=False, repr=False)
    _wall_seconds: float = field(default=0.0, init=False, repr=False)
    _cpu_seconds: float = field(default=0.0, init=False, repr=False)
    _cuda_start: Any = field(default=None, init=False, repr=False)
    _cuda_end: Any = field(default=None, init=False, repr=False)
    _gpu_event_seconds: Optional[float] = field(default=None, init=False, repr=False)
    _peak_memory_bytes: Optional[int] = field(default=None, init=False, repr=False)

    @property
    def uses_cuda(self) -> bool:
        try:
            return torch.device(self.device).type == "cuda" and torch.cuda.is_available()
        except (RuntimeError, ValueError):
            return False

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    def __enter__(self) -> "SearchEfficiencyMonitor":
        if self.uses_cuda:
            torch.cuda.synchronize(self.torch_device)
            torch.cuda.reset_peak_memory_stats(self.torch_device)
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        self._cpu_start = process_time()
        self._wall_start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Synchronize before closing the wall-clock interval.  Otherwise queued
        # CUDA work could finish after the primary GPU-allocation timer stops,
        # systematically under-reporting GPU-hours.
        if self.uses_cuda and self._cuda_start is not None and self._cuda_end is not None:
            self._cuda_end.record()
            torch.cuda.synchronize(self.torch_device)
            self._gpu_event_seconds = float(
                self._cuda_start.elapsed_time(self._cuda_end) / 1000.0
            )
            self._peak_memory_bytes = int(
                torch.cuda.max_memory_allocated(self.torch_device)
            )
        if self._wall_start is not None:
            self._wall_seconds = perf_counter() - self._wall_start
        if self._cpu_start is not None:
            self._cpu_seconds = process_time() - self._cpu_start
        return None

    def summary(
        self,
        *,
        candidate_count: int,
        feasible_count: int,
        search_method: str,
    ) -> dict[str, Any]:
        candidate_count = max(0, int(candidate_count))
        gpu_reserved_wall_seconds = self._wall_seconds if self.uses_cuda else 0.0
        device_name = None
        device_index = None
        if self.uses_cuda:
            resolved = self.torch_device
            device_index = (
                torch.cuda.current_device()
                if resolved.index is None
                else int(resolved.index)
            )
            device_name = torch.cuda.get_device_name(device_index)
        return {
            "schema_version": 1,
            "measurement_scope": "whole_search_call",
            "search_method": str(search_method),
            "device": str(self.device),
            "cuda_used": self.uses_cuda,
            "cuda_device_index": device_index,
            "cuda_device_name": device_name,
            "wall_clock_seconds": float(self._wall_seconds),
            "host_process_seconds": float(self._cpu_seconds),
            "gpu_reserved_wall_seconds": float(gpu_reserved_wall_seconds),
            "gpu_reserved_hours": float(gpu_reserved_wall_seconds / 3600.0),
            "gpu_event_seconds": self._gpu_event_seconds,
            "peak_cuda_memory_bytes": self._peak_memory_bytes,
            "candidate_count": candidate_count,
            "feasible_candidate_count": max(0, int(feasible_count)),
            "wall_seconds_per_candidate": (
                float(self._wall_seconds / candidate_count)
                if candidate_count
                else None
            ),
            "gpu_reserved_seconds_per_candidate": (
                float(gpu_reserved_wall_seconds / candidate_count)
                if candidate_count and self.uses_cuda
                else None
            ),
            "gpu_time_definition": {
                "comparison_primary": "gpu_reserved_wall_seconds",
                "active_accelerator_proxy": "gpu_event_seconds",
                "note": (
                    "GPU-hours use synchronized wall time while CUDA is assigned; "
                    "CUDA-event time is reported separately and must not be mixed with it."
                ),
            },
        }
