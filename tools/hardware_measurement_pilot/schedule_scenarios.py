"""Conditional build-queue and single-board scheduling, without hardware I/O.

Measured build process durations come from their recorded concurrent context;
they are not isolated sequential benchmarks. A scenario repeats an explicitly
chosen mix in queue order. It is not a prediction of uncovered implementations.
"""
from __future__ import annotations

import heapq
import math


def schedule_pipeline(jobs, *, build_workers=3, implementation_coverage_complete=False,
                      physical_bindings_complete=False, allow_observed_protocol_deviations=False):
    """List-schedule builds, then schedule one board by build completion time.

Each job has job_id, build_seconds, optional board_seconds and board_accepted.
board_seconds is the whole exclusive-board occupancy, including all batches,
loading and checks; it must not be the per-call inference latency. Failed-build
and retry durations belong in build_seconds, with provenance in the caller.
By default, only fully recorded, accepted board occupancy enters the combined
scenario. With allow_observed_protocol_deviations=True, an explicitly validated
observation may also enter that conditional scenario while board_accepted stays
false; this never establishes full-measurement or protocol acceptance.
"""
    if isinstance(build_workers, bool) or not isinstance(build_workers, int) or build_workers < 1:
        raise ValueError("build_workers must be a positive integer")
    jobs = list(jobs)
    if not jobs or len({item['job_id'] for item in jobs}) != len(jobs):
        raise ValueError("nonempty jobs with unique job_id are required")
    def duration(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("stage duration must be finite and positive")
        return float(value)
    ready = [(0.0, index) for index in range(build_workers)]
    heapq.heapify(ready)
    scheduled = []
    for position, item in enumerate(jobs):
        seconds = duration(item['build_seconds'])
        start, worker = heapq.heappop(ready)
        finish = start + seconds
        heapq.heappush(ready, (finish, worker))
        board_seconds = item.get('board_seconds')
        if board_seconds is not None:
            board_seconds = duration(board_seconds)
        scheduled.append(dict(job_id=item['job_id'], queue_position=position, worker=worker,
            build_start_seconds=start, build_finish_seconds=finish, build_seconds=seconds,
            board_seconds=board_seconds, board_accepted=item.get('board_accepted') is True,
            board_observation_validated=item.get('board_observation_validated') is True,
            board_start_seconds=None, board_finish_seconds=None))
    accepted_complete = all(item['board_seconds'] is not None and item['board_accepted'] for item in scheduled)
    board_complete = all(item['board_seconds'] is not None and (item['board_accepted'] or
        (allow_observed_protocol_deviations is True and item['board_observation_validated'])) for item in scheduled)
    board_available = 0.0
    if board_complete:
        for item in sorted(scheduled, key=lambda row: (row['build_finish_seconds'], row['queue_position'])):
            start = max(item['build_finish_seconds'], board_available)
            board_available = start + item['board_seconds']
            item.update(board_start_seconds=start, board_finish_seconds=board_available)
    build_makespan = max(item['build_finish_seconds'] for item in scheduled)
    reasons = []
    if not board_complete:
        reasons.append('MISSING_ACCEPTED_FULL_BOARD_OCCUPANCY')
    elif not accepted_complete:
        reasons.append('BOARD_PROTOCOL_CONDITIONS_NOT_MET_OBSERVATIONS_RETAINED')
    if implementation_coverage_complete is not True:
        reasons.append('IMPLEMENTATION_COVERAGE_INCOMPLETE')
    if physical_bindings_complete is not True:
        reasons.append('PHYSICAL_BINDINGS_INCOMPLETE')
    return dict(schema='pilot_conditional_schedule_v1', build_workers=build_workers, board_workers=1,
        job_count=len(scheduled), build_process_seconds=sum(item['build_seconds'] for item in scheduled),
        conditional_build_makespan_seconds=build_makespan,
        accepted_board_occupancy_complete=accepted_complete,
        observed_board_occupancy_complete=board_complete,
        allow_observed_protocol_deviations=allow_observed_protocol_deviations is True,
        board_occupancy_seconds=sum(item['board_seconds'] for item in scheduled) if board_complete else None,
        conditional_pipeline_makespan_seconds=board_available if board_complete else None,
        full_measurement_makespan_seconds=board_available if not reasons else None,
        full_measurement_status='CONDITIONAL_SCHEDULE_WITH_COMPLETE_INPUTS' if not reasons else 'NOT_ESTIMABLE',
        missing_evidence=reasons, jobs=scheduled,
        calendar_time_status='NOT_ESTIMATED_AVAILABILITY_AND_RECOVERY_CALENDAR_REQUIRED',
        scope='CONDITIONAL_LIST_SCHEDULE_NOT_OBSERVED_ELAPSED_TIME_OR_FORMAL_COST')


def repeated_build_mix(per_case_seconds, nominal_items, *, build_workers=3):
    """Repeat insertion-ordered measured mix; deliberately has no board estimate."""
    if isinstance(nominal_items, bool) or not isinstance(nominal_items, int) or nominal_items < 1:
        raise ValueError('nominal_items must be a positive integer')
    mix = list(per_case_seconds.items())
    if not mix:
        raise ValueError('measured build mix is empty')
    jobs = [dict(job_id=f'{index + 1}:{mix[index % len(mix)][0]}',
                 build_seconds=mix[index % len(mix)][1]) for index in range(nominal_items)]
    result = schedule_pipeline(jobs, build_workers=build_workers)
    result['mix_rule'] = 'Cyclic repetition of the explicit insertion-ordered observed P4 mix; no uncovered-family inference'
    return result
