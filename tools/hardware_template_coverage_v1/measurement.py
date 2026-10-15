"""Fail-closed statistics and elapsed-time extrapolation for Experiment 3.

This module consumes actual raw records. It does not manufacture measurements,
fill missing durations, trim slow samples, or turn nominal policy counts into
physical synthesis counts. All times in timing CSV files are nanoseconds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime


SCOPES = ("compute", "transfer", "call")
HASH_FIELDS = ("case_sha256", "numeric_contract_sha256", "bitstream_sha256", "expected_output_sha256")
CSV_FIELDS = ("case_id", "iteration", "phase", *HASH_FIELDS, "clock_hz", "board_id",
              "compute_ns", "transfer_ns", "call_ns", "compared_elements", "mismatched_elements",
              "observed_output_sha256", "batch_id", "group_id", "group_size", "group_call_total_ns")
REQUIRED_BUILD_STAGES = ("python_reference", "csim", "hls_synthesis", "route", "bitstream", "board_test")


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " is boolean")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + " must be finite and positive")
    return number


def _int(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(name + " is boolean")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(number)


def statistics_ns(values):
    """Sample SD (ddof=1), linear-interpolated P95, no exclusions."""
    checked = [_positive(value, "duration_ns") for value in values]
    if len(checked) < 2:
        raise ValueError("sample SD requires at least two values")
    ordered = sorted(checked)
    position = 0.95 * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    p95 = ordered[left] + (position-left) * (ordered[right]-ordered[left])
    mean = statistics.mean(checked)
    deviation = statistics.stdev(checked)
    return dict(n=len(checked), mean_ns=mean, median_ns=statistics.median(checked),
                p95_ns=p95, sample_std_ns=deviation, cv=deviation/mean,
                minimum_ns=min(checked), maximum_ns=max(checked), ddof=1,
                quantile_method="linear", trimmed_samples=0)


def summarize(rows, manifest):
    """Validate bound case identities and summarize every expected case.

    cases bind hashes/board/expected_elements. batches bind actual clock,
    independent reload/restart event, governor and temperatures. At least three
    batches each have 50 warmups and 1000 measured calls by default. Host-call
    timing uses one total/N observation per group. Only PL compute/transfer CV
    has the default 5% gate. Host median stability is descriptive unless an
    explicit DRAFT threshold is provided. Hash strings are not file checks.
    """
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("manifest must bind at least one case")
    warmup_min = _int(manifest.get("warmup_minimum", 50), "warmup_minimum", 50)
    measured_min = _int(manifest.get("measured_minimum", 1000), "measured_minimum", 1000)
    cv_limit = _positive(manifest.get("cv_limit", .05), "cv_limit")
    if cv_limit > .05:
        raise ValueError("CV acceptance cannot exceed 5 percent")
    batches_min = _int(manifest.get("independent_batches_minimum", 3), "independent_batches_minimum", 3)
    host_limit = manifest.get("host_median_relative_limit")
    if host_limit is not None:
        host_limit = _positive(host_limit, "host_median_relative_limit")
        if manifest.get("host_limit_status") != "DRAFT":
            raise ValueError("optional host median threshold must be explicitly marked DRAFT")
    expected = {}
    for spec in manifest["cases"]:
        case = spec.get("case_id")
        if not isinstance(case, str) or not case or case in expected:
            raise ValueError("nonempty unique case_id required")
        if any(not _hash(spec.get(key)) for key in HASH_FIELDS):
            raise ValueError(case + ": four valid SHA256 bindings required")
        _int(spec.get("expected_elements"), "expected_elements", 1)
        if not isinstance(spec.get("board_id"), str) or not spec["board_id"]:
            raise ValueError(case + ": board identity missing")
        expected[case] = spec
    if not isinstance(manifest.get("batches"), list):
        raise ValueError("manifest batches metadata list required")
    batch_specs, event_ids, windows = {}, defaultdict(set), defaultdict(list)
    condition_deviations = []
    global_errors = []
    for batch in manifest["batches"]:
        case, batch_id = batch.get("case_id"), batch.get("batch_id")
        try:
            if case not in expected or not isinstance(batch_id, str) or not batch_id or (case, batch_id) in batch_specs:
                raise ValueError("invalid/duplicate case_id and batch_id")
            if batch.get("independence_event") not in ("reload_bitstream", "restart_runtime"):
                raise ValueError("independent batch requires bitstream reload or runtime restart")
            event = batch.get("independence_id")
            if not isinstance(event, str) or not event or event in event_ids[case]:
                raise ValueError("unique independence event identity required")
            event_ids[case].add(event)
            frequency = _positive(batch.get("clock_hz"), "batch actual clock_hz")
            if _positive(batch.get("clock_hz_end"), "batch ending clock_hz") != frequency:
                raise ValueError("PL clock changed within batch")
            reported_deviations = batch.get("condition_deviations", [])
            if not isinstance(reported_deviations, list) or any(not isinstance(item, str) or not item for item in reported_deviations):
                raise ValueError("condition_deviations must be a list of nonempty explanations")
            for deviation in reported_deviations:
                condition_deviations.append(dict(case_id=case, batch_id=batch_id,
                    condition="runner_recorded_condition", observed=deviation,
                    implication="preserve the measured observations and retain the recorded environment limitation"))
            if batch.get("cpu_governor") != "performance":
                condition_deviations.append(dict(case_id=case, batch_id=batch_id,
                    condition="cpu_governor", observed=batch.get("cpu_governor"),
                    implication="host timing conditions differ; preserve raw data and limit interpretation"))
            for field in ("temperature_c", "temperature_c_end"):
                if batch.get(field) is None:
                    condition_deviations.append(dict(case_id=case, batch_id=batch_id,
                        condition=field, observed=None, implication="temperature not observed; no thermal-control claim"))
                elif isinstance(batch.get(field), bool) or not math.isfinite(float(batch.get(field))):
                    raise ValueError("finite batch temperatures required")
            start = datetime.fromisoformat(batch["started_utc"].replace("Z", "+00:00"))
            finish = datetime.fromisoformat(batch["finished_utc"].replace("Z", "+00:00"))
            if start.tzinfo is None or finish.tzinfo is None or finish <= start:
                raise ValueError("ordered timezone-aware batch start/end required")
            batch_specs[(case, batch_id)] = batch
            windows[case].append((start, finish, batch_id))
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            global_errors.append(f"batch {case}/{batch_id}: {error}")
    for case, intervals in windows.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                global_errors.append(f"{case}: independent batch windows overlap ({previous[2]}, {current[2]})")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        case = row.get("case_id")
        if case not in expected:
            global_errors.append(f"row {index}: unexpected case_id {case!r}")
        elif (case, row.get("batch_id")) not in batch_specs:
            global_errors.append(f"row {index}: missing or invalid independent batch metadata")
        else:
            grouped[(case, row["batch_id"])].append((index, row))
    results = []
    for case, spec in expected.items():
        errors, samples, batch_results = [], {scope: [] for scope in SCOPES}, []
        total_warmups, total_measured, total_parity, total_raw = 0, 0, 0, 0
        active_batches = [(key, value) for key, value in batch_specs.items() if key[0] == case]
        if len(active_batches) < batches_min:
            errors.append(f"only {len(active_batches)} independent batches; require {batches_min}")
        all_group_sizes = defaultdict(int)
        all_host_by_size = defaultdict(list)
        for key, batch in active_batches:
            batch_id = key[1]
            batch_errors, seen, groups = [], set(), defaultdict(list)
            batch_samples = {scope: [] for scope in SCOPES}
            batch_host_by_size = defaultdict(list)
            warmups, measured, saw_measured, parity_checks = 0, 0, False, 0
            for index, row in grouped[key]:
                try:
                    missing = set(CSV_FIELDS) - set(row)
                    if missing:
                        raise ValueError("missing fields " + ",".join(sorted(missing)))
                    for field in HASH_FIELDS:
                        if row[field] != spec[field] or not _hash(row[field]):
                            raise ValueError("binding mismatch: " + field)
                    if row["board_id"] != spec["board_id"]:
                        raise ValueError("board identity mismatch")
                    if _positive(row["clock_hz"], "clock_hz") != float(batch["clock_hz"]):
                        raise ValueError("actual PL clock mismatch")
                    phase = row["phase"]
                    if phase not in ("warmup", "measured"):
                        raise ValueError("phase must be warmup or measured")
                    iteration = _int(row["iteration"], "iteration")
                    if (phase, iteration) in seen:
                        raise ValueError("duplicate phase/iteration")
                    seen.add((phase, iteration))
                    if phase == "warmup":
                        if saw_measured:
                            raise ValueError("warmup occurs after measured trials")
                        warmups += 1
                    else:
                        saw_measured = True
                        measured += 1
                    durations = {scope: _positive(row[scope + "_ns"], scope + "_ns") for scope in SCOPES}
                    group_size = _int(row["group_size"], "group_size", 1)
                    total = _positive(row["group_call_total_ns"], "group_call_total_ns")
                    if not str(row["group_id"]):
                        raise ValueError("group_id must be nonempty")
                    if not math.isclose(durations["call"], total/group_size, rel_tol=1e-12, abs_tol=1e-6):
                        raise ValueError("call_ns must equal group_call_total_ns / group_size")
                    compared = _int(row["compared_elements"], "compared_elements", 1)
                    if compared != int(spec["expected_elements"]):
                        raise ValueError("incomplete output comparison")
                    if _int(row["mismatched_elements"], "mismatched_elements") != 0 or row["observed_output_sha256"] != spec["expected_output_sha256"]:
                        raise ValueError("bit-exact parity failed")
                    parity_checks += 1
                    groups[(phase, str(row["group_id"]))].append((group_size, total, durations))
                    if phase == "measured":
                        for scope in ("compute", "transfer"):
                            batch_samples[scope].append(durations[scope])
                except (ValueError, TypeError, OverflowError) as error:
                    batch_errors.append(f"row {index}: {error}")
            for (phase, group_id), members in groups.items():
                sizes, totals = {m[0] for m in members}, {m[1] for m in members}
                if len(sizes) != 1 or len(totals) != 1 or len(members) != members[0][0]:
                    batch_errors.append(f"{phase} group {group_id}: incomplete/inconsistent raw group; exclusions forbidden")
                    continue
                size, total = members[0][:2]
                if total < sum(m[2]["compute"]+m[2]["transfer"] for m in members):
                    batch_errors.append(f"{phase} group {group_id}: call window cannot contain measured PL intervals")
                if phase == "measured":
                    all_group_sizes[size] += 1
                    batch_samples["call"].append(total/size)
                    batch_host_by_size[size].append(total/size)
                    all_host_by_size[size].append(total/size)
            if warmups < warmup_min:
                batch_errors.append(f"only {warmups} warmup rows; require {warmup_min}")
            if measured < measured_min:
                batch_errors.append(f"only {measured} measured rows; require {measured_min}")
            stats = {}
            for scope in SCOPES:
                values = batch_samples[scope]
                if scope != "call" and len(values) != measured:
                    batch_errors.append(f"{scope}: invalid or incomplete raw observations; exclusions forbidden")
                stats[scope] = statistics_ns(values) if len(values) >= 2 else None
                if stats[scope] is None:
                    batch_errors.append(scope + ": fewer than two valid observations")
                elif scope != "call" and stats[scope]["cv"] > cv_limit:
                    batch_errors.append(f"{scope}: CV exceeds {cv_limit}")
                samples[scope].extend(values)
            batch_results.append(dict(batch_id=batch_id, status="PASS" if not batch_errors else "FAIL",
                                      errors=batch_errors, metadata=batch, warmup_rows=warmups,
                                      measured_rows=measured, host_group_observations=len(batch_samples["call"]),
                                      host_group_size_distribution={size: len(values) for size, values in sorted(batch_host_by_size.items())},
                                      host_scopes_by_group_size={size: statistics_ns(values) if len(values) >= 2 else None
                                          for size, values in sorted(batch_host_by_size.items())},
                                      parity_checks=parity_checks, scopes=stats))
            errors.extend(batch_id + ": " + error for error in batch_errors)
            total_warmups += warmups
            total_measured += measured
            total_parity += parity_checks
            total_raw += len(grouped[key])
        scope_stats = {}
        for scope in SCOPES:
            values = samples[scope]
            if len(values) >= 2:
                scope_stats[scope] = statistics_ns(values)
                if scope != "call" and scope_stats[scope]["cv"] > cv_limit:
                    errors.append(f"{scope}: pooled CV exceeds {cv_limit}")
            else:
                scope_stats[scope] = None
        medians = [row["scopes"]["call"]["median_ns"] for row in batch_results if row["scopes"]["call"]]
        reference_median = statistics.median(medians) if medians else None
        deviation = max(abs(value/reference_median-1) for value in medians) if medians else None
        if host_limit is not None and len(all_group_sizes) > 1:
            errors.append("explicit host median limit cannot pool different group sizes; compare within N strata")
        if host_limit is not None and (deviation is None or deviation > host_limit):
            errors.append("host batch median deviation exceeds explicit DRAFT limit")
        results.append(dict(case_id=case, status="PASS" if not errors else "FAIL", errors=errors,
                            warmup_rows=total_warmups, measured_rows=total_measured, parity_checks=total_parity,
                            expected_raw_rows=total_raw, independent_batches=len(active_batches), batches=batch_results,
                            scopes=scope_stats, host_group_size=next(iter(all_group_sizes)) if len(all_group_sizes)==1 else None,
                            host_group_size_distribution=dict(sorted(all_group_sizes.items())),
                            host_scopes_by_group_size={size: statistics_ns(values) if len(values) >= 2 else None
                                for size, values in sorted(all_host_by_size.items())},
                            host_group_observations=len(samples["call"]),
                            host_median_stability=dict(batch_medians_ns=medians, median_of_batch_medians_ns=reference_median,
                                maximum_relative_deviation=deviation, threshold=host_limit,
                                comparability="SAME_GROUP_SIZE" if len(all_group_sizes) == 1 else "MIXED_GROUP_SIZES_USE_N_STRATIFIED_STATISTICS",
                                status="DESCRIPTIVE_NOT_AN_ACCEPTANCE_GATE" if host_limit is None else "EXPLICIT_DRAFT_THRESHOLD"),
                            all_output_bits_equal=total_raw > 0 and total_parity == total_raw,
                            formal_cost_eligible=False))
    passed = not global_errors and all(result["status"] == "PASS" for result in results)
    return dict(schema_version="hardware_pilot_timing_summary_v2", status="PASS" if passed else "FAIL",
                global_errors=global_errors, cases=results, measured_minimum=measured_min,
                warmup_minimum=warmup_min, independent_batches_minimum=batches_min, cv_limit=cv_limit,
                host_median_relative_limit=host_limit, protocol_status="DRAFT", evidence_scope="PILOT_ONLY",
                condition_deviations=condition_deviations, protocol_conditions_met=not condition_deviations,
                phase_clock_evidence_verified=manifest.get("phase_clock_evidence_verified") is True,
                hardware_measurement_eligible=passed and not condition_deviations and manifest.get("phase_clock_evidence_verified") is True,
                note="PASS covers raw timing/parity/batch statistics only. Raw observations are never trimmed; host group means are counted once, no default host CV gate; RTL/clock proof remains separate.",
                formal_cost_eligible=False)


def summarize_csv(path, manifest):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        result = summarize(list(csv.DictReader(stream)), manifest)
    result["raw_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def extrapolate(stage_records, inventory_counts, *, physical_bindings_complete=False,
                available_hours_per_day=24, parallel_workers=1, shared_setup_seconds=0,
                minimum_complete_per_stratum=2):
    """Stratified sequential work estimate with an explicit ideal-parallel bound.

    Each record binds case_id + attempt_id + stratum, status=COMPLETE or FAILED,
    parity_pass/timing_pass, and stages={stage: seconds}. Successful cases need
    all REQUIRED_BUILD_STAGES. Failed-attempt time is charged to its stratum.
    inventory_counts must count deduplicated physical configurations, not
    architecture x profile x named selector decisions. Counts may provisionally
    be supplied before physical binding, but no full-days estimate is issued.
    """
    hours = _positive(available_hours_per_day, "available_hours_per_day")
    if hours > 24:
        raise ValueError("available_hours_per_day cannot exceed 24")
    workers = _int(parallel_workers, "parallel_workers", 1)
    minimum = _int(minimum_complete_per_stratum, "minimum_complete_per_stratum", 1)
    if isinstance(shared_setup_seconds, bool):
        raise ValueError("invalid shared setup duration")
    setup = float(shared_setup_seconds)
    if not math.isfinite(setup) or setup < 0:
        raise ValueError("invalid shared setup duration")
    if not inventory_counts:
        raise ValueError("physical inventory counts required")
    counts = {key: _int(value, key + " count", 1) for key, value in inventory_counts.items()}
    grouped, errors, seen = defaultdict(list), [], set()
    for record in stage_records:
        key = (record.get("case_id"), record.get("attempt_id"))
        if not all(isinstance(value, str) and value for value in key) or key in seen:
            errors.append("missing/duplicate case_id + attempt_id")
            continue
        seen.add(key)
        stratum = record.get("stratum")
        if stratum not in counts:
            errors.append("unbound stratum " + str(stratum))
            continue
        try:
            stage_values = {name: _positive(value, name) for name, value in record.get("stages", {}).items()}
            if record.get("status") == "COMPLETE":
                missing = set(REQUIRED_BUILD_STAGES) - set(stage_values)
                if missing:
                    raise ValueError("completed record missing stages " + ",".join(sorted(missing)))
                if record.get("parity_pass") is not True or record.get("timing_pass") is not True:
                    raise ValueError("completed case lacks parity/timing acceptance")
            elif record.get("status") != "FAILED" or not stage_values:
                raise ValueError("attempt status must be COMPLETE or timed FAILED")
            grouped[stratum].append((record, stage_values))
        except (ValueError, TypeError, OverflowError, AttributeError) as error:
            errors.append(str(key) + ": " + str(error))
    strata, missing_strata, total_seconds = [], [], setup
    for stratum, count in counts.items():
        records = grouped[stratum]
        complete = [(record, times) for record, times in records if record["status"] == "COMPLETE"]
        unique_cases = {record["case_id"] for record, _ in complete}
        if len(unique_cases) < minimum:
            missing_strata.append(stratum)
            strata.append(dict(stratum=stratum, count=count, complete_cases=len(unique_cases),
                               status="INSUFFICIENT_COMPLETE_REPRESENTATIVES", projected_seconds=None))
            continue
        if len(complete) != len(unique_cases):
            errors.append(stratum + ": repeated successful case would bias the representative average")
        total_attempt_seconds = sum(sum(times.values()) for _, times in records)
        seconds_per_success = total_attempt_seconds / len(complete)
        projected = count * seconds_per_success
        total_seconds += projected
        stage_means = {stage: statistics.mean(times[stage] for _, times in complete)
                       for stage in REQUIRED_BUILD_STAGES}
        strata.append(dict(stratum=stratum, count=count, complete_cases=len(unique_cases),
                           attempted_records=len(records), seconds_per_success_including_failures=seconds_per_success,
                           mean_stage_seconds=stage_means, projected_seconds=projected, status="COVERED"))
    if not physical_bindings_complete:
        errors.append("exact physical mapping configurations have not been bound and deduplicated")
    ready = not errors and not missing_strata
    sequential_days = total_seconds / (hours * 3600) if ready else None
    return dict(schema_version="hardware_pilot_extrapolation_v1",
                status="ESTIMATE_FROM_COMPLETE_STRATA" if ready else "PROVISIONAL_MISSING_EVIDENCE",
                strata=strata, missing_strata=missing_strata, errors=errors,
                required_stages=list(REQUIRED_BUILD_STAGES), available_hours_per_day=hours,
                shared_setup_seconds=setup, sequential_days=sequential_days,
                ideal_parallel_lower_bound_days=(total_seconds-setup)/(workers*hours*3600)+setup/(hours*3600) if ready else None,
                calendar_prediction=None, parallel_workers=workers,
                exceeds_three_weeks_sequential=sequential_days > 21 if ready else None,
                mapping_reduction_recommendation=("REVIEW_2_TO_3_DISTINCT_CONFIGURATIONS" if sequential_days > 21 else "NO_TIME_TRIGGER") if ready else "DEFER_UNTIL_MEASURED",
                note="Worker scaling is an ideal lower bound; licensed HLS/route hosts and one shared board may serialize work. Pilot strata do not capture every shape-dependent runtime.",
                formal_cost_eligible=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    result = summarize_csv(args.csv, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"])
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
