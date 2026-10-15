"""Read-only exact snapshot and statistics replay for tiled templates.

The numerical validator/comparison remain exact frozen-pilot extractions.
The public validator additionally preserves declared host timing protocol
metadata; legacy artifacts stay unchanged and explicitly unspecified.
PASS_IMPORT_AUDIT does not change numerical or protocol eligibility fields.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
from tools.hardware_template_coverage_v1.measurement import summarize_csv
ROOT = Path(__file__).resolve().parents[2]
__all__ = ["validate_timing_snapshot", "compare_replayed_summary", "timing_protocol"]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else default

def compare_replayed_summary(stored, replay):
    """Compare JSON data exactly, except finite computed statistics at <=8 ULP.

    ARM and Windows libm may round stdev/sqrt by one binary64 ULP. This is a
    representational comparison only: values are neither replaced nor rounded,
    metadata/thresholds stay exact, and every recomputed PASS/FAIL stays exact.
    """
    stored = json.loads(json.dumps(stored, allow_nan=False))
    replay = json.loads(json.dumps(replay, allow_nan=False))
    differences = []
    statistical_fields = {"mean_ns", "median_ns", "p95_ns", "sample_std_ns", "cv", "minimum_ns", "maximum_ns"}
    def computed_statistic(path):
        if len(path) < 4 or path[0] != "cases" or type(path[1]) is not int:
            return False
        tail = path[2:]
        if len(tail) >= 2 and tail[0] == "batches" and type(tail[1]) is int:
            tail = tail[2:]
        if len(tail) == 3 and tail[2] in statistical_fields:
            return ((tail[0] == "scopes" and tail[1] in ("compute", "transfer", "call")) or
                    (tail[0] == "host_scopes_by_group_size" and isinstance(tail[1], str) and tail[1].isdigit()))
        return (len(tail) == 2 and tail[0] == "host_median_stability" and tail[1] in
                ("median_of_batch_medians_ns", "maximum_relative_deviation")) or (
                len(tail) == 3 and tail[:2] == ("host_median_stability", "batch_medians_ns") and type(tail[2]) is int)
    def ordered_bits(value):
        bits = struct.unpack("!Q", struct.pack("!d", value))[0]
        sign = 1 << 63
        return sign - (bits & (sign - 1)) if bits & sign else sign + bits
    def check(actual, expected, path):
        label = "/".join(map(str, path)) or "<root>"
        if type(actual) is not type(expected):
            raise ValueError("summary replay type differs: " + label)
        if isinstance(expected, dict):
            if actual.keys() != expected.keys():
                raise ValueError("summary replay keys differ: " + label)
            for key in expected:
                check(actual[key], expected[key], path + (key,))
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError("summary replay list length differs: " + label)
            for index, (left, right) in enumerate(zip(actual, expected)):
                check(left, right, path + (index,))
        elif isinstance(expected, float):
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise ValueError("nonfinite summary float: " + label)
            if actual != expected:
                distance = abs(ordered_bits(actual) - ordered_bits(expected))
                if not computed_statistic(path) or distance > 8:
                    raise ValueError("summary replay float differs beyond allowed statistic ULP: " + label)
                differences.append(dict(path=label, stored=actual, replay=expected, ulp_distance=distance))
        elif actual != expected:
            raise ValueError("summary replay exact value differs: " + label)
    check(stored, replay, ())
    return dict(status="PASS_REPLAY_EQUIVALENCE", json_roundtrip_for_dictionary_keys=True,
        finite_statistical_float_max_ulp=8, comparison="IEEE754_BINARY64_ORDERED_INTEGER_DISTANCE",
        exact_fields="dictionary keys, list lengths/order, types, counts, identity, strings, booleans, metadata, thresholds, and all PASS/FAIL fields",
        tolerated_float_differences=differences, original_values_modified=False)



def _cpu_governor_display(output, batch_id):
    """Display actual policy reads separately from the frozen coarse summary."""
    details = []
    for name in ("environment_start.json", "environment_end.json"):
        environment = read_json(Path(output) / batch_id / name, {})
        cpu = environment.get("cpu", {})
        policies = cpu.get("policies", [])
        readable = [dict(path=item.get("path"), actual=item.get("actual"), available=item.get("available", []),
                         condition_deviation=item.get("condition_deviation")) for item in policies if item.get("actual")]
        details.append(dict(stage=name, raw_summary=cpu.get("cpu_governor"), policies=readable,
                            frequency_raw_khz=cpu.get("frequency_raw_khz", {})))
    observed = sorted({policy["actual"] for record in details for policy in record["policies"]})
    available = sorted({value for record in details for policy in record["policies"] for value in policy["available"]})
    performance_available = "performance" in available if observed else None
    return dict(observed_policy_values=observed, evidence=details,
        requested_governor="performance", available_policy_values=available,
        requested_governor_available=performance_available,
        display=("/".join(observed) + "（policy实读；performance条件不可用；原summary标记UNAVAILABLE）")
                if observed and any(record["raw_summary"] == "UNAVAILABLE" for record in details)
                else "/".join(observed) if observed else "UNAVAILABLE（未取得policy实际值）")



def validate_timing_snapshot(snapshot, package):
    """Read-only import audit. PASS means faithfully importable, not gate PASS."""
    from tools.hardware_template_coverage_v1.board_runner import _case_identity, _phase_identity, _checked_files
    snapshot, package = Path(snapshot).resolve(), Path(package).resolve()
    evidence, output = snapshot / "evidence", snapshot / "evidence/output"
    download = read_json(snapshot / "download.json", {})
    manifest = read_json(evidence / "snapshot_manifest.json", {})
    if (download.get("archive_sha256") != sha(snapshot / "snapshot.tar.gz") or
            manifest.get("schema") != "pilot_board_snapshot_v1"):
        raise ValueError("timing snapshot archive/manifest binding differs")
    _checked_files(evidence, manifest.get("files"))
    needed = {"request.json", "output/summary.json", "output/session.json", "output/timing_manifest.json", "output/timing_raw.csv"}
    needed.update(f"output/batch{batch}/correctness_fixtures.json" for batch in (1, 2, 3))
    if not needed.issubset(manifest.get("files", {})):
        raise ValueError("timing snapshot is incomplete; leave unselected and preserve it")
    request = read_json(evidence / "request.json", {})
    case = _case_identity(package)
    if (request.get("config", {}).get("mode") != "timing" or
            request.get("config", {}).get("case_sha256") != case["case_sha256"]):
        raise ValueError("snapshot is not a timing request for this exact case")
    phase, _ = _phase_identity(package, case, package / "phase_validation.json")
    summary = read_json(output / "summary.json", {})
    timing = read_json(output / "timing_manifest.json", {})
    session = read_json(output / "session.json", {})
    case_specs = timing.get("cases", [])
    if len(case_specs) != 1:
        raise ValueError("one exact case is required per imported board run")
    expected = dict(case_sha256=case["case_sha256"], numeric_contract_sha256=case["numeric_contract_sha256"],
                    bitstream_sha256=phase["bitstream_sha256"])
    if any(case_specs[0].get(key) != value or session.get(key) != value for key, value in expected.items()):
        raise ValueError("timing manifest/session identity differs")
    if (session.get("phase_validation_sha256") != sha(package / "phase_validation.json") or
            session.get("board_bundle_sha256") != sha(package / "board_bundle.json") or
            summary.get("board_bundle_sha256") != sha(package / "board_bundle.json")):
        raise ValueError("timing phase/bundle identity differs")
    # Replay every derived statistic using the same frozen summarizer source.
    if sha(package / "measurement.py") != sha(ROOT / "tools/hardware_template_coverage_v1/measurement.py"):
        raise ValueError("frozen measurement source differs; use its recorded version for replay")
    replay = summarize_csv(output / "timing_raw.csv", timing)
    replay.update(correctness_fixtures_passed=15, integrated_phase_validation="PASS_BOUND_ARTIFACT",
                  board_bundle_sha256=sha(package / "board_bundle.json"), protocol_status="DRAFT", formal_cost_eligible=False)
    comparison = compare_replayed_summary(summary, replay)
    expected_fixtures = {item["fixture_id"]: item for item in case["fixtures"]}
    for batch in (1, 2, 3):
        folder = output / f"batch{batch}"
        fixtures = read_json(folder / "correctness_fixtures.json", [])
        if len(fixtures) != len(expected_fixtures) or {row.get("fixture_id") for row in fixtures} != set(expected_fixtures):
            raise ValueError("timing correctness-fixture coverage differs")
        for row in fixtures:
            name = row["fixture_id"]
            spec = expected_fixtures[name]
            binary = f"output/batch{batch}/{name}_board.bin"
            if (binary not in manifest["files"] or row.get("mismatched_elements") != 0 or
                    row.get("compared_elements") != spec["expected_elements"] or
                    row.get("observed_output_sha256") != spec["expected_output_sha256"] or
                    (folder / (name + "_board.bin")).read_bytes() != (package / (name + "_expected.bin")).read_bytes()):
                raise ValueError("timing fixture does not match full integer output")
    return dict(status="PASS_IMPORT_AUDIT", case_id=case["case_id"], case_sha256=case["case_sha256"],
        snapshot=snapshot.as_posix(), output=output.as_posix(), snapshot_manifest_sha256=sha(evidence / "snapshot_manifest.json"),
        summary_sha256=sha(output / "summary.json"), raw_csv_sha256=sha(output / "timing_raw.csv"),
        numerical_status=summary.get("status"), protocol_conditions_met=summary.get("protocol_conditions_met"),
        hardware_measurement_eligible=summary.get("hardware_measurement_eligible"),
        condition_deviations=summary.get("condition_deviations", []),
        replay_comparison=comparison,
        cpu_governor_observations={f"batch{index}": _cpu_governor_display(output, f"batch{index}") for index in (1, 2, 3)},
        scope="READ_ONLY_SNAPSHOT_HASH_IDENTITY_FULL_FIXTURES_AND_RAW_STATISTICS_REPLAY")


# Preserve the frozen numerical replay function for audit and regression tests.
_validate_timing_snapshot_without_runtime_protocol = validate_timing_snapshot


def timing_protocol(session, manifest):
    """Preserve host interval declarations without merging distinct protocols."""
    fields = ("runtime_protocol_id", "host_timing_guard_version",
              "individual_call_ns_scope", "group_call_ns_scope")
    if not any(field in source for source in (session, manifest) for field in fields):
        return dict(status="LEGACY_UNSPECIFIED", runtime_protocol_id="LEGACY_UNSPECIFIED",
            host_timing_guard_version="LEGACY_UNSPECIFIED", individual_call_ns_scope=None,
            group_call_ns_scope=session.get("call_scope"), call_scope=session.get("call_scope"),
            individual_invoke_identity_checks_inside_call_ns=None,
            invocation_identity_checks_inside_group_total_ns=None,
            fixture_selection_scope=session.get("fixture_selection_scope"),
            host_timing_comparability="Legacy artifact has no recorded runtime/guard protocol identifier",
            host_statistics_pooling="NOT_PERFORMED_SINGLE_SELECTED_RUN_PER_CASE",
            cross_protocol_host_comparison="DO_NOT_POOL_OR_DIRECTLY_COMPARE_DISTINCT_OR_UNSPECIFIED_PROTOCOLS")
    for field in fields:
        if (not isinstance(session.get(field), str) or not session[field] or
                session.get(field) != manifest.get(field)):
            raise ValueError("host timing protocol session/manifest differs or is incomplete: " + field)
    if session.get("call_scope") != session["group_call_ns_scope"]:
        raise ValueError("session call_scope must equal the recorded group_call_ns_scope")
    guarded = session["runtime_protocol_id"] == "t6_tiled_p3t1_pl_epoch_guard_v1"
    if guarded:
        if session["host_timing_guard_version"] != "p3t1_pl_epoch_guard_v1":
            raise ValueError("guarded runtime protocol/guard version pair differs")
        for field, count in (("individual_invoke_identity_checks_inside_call_ns", 2),
                             ("invocation_identity_checks_inside_group_total_ns", 4)):
            if type(manifest.get(field)) is not int or manifest[field] != count:
                raise ValueError("guarded host timing identity-check count differs: " + field)
        for field in ("fixture_selection_scope", "host_timing_comparability"):
            if (not isinstance(session.get(field), str) or not session[field] or
                    session[field] != manifest.get(field)):
                raise ValueError("guarded host timing scope metadata differs: " + field)
    result = {field: session[field] for field in fields}
    result.update(status="RECORDED_GUARDED_HOST_PROTOCOL" if guarded else "RECORDED_OTHER_PROTOCOL",
        call_scope=session["call_scope"],
        individual_invoke_identity_checks_inside_call_ns=manifest.get("individual_invoke_identity_checks_inside_call_ns"),
        invocation_identity_checks_inside_group_total_ns=manifest.get("invocation_identity_checks_inside_group_total_ns"),
        fixture_selection_scope=session.get("fixture_selection_scope"),
        host_timing_comparability=manifest.get("host_timing_comparability"),
        host_statistics_pooling="NOT_PERFORMED_SINGLE_SELECTED_RUN_PER_CASE",
        cross_protocol_host_comparison="DO_NOT_POOL_OR_DIRECTLY_COMPARE_DISTINCT_OR_UNSPECIFIED_PROTOCOLS")
    return result


def validate_timing_snapshot(snapshot, package):
    """Replay frozen numerical gates and preserve separately declared host protocol."""
    result = _validate_timing_snapshot_without_runtime_protocol(snapshot, package)
    output = Path(snapshot).resolve() / "evidence/output"
    result["runtime_protocol"] = timing_protocol(read_json(output / "session.json", {}),
                                                read_json(output / "timing_manifest.json", {}))
    return result


def compare_imported_timing_audit(stored, replay):
    """Keep old imports exact, allowing existing local paths to the same directory."""
    stored = dict(stored)
    aliases = []
    for key in ("snapshot", "output"):
        left, right = stored.get(key), replay.get(key)
        if isinstance(left, str) and isinstance(right, str) and left != right:
            old_path, current_path = Path(left), Path(right)
            if (old_path.is_dir() and current_path.is_dir() and
                    old_path.resolve(strict=True) == current_path.resolve(strict=True) and
                    old_path.samefile(current_path)):
                aliases.append(dict(field=key, stored=left, replay=right,
                                    resolved=current_path.resolve(strict=True).as_posix()))
                stored[key] = right
    if "runtime_protocol" not in stored:
        replay = {key: value for key, value in replay.items() if key != "runtime_protocol"}
    result = compare_replayed_summary(stored, replay)
    if aliases:
        result["verified_local_directory_aliases"] = aliases
    return result

