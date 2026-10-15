"""Read-only evidence collection and versioned T6 coverage reports.

65 exact primitive identities receive software/HLS gates. Only the 13 bound
representatives receive route/board gates. Missing, changed or failed evidence
never becomes PASS. Raw environment eligibility and timing values are kept.
This module performs no build, remote operation, board action or cost admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

from . import board_runner as board
from .remote_worker import hls_resource_fit
from .timing_validation import compare_imported_timing_audit, timing_protocol, validate_timing_snapshot


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hardware_template_coverage_20261013"
DEFAULT = BASE / "attempt01"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _public_report(value):
    """Private lock ownership credentials must not enter derived reports."""
    if isinstance(value, dict):
        return {key: _public_report(item) for key, item in value.items()
                if "owner_token" not in str(key).lower()}
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    return value


def child(root, name):
    return board._child(Path(root), name)


def _bindings(attempt, manifest, inventory, issues):
    try:
        if manifest.get("schema") != "t6_tiled_source_v1" or len(manifest.get("cases", [])) != 65:
            raise ValueError("manifest requires exactly 65 T6 cases")
        if manifest.get("files", {}).get("inventory.json") != sha(attempt / "inventory.json"):
            raise ValueError("inventory/13-case whitelist is not bound to source manifest")
        board._checked_files(attempt, manifest["files"])
        expected = {row["case_id"]: row["case_sha256"] for row in manifest["cases"]}
        if len(expected) != 65 or set(expected) != {case["case_id"] for case in inventory["pilot_cases"]}:
            raise ValueError("manifest and inventory IDs differ")
        for name, identity in expected.items():
            case = board._case_identity(attempt / "cases" / name)
            if case["case_sha256"] != identity:
                raise ValueError("canonical case identity differs: " + name)
        return "PASS_SOURCE_BINDINGS"
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("Source binding: " + str(error))
        return "FAIL_SOURCE_BINDINGS"


def _snapshot(attempt, issues):
    pointer = load(attempt / "remote_evidence/latest_fetch.json", {})
    if not pointer:
        return None, {}, "NOT_FETCHED"
    try:
        directory = Path(pointer["directory"]).resolve()
        if not directory.is_relative_to((attempt / "remote_evidence").resolve()):
            raise ValueError("remote fetch pointer escapes this attempt")
        archive = directory / Path(pointer["archive"]).name
        if sha(archive) != pointer["sha256"]:
            raise ValueError("remote fetch archive SHA differs")
        manifest = load(directory / "fetch_manifest.json", {})
        board._checked_files(directory, manifest.get("files"))
        if len(manifest["files"]) != pointer["files"]:
            raise ValueError("remote fetch file count differs")
        return directory, manifest["files"], "PASS_FETCH_BINDINGS"
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("Remote snapshot: " + str(error))
        return None, {}, "FAIL_FETCH_BINDINGS"


def _stage(snapshot, files, case, name, source_sha, sources_ok, issues):
    result = dict(status="NOT_RUN", evidence_directory=None, record={})
    if snapshot is None:
        return result
    relative = f"results/{case['case_id']}/{name}/latest.json"
    if relative not in files:
        return result
    path = child(snapshot, relative)
    try:
        record = load(path, {})
        if not sources_ok or record.get("case_sha256") != case["case_sha256"] or record.get("source_manifest_sha256") != source_sha:
            raise ValueError("stage does not bind this case and source manifest")
        directory = child(snapshot, record["evidence_directory"])
        prefix = record["evidence_directory"].rstrip("/") + "/"
        result.update(status=record.get("status", "UNKNOWN"), record=record,
                      evidence_directory=directory.as_posix(), status_source=path.as_posix(),
                      status_sha256=sha(path), bound_files={name[len(prefix):]: digest
                          for name, digest in files.items() if name.startswith(prefix)})
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["status"] = "STALE_OR_INVALID_BINDING"
        issues.append(f"{case['case_id']} {name}: {error}")
    return result


def _native(case, record, summary, source_sha, sources_ok, folder, issues):
    result = dict(status=record.get("status", "NOT_RUN"), fixtures=None, elements=None,
                  compile_seconds=record.get("compile_seconds"), run_seconds=record.get("run_seconds"))
    if result["status"] != "PASS":
        return result
    try:
        if not sources_ok or summary.get("source_manifest_sha256") != source_sha or record.get("case_sha256") != case["case_sha256"]:
            raise ValueError("native summary/source/case identity differs")
        expected_elements = sum(row["expected_elements"] for row in case["fixtures"])
        tested = record.get("result", {})
        if (record.get("compile_returncode") != 0 or record.get("returncode") != 0 or
                record.get("binary_sha256_parity") is not True or
                tested.get("fixtures") != 5 or tested.get("mismatches") != 0 or tested.get("elements") != expected_elements):
            raise ValueError("native counts/returncodes/full output parity differ")
        for fixture in case["fixtures"]:
            if sha(folder / (fixture["fixture_id"] + "_actual.bin")) != fixture["expected_output_sha256"]:
                raise ValueError("native actual binary differs: " + fixture["fixture_id"])
        result.update(fixtures=5, elements=expected_elements)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["status"] = "FAIL_NATIVE_EVIDENCE"
        issues.append(case["case_id"] + " native: " + str(error))
    return result


def _hls(case, stage, issues):
    result = dict(process_status=stage["status"], csim="NOT_RUN", synthesis="NOT_RUN",
                  resource_fit="NOT_RUN", resources={}, evidence_sources={})
    if not stage.get("evidence_directory") or stage["status"] == "STALE_OR_INVALID_BINDING":
        return result
    directory = Path(stage["evidence_directory"])
    bound_files = stage.get("bound_files", {})
    for filename in ("csim_parity.json", "c_result.json", "pilot_kernel_csynth.xml", "hls_resource_fit.json"):
        if (directory / filename).is_file() and filename not in bound_files:
            result.update(csim="INVALID_UNBOUND_EVIDENCE", synthesis="INVALID_UNBOUND_EVIDENCE", resource_fit="INVALID_UNBOUND_EVIDENCE")
            issues.append(case["case_id"] + " HLS: unbound extra evidence file " + filename)
            return result
    parity, tested = load(directory / "csim_parity.json", {}), load(directory / "c_result.json", {})
    fixtures = {row["fixture_id"]: row for row in case["fixtures"]}
    witnesses = parity.get("fixtures", [])
    if parity:
        try:
            if parity.get("status") != "PASS" or len(witnesses) != 5 or {row.get("fixture") for row in witnesses} != set(fixtures):
                raise ValueError("HLS C-simulation hash witnesses incomplete")
            if tested.get("fixtures") != 5 or tested.get("mismatches") != 0 or tested.get("elements") != sum(row["expected_elements"] for row in fixtures.values()):
                raise ValueError("HLS C-simulation element/count parity differs")
            for witness in witnesses:
                expected = fixtures[witness["fixture"]]
                if (witness.get("expected_sha256") != expected["expected_output_sha256"] or
                        witness.get("actual_sha256") != expected["expected_output_sha256"] or
                        witness.get("bytes") != expected["expected_elements"] * case["output_bits"] // 8):
                    raise ValueError("HLS witness is not the exact full fixture output")
            result["csim"] = "PASS"
        except (ValueError, KeyError, TypeError) as error:
            result["csim"] = "FAIL_CSIM_EVIDENCE"
            issues.append(case["case_id"] + " csim: " + str(error))
        result["evidence_sources"]["csim"] = str(directory / "csim_parity.json")
    xml = directory / "pilot_kernel_csynth.xml"
    if xml.is_file():
        try:
            tree = ET.parse(xml)
            fit = hls_resource_fit(tree)
            stored = load(directory / "hls_resource_fit.json", {})
            gate_fit = stage["record"].get("gate", {}).get("resource_fit", {})
            if stored != fit or gate_fit != fit:
                raise ValueError("HLS resource-fit record differs from raw XML replay")
            result.update(synthesis="PASS", resource_fit=fit["status"], resources=fit,
                          evidence_sources=dict(result["evidence_sources"], synthesis=xml.as_posix(),
                                                resource_fit=str(directory / "hls_resource_fit.json")))
        except (OSError, ValueError, KeyError, ET.ParseError, TypeError) as error:
            result["synthesis"] = "INVALID_SYNTHESIS_EVIDENCE"
            result["resource_fit"] = "INVALID_RESOURCE_FIT"
            issues.append(case["case_id"] + " HLS resource fit: " + str(error))
    return result


def route_resources(path):
    """Vivado whole-shell resource use; Block RAM Tile is a 36K equivalent."""
    if not path.is_file():
        return {}
    result = {}
    names = {"CLB LUTs": "LUT", "CLB Registers": "FF", "DSPs": "DSP", "Block RAM Tile": "BRAM36_equivalent"}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [v.strip() for v in line.strip().split("|")]
        if len(cells) >= 4 and cells[1].rstrip("*").strip() in names and re.fullmatch(r"[\d,.]+", cells[2]):
            result.setdefault(names[cells[1].rstrip("*").strip()], float(cells[2].replace(",", "")))
    if "BRAM36_equivalent" in result:
        result["BRAM18"] = result["BRAM36_equivalent"] * 2
    return result


def _route(stage, hls, selected, issues, case_id):
    result = dict(status="NOT_RUN" if selected else "NOT_REQUIRED_REPRESENTATIVE_ONLY",
                  bitstream="NOT_GENERATED", bitstream_sha256=None, resources={}, WNS_ns=None)
    if not stage.get("evidence_directory"):
        if stage["status"] != "NOT_RUN": result["status"] = stage["status"]
        return result
    result["status"] = stage["status"]
    if stage["status"] != "PASS":
        return result
    try:
        if not selected or hls["process_status"] != "PASS" or hls["resource_fit"] != "FIT" or hls["csim"] != "PASS":
            raise ValueError("route must be a selected representative with passed exact HLS gates")
        directory = Path(stage["evidence_directory"])
        for filename in ("shell_metadata.json", "bitstream_manifest.json", "utilization.rpt", "timing.rpt"):
            if filename not in stage.get("bound_files", {}):
                raise ValueError("route report is absent from fetch manifest: " + filename)
        metadata = load(directory / "shell_metadata.json", {})
        if (metadata.get("schema_version") != 3 or metadata.get("scope") != "TEMPLATE_COVERAGE_ONLY" or
                metadata.get("part") != "xczu7ev-ffvc1156-2-e" or metadata.get("phase_counter_abi") != "P3T1" or
                metadata.get("phase_contract") != board.PHASE_CONTRACT or metadata.get("route_timing_drc") != "PASS" or
                metadata.get("clock_hz") != 187500000 or metadata.get("routing_errors") != 0 or
                metadata.get("routable_nets", 0) <= 0 or metadata.get("routable_nets") != metadata.get("fully_routed_nets")):
            raise ValueError("route metadata ABI/timing/routing coverage differs")
        for name in ("WNS_ns", "WHS_ns", "WPWS_ns"):
            if not math.isfinite(float(metadata[name])) or metadata[name] < 0:
                raise ValueError("negative/nonfinite timing slack")
        artifacts = stage["record"].get("gate", {}).get("artifacts", [])
        bit = [row for row in artifacts if row.get("path") == "vivado_shell/top.bit"]
        stored = load(directory / "bitstream_manifest.json", {}).get("files", [])
        if len(bit) != 1 or stored != artifacts or not re.fullmatch(r"[0-9a-f]{64}", bit[0].get("sha256", "")):
            raise ValueError("route/bitstream identity is not exact")
        result.update(bitstream="GENERATED_REMOTE_SHA_BOUND", bitstream_sha256=bit[0]["sha256"],
                      resources=route_resources(directory / "utilization.rpt"),
                      WNS_ns=metadata["WNS_ns"], WHS_ns=metadata["WHS_ns"], WPWS_ns=metadata["WPWS_ns"],
                      nominal_clock_hz=metadata["clock_hz"], evidence_sources=dict(
                          metadata=str(directory / "shell_metadata.json"), utilization=str(directory / "utilization.rpt"),
                          timing=str(directory / "timing.rpt"), bitstream_manifest=str(directory / "bitstream_manifest.json")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["status"] = "FAIL_ROUTE_EVIDENCE"
        issues.append(case_id + " route: " + str(error))
    return result


def _board(attempt, folder, case, route, selected, issues):
    result = dict(correctness="NOT_RUN" if selected else "NOT_REQUIRED_REPRESENTATIVE_ONLY",
                  timing="NOT_RUN" if selected else "NOT_REQUIRED_REPRESENTATIVE_ONLY",
                  import_audit="NOT_RUN", protocol_conditions_met=None, hardware_measurement_eligible=None,
                  measured_calls=None, independent_batches=None, stats=None, condition_deviations=[])
    pointer = load(folder / "board/selected_run.json", {})
    if not pointer:
        return result
    try:
        if not selected or pointer.get("evidence_kind") != "BOARD_MEASUREMENT":
            raise ValueError("board measurement is not an authorized selected representative")
        output = child(folder / "board", pointer["run_directory"])
        imported = load(output / "verified_snapshot_import.json", {})
        snapshot = Path(imported["snapshot"]).resolve()
        if not snapshot.is_relative_to((attempt / "board_snapshots").resolve()):
            raise ValueError("board snapshot points outside the current attempt")
        summary = load(output / "summary.json", {})
        candidates = []
        for case_path in (attempt / "board_packages" / case["case_id"]).glob("*/case/case.json"):
            package = case_path.parent
            if (load(case_path, {}).get("case_sha256") == case["case_sha256"] and
                    (package / "board_bundle.json").is_file() and
                    sha(package / "board_bundle.json") == summary.get("board_bundle_sha256")):
                candidates.append(package)
        if len(candidates) != 1:
            raise ValueError("one matching board package is required")
        package = candidates[0]
        if route["status"] != "PASS" or sha(package / "vivado_shell/top.bit") != route["bitstream_sha256"]:
            raise ValueError("board bitstream differs from passed source-bound route")
        board._case_identity(package)
        board._route_identity(package)
        checked = validate_timing_snapshot(snapshot, package)
        if checked["case_sha256"] != case["case_sha256"] or checked["case_id"] != case["case_id"]:
            raise ValueError("replayed board snapshot binds a different case")
        snapshot_manifest = load(snapshot / "evidence/snapshot_manifest.json", {})
        copied_files = {name[len("output/"):]: digest for name, digest in snapshot_manifest["files"].items()
                        if name.startswith("output/")}
        board._checked_files(output, copied_files)
        compare_imported_timing_audit(imported, checked)
        if imported.get("summary_sha256") != sha(output / "summary.json"):
            raise ValueError("selected summary changed since verified import")
        case_summary = next(row for row in summary["cases"] if row["case_id"] == case["case_id"])
        result.update(correctness="PASS_15_FULL_FIXTURES", timing=case_summary["status"],
                      import_audit=checked["status"], protocol_conditions_met=summary.get("protocol_conditions_met"),
                      hardware_measurement_eligible=summary.get("hardware_measurement_eligible"),
                      phase_clock_evidence_verified=summary.get("phase_clock_evidence_verified"),
                      condition_deviations=summary.get("condition_deviations", []), stats=case_summary["scopes"],
                      measured_calls=case_summary["measured_rows"], independent_batches=case_summary["independent_batches"],
                      host_median_stability=case_summary["host_median_stability"],
                      cpu_governor_observations=checked["cpu_governor_observations"],
                      actual_clock_hz=[row["clock_hz"] for row in load(output / "timing_manifest.json")["batches"]],
                      summary_source=str(output / "summary.json"), raw_source=str(output / "timing_raw.csv"),
                      package=package.as_posix(), snapshot=snapshot.as_posix(), summary_sha256=sha(output / "summary.json"),
                      raw_csv_sha256=sha(output / "timing_raw.csv"), replay_comparison=checked["replay_comparison"])
        result["runtime_protocol"] = checked["runtime_protocol"]
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ET.ParseError) as error:
        result.update(correctness="FAIL_BOARD_EVIDENCE", timing="UNACCEPTED", import_audit="FAIL")
        issues.append(case["case_id"] + " board: " + str(error))
    return result


def _stage_rows(case_id, native, hls, route, case):
    rows = [dict(case_id=case_id, stage="python_reference_and_fixture_preparation", seconds=case.get("preparation_wall_seconds"),
                 evidence_layer="SOFTWARE", source="case.json", status="RECORDED"),
            dict(case_id=case_id, stage="native_compile", seconds=native.get("compile_seconds"), evidence_layer="SOFTWARE", status=native["status"]),
            dict(case_id=case_id, stage="native_run", seconds=native.get("run_seconds"), evidence_layer="SOFTWARE", status=native["status"])]
    for name, stage in (("hls", hls), ("route", route)):
        record = stage.get("record", {})
        if record:
            rows.append(dict(case_id=case_id, stage=name + "_process_envelope", seconds=record.get("wall_seconds"),
                             evidence_layer="PARENT_ENVELOPE_DO_NOT_SUM_WITH_SUBSTAGES", status=stage["status"],
                             source=stage.get("status_source")))
        for row in record.get("substage_records", []):
            if name == "route" and row.get("stage") in ("csim", "hls_synthesis", "ip_export"):
                continue  # A route export can repeat its preceding HLS log.
            rows.append(dict(case_id=case_id, stage=row.get("stage"), seconds=row.get("wall_seconds"),
                status="PASS" if row.get("returncode") == 0 else "FAIL", evidence_layer=name.upper(),
                source=stage.get("evidence_directory")))
    return rows


def _verified_local_snapshot(snapshot):
    """Bind history to the downloaded archive and its extracted bytes."""
    snapshot = Path(snapshot)
    evidence = snapshot / "evidence"
    archive = snapshot / "snapshot.tar.gz"
    downloaded = load(snapshot / "download.json", {})
    manifest = load(evidence / "snapshot_manifest.json", {})
    if (downloaded.get("ok") is not True or downloaded.get("archive_sha256") != sha(archive) or
            downloaded.get("archive_bytes") != archive.stat().st_size or
            manifest.get("schema") != "pilot_board_snapshot_v1"):
        raise ValueError("local snapshot archive/download binding differs")
    board._checked_files(evidence, manifest.get("files"))
    # The extracted manifest must itself be the member of the bound archive.
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or any(not member.isfile() for member in members) or
                set(names) != set(manifest["files"]) | {"snapshot_manifest.json"}):
            raise ValueError("local snapshot archive member set differs")
        for member in members:
            path = child(evidence, member.name)
            digest = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            expected = sha(path) if member.name == "snapshot_manifest.json" else manifest["files"][member.name]
            if digest != expected:
                raise ValueError("local snapshot archived member SHA differs: " + member.name)
    for name in ("request.json", "status.json"):
        if name not in manifest["files"]:
            raise ValueError("local snapshot requires bound request/status")
    request, status = load(evidence / "request.json", {}), load(evidence / "status.json", {})
    if (manifest.get("job_id") != request.get("job_id") or manifest.get("job_id") != status.get("job_id") or
            manifest.get("observed_state") != status.get("state") or
            downloaded.get("observed_state") != status.get("state")):
        raise ValueError("local snapshot job/state identities differ")
    return evidence, manifest, request, status


def _recorded_call_counts(evidence, manifest, case):
    """Count preserved rows only; never calculate accepted performance statistics."""
    filename = "output/timing_raw.csv"
    if filename not in manifest["files"]:
        return []
    counts = {}
    with (evidence / filename).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("case_id") != case["case_id"] or row.get("case_sha256") != case["case_sha256"]:
                raise ValueError("historical CSV case identity differs")
            phase = row.get("phase")
            if phase not in ("warmup", "measured"):
                raise ValueError("historical CSV phase is unknown")
            batch = counts.setdefault(row["batch_id"], dict(batch_id=row["batch_id"], warmup_rows=0,
                measured_rows=0, completed_measured_rows=0, incomplete_measured_rows=0,
                incomplete_iterations=[]))
            batch[phase + "_rows"] += 1
            if phase == "measured":
                complete = (all(row.get(key) not in (None, "") for key in
                    ("compute_ns", "transfer_ns", "compared_elements", "mismatched_elements", "observed_output_sha256")) and
                    row.get("mismatched_elements") == "0" and
                    row.get("observed_output_sha256") == row.get("expected_output_sha256"))
                batch["completed_measured_rows" if complete else "incomplete_measured_rows"] += 1
                if not complete:
                    batch["incomplete_iterations"].append(int(row["iteration"]))
    return list(counts.values())


def _board_history(attempt, rows, issues):
    """Keep unselected probes, failed runs and quarantines visible without admission."""
    cases = {row["case_id"]: row for row in rows}
    histories = []
    directory = attempt / "board_snapshots"
    if not directory.is_dir():
        return histories
    for snapshot in sorted(directory.iterdir()):
        if not snapshot.is_dir():
            continue
        result = dict(snapshot=snapshot.as_posix(), snapshot_name=snapshot.name,
            binding="INVALID_SNAPSHOT_BINDING", case_id=None, job_id=None, mode=None,
            observed_state=None, selected_for_report=False, accepted_for_pass_or_aggregate=False,
            recorded_calls=[], statistics=None, formal_cost_eligible=False)
        try:
            if not snapshot.resolve().is_relative_to(directory.resolve()) or snapshot.is_symlink():
                raise ValueError("local snapshot escapes board_snapshots")
            evidence, manifest, request, status = _verified_local_snapshot(snapshot)
            config = request.get("config", {})
            case = cases.get(config.get("case_id"))
            if not case or case["case_sha256"] != config.get("case_sha256"):
                raise ValueError("local snapshot does not bind a current exact case")
            selected = case["board"].get("snapshot") == snapshot.resolve().as_posix()
            result.update(binding="BOUND_LOCAL_SNAPSHOT", case_id=case["case_id"],
                case_sha256=case["case_sha256"], job_id=request["job_id"], mode=config.get("mode"),
                observed_state=status["state"], selected_for_report=selected,
                archive_sha256=sha(snapshot / "snapshot.tar.gz"),
                snapshot_manifest_sha256=sha(evidence / "snapshot_manifest.json"),
                raw_files_sha256=manifest["files"], recorded_calls=_recorded_call_counts(evidence, manifest, case),
                started_utc=status.get("started_utc"), updated_utc=status.get("updated_utc"))
            if all(name in manifest["files"] for name in ("output/session.json", "output/timing_manifest.json")):
                result["runtime_protocol"] = timing_protocol(load(evidence / "output/session.json", {}),
                    load(evidence / "output/timing_manifest.json", {}))
            failure_name = "output/failure.json"
            failure = load(evidence / failure_name, {}) if failure_name in manifest["files"] else {}
            result["failure"] = {key: failure.get(key) for key in ("status", "error_type", "error") if key in failure}
            result["classification"] = ("SELECTED_RUN_ALREADY_COUNTED_SEPARATELY" if selected else
                "DMA_QUARANTINED_INVALID_RUN" if status["state"] == "DMA_QUARANTINED" else
                "PROBE_ONLY_NOT_TIMING_ACCEPTED" if config.get("mode") == "probe" else
                "UNSELECTED_RUN_NOT_ACCEPTED")
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
            result.update(binding="INVALID_SNAPSHOT_BINDING", recorded_calls=[], statistics=None)
            result.pop("runtime_protocol", None)
            result["binding_error"] = str(error)
            issues.append("Board history " + snapshot.name + ": " + str(error))
        histories.append(result)
    for case in rows:
        history = [item for item in histories if item["case_id"] == case["case_id"] and not item["selected_for_report"]]
        case["board"]["unselected_history"] = history
        case["board"]["history_status"] = "UNSELECTED_ATTEMPTS_PRESENT" if history else "NO_UNSELECTED_ATTEMPT_OBSERVED"
    return histories


def _incident_history(attempt, issues):
    incidents = []
    for path in sorted((attempt / "board_incidents").glob("*/incident.json")):
        item = dict(path=path.as_posix(), binding="INVALID_INCIDENT_BINDING")
        try:
            manifest = load(path.parent / "evidence_manifest.json", {})
            if manifest.get("schema") != "t6_local_incident_evidence_v1":
                raise ValueError("incident evidence manifest schema differs")
            board._checked_files(attempt, manifest.get("files"))
            if path.relative_to(attempt).as_posix() not in manifest["files"]:
                raise ValueError("incident record is absent from its evidence manifest")
            record = load(path, {})
            item.update(binding="BOUND_LOCAL_INCIDENT", incident_sha256=sha(path),
                evidence_manifest_sha256=sha(path.parent / "evidence_manifest.json"),
                case_id=record.get("case_id"), job_id=record.get("job_id"),
                conclusion=record.get("conclusion"), run_valid=record.get("run_valid"),
                accepted_for_pass_or_aggregate=False, timeline=record.get("timeline", []),
                protective_action=record.get("protective_action", {}))
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["binding_error"] = str(error)
            issues.append("Board incident " + path.parent.name + ": " + str(error))
        incidents.append(item)
    return incidents


def collect(attempt=DEFAULT):
    attempt = Path(attempt).resolve()
    inventory = load(attempt / "inventory.json", {})
    if inventory.get("schema_version") != "hardware_template_coverage_inventory_v1" or inventory.get("pilot_case_count") != 65:
        raise ValueError("report requires the current exact 65-case T6 inventory")
    selected = set(inventory.get("representative_case_ids", []))
    if len(selected) != 13:
        raise ValueError("report requires the bound 13-case representative set")
    issues = []
    manifest = load(attempt / "source_manifest.json", {})
    source_sha = sha(attempt / "source_manifest.json")
    binding = _bindings(attempt, manifest, inventory, issues)
    snapshot, fetched_files, fetch_binding = _snapshot(attempt, issues)
    native_summary = load(attempt / "native_summary.json", {})
    native_rows = {row["case_id"]: row for row in native_summary.get("cases", [])}
    contract = load(attempt.parent / "coverage_contract.json", {})
    contract_rows = {row["primitive_id"]: row for row in contract.get("rows", [])}
    if set(contract_rows) != {row["primitive_id"] for row in inventory["pilot_cases"]}:
        issues.append("Coverage contract primitive IDs differ; storage/history annotations are unaccepted")
        contract_rows = {}
    rows, stages, commands = [], [], []
    for planned in inventory["pilot_cases"]:
        name = planned["case_id"]
        folder = attempt / "cases" / name
        case = load(folder / "case.json", planned)
        native = _native(case, native_rows.get(name, {}), native_summary, source_sha, binding == "PASS_SOURCE_BINDINGS", folder, issues)
        hls_stage = _stage(snapshot, fetched_files, case, "hls", source_sha, binding == "PASS_SOURCE_BINDINGS", issues)
        route_stage = _stage(snapshot, fetched_files, case, "route", source_sha, binding == "PASS_SOURCE_BINDINGS", issues)
        hls = _hls(case, hls_stage, issues)
        routed = _route(route_stage, hls, name in selected, issues, name)
        tested = _board(attempt, folder, case, routed, name in selected, issues)
        history = contract_rows.get(case["primitive_id"], {})
        rows.append(dict(case_id=name, primitive_id=case["primitive_id"], family=case["family"],
            shape=case["shape"], precision={"input_bits": case["input_bits"], "weight_bits": case["weight_bits"], "output_bits": case["output_bits"]},
            mapping=case["mapping"], representative=name in selected, case_sha256=case.get("case_sha256"),
            native=native, hls=hls, route=routed, board=tested,
            historical_assessment=history.get("historical_assessment"),
            historical_full_frame_bram18_lower_bound=history.get("historical_full_frame_bram18_lower_bound"),
            historical_existing_template_exceeds=history.get("historical_existing_template_exceeds"),
            historical_missing_template=history.get("historical_missing_template"),
            declared_arrays=history.get("proposed_tiled_storage", {}), formal_cost_eligible=False))
        stages.extend(_stage_rows(name, native, hls_stage, route_stage, case))
        for stage in (hls_stage, route_stage):
            record = stage.get("record", {})
            if record.get("command"):
                commands.append(dict(kind="RECORDED_TOOL_COMMAND", case_id=name, stage=record["stage"],
                                     argv=record["command"], cwd=record.get("cwd"), source=stage.get("status_source")))
    for path in (attempt / "board_execution").glob("*/inputs.json"):
        execution = load(path, {})
        if execution.get("case_sha256") != next((row["case_sha256"] for row in rows
                if row["case_id"] == execution.get("case_id")), None):
            issues.append("Board execution command refers to a different case: " + str(path))
            continue
        if execution.get("command"):
            commands.append(dict(kind="RECORDED_BOARD_DRIVER_COMMAND", case_id=execution["case_id"],
                argv=execution["command"], cwd=execution.get("cwd"), source=str(path)))
        else:
            commands.append(dict(kind="BOARD_COMMAND_NOT_RECORDED", case_id=execution.get("case_id"),
                package=execution.get("package"), source=str(path),
                limitation="Execution inputs bind hashes but no exact argv; do not label a reconstruction as the executed command"))
    board_history = _board_history(attempt, rows, issues)
    incidents = _incident_history(attempt, issues)
    counts = dict(primitive_count=len(rows), representative_count=len(selected),
        native_pass=sum(row["native"]["status"] == "PASS" for row in rows),
        native_fixture_count=sum(row["native"].get("fixtures") or 0 for row in rows),
        native_compared_elements=sum(row["native"].get("elements") or 0 for row in rows),
        hls_csim_pass=sum(row["hls"]["csim"] == "PASS" for row in rows),
        hls_synthesis_pass=sum(row["hls"]["synthesis"] == "PASS" for row in rows),
        hls_resource_fit=sum(row["hls"]["resource_fit"] == "FIT" for row in rows),
        hls_resource_oversized=sum(row["hls"]["resource_fit"] == "OVERSIZED" for row in rows),
        hls_process_pass=sum(row["hls"]["process_status"] == "PASS" for row in rows),
        route_pass=sum(row["route"]["status"] == "PASS" for row in rows),
        board_full_fixture_pass=sum(row["board"]["correctness"] == "PASS_15_FULL_FIXTURES" for row in rows),
        board_timing_pass=sum(row["board"]["timing"] == "PASS" for row in rows),
        board_measurement_eligible=sum(row["board"]["hardware_measurement_eligible"] is True for row in rows),
        board_measured_calls=sum(row["board"].get("measured_calls") or 0 for row in rows),
        board_independent_batches=sum(row["board"].get("independent_batches") or 0 for row in rows))
    complete = all(counts[name] == expected for name, expected in (
        ("native_pass", 65), ("hls_csim_pass", 65), ("hls_synthesis_pass", 65),
        ("hls_resource_fit", 65), ("hls_process_pass", 65), ("route_pass", 13),
        ("board_full_fixture_pass", 13), ("board_timing_pass", 13))) and not issues
    status = ("IMPLEMENTATION_GATES_COMPLETE_WITH_PROTOCOL_DEVIATIONS" if complete and counts["board_measurement_eligible"] != 13
              else "IMPLEMENTATION_GATES_COMPLETE" if complete else "INCOMPLETE")
    sources = {"inventory": {"path": str(attempt / "inventory.json"), "sha256": sha(attempt / "inventory.json")},
               "source_manifest": {"path": str(attempt / "source_manifest.json"), "sha256": source_sha},
               "generator": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}}
    if (attempt.parent / "coverage_contract.json").is_file():
        sources["coverage_contract"] = dict(path=str(attempt.parent / "coverage_contract.json"), sha256=sha(attempt.parent / "coverage_contract.json"))
    if (attempt / "native_summary.json").is_file():
        sources["native_summary"] = dict(path=str(attempt / "native_summary.json"), sha256=sha(attempt / "native_summary.json"))
    if snapshot is not None:
        sources["remote_fetch"] = dict(path=snapshot.as_posix(), manifest_sha256=sha(snapshot / "fetch_manifest.json"))
    return dict(schema="t6_coverage_evidence_report_v1", status=status, attempt=attempt.as_posix(),
        generated_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        source_binding=binding, remote_fetch_binding=fetch_binding, counts=counts,
        counts_by_family=dict(Counter(row["family"] for row in rows)),
        history=dict(existing_template_exceeds_count=contract.get("historical_existing_template_exceeds_count"),
                     missing_template_count=contract.get("historical_absent_template_count"),
                     target_union_count=contract.get("targeted_primitive_union_count"),
                     physical_capacity_exceeds_count=contract.get("physical_full_frame_capacity_exceeds_count"),
                     absent_and_capacity_overlap=contract.get("absent_and_physical_capacity_overlap_count"),
                     available_bram18=contract.get("available_bram18")),
        d01_boundary=contract.get("d01_boundary", {}), rows=rows, stage_records=stages,
        recorded_commands=commands, board_snapshot_history=board_history,
        board_incidents=incidents, issues=issues, formal_cost_eligible=False,
        evidence_boundaries=["Native software parity is not Vitis HLS C simulation",
            "HLS estimates/FIT are not integrated routed fit", "Only 13 representatives have route and board gates",
            "Declared array bits are analytic lower bounds, not measured BRAM resources",
            "Raw protocol_conditions_met/hardware_measurement_eligible are never promoted",
            "Unselected probes/failed/quarantined snapshots remain history only; their partial samples never enter PASS or aggregate timing",
            "One selected run per exact case; host timing statistics are never pooled across runs or distinct/unspecified runtime protocols",
            "Synthetic fixtures/scale/ReLU6 cap are not trained weights/calibration",
            "65 applies only to the current domain; D01 may introduce shapes needing new validation",
            "No formal cost table, power measurement, whole-network route or trained-network effectiveness claim"])


def _fmt(value, divisor=1):
    if value is None: return "—"
    if isinstance(value, bool): return str(value).lower()
    return f"{value/divisor:.6g}" if isinstance(value, (int, float)) else str(value)


def _shape(shapes):
    return " + ".join("×".join(map(str, shape)) for shape in shapes)


def markdown(data):
    c, hist = data["counts"], data["history"]
    lines = ["# 待办6：分块模板覆盖与验证报告", "",
        f"生成时间（UTC）：{data['generated_utc']}。结果状态：**{data['status']}**。", "",
        f"当前65种原语：native（本地C++）{c['native_pass']}/65，HLS C仿真{c['hls_csim_pass']}/65，综合{c['hls_synthesis_pass']}/65，"
        f"HLS资源估计FIT（可容纳）{c['hls_resource_fit']}/65、OVERSIZED（超容量）{c['hls_resource_oversized']}/65。"
        f"13项代表：布线{c['route_pass']}/13，完整输出逐位一致{c['board_full_fixture_pass']}/13，计时统计{c['board_timing_pass']}/13。", "",
        f"软件比较共{c['native_fixture_count']}个fixture（测试输入），{c['native_compared_elements']:,}个输出元素；"
        f"验收计入的板上独立批次为{c['board_independent_batches']}个、计时调用为{c['board_measured_calls']:,}次；未获验收的probe、部分计时与隔离记录在历史表中另列。", "",
        "当前启用W8A8与HEAD_W8A16，所有卷积权重为INT8，INT32累加按模2³²回绕；INT64重定标使用nearest-even（中点取偶）并饱和。"
        "所有fixture保留原语真实linear/ReLU/ReLU6；ReLU6整数上限12与整数scale为合成验证参数，未导出训练权重或校准结果。", "",
        f"来源绑定：{data['source_binding']}；远程快照绑定：{data['remote_fetch_binding']}。", "",
        "## 完成边界", "",
        "65项全部完成native、HLS C仿真、综合/IP导出及HLS资源FIT，再完成13项代表的布线/bit/完整整数板测/计时，才可称当前实现验收完成。"
        "其余52项不因此获得独立板测或正式成本行；缺失和失败均保留其状态。", "",
        f"严格实验环境条件下hardware_measurement_eligible（硬件测量合格）为true的板测：{c['board_measurement_eligible']}/13。"
        "CPU policy（频率策略）实读、条件偏差及原始false保持不变，D09仍需在正式协议中处理。", "",
        "D01改变骨干/通道/算子/连接/预测头后，须重新生成清单并验证新增形状；当前65种覆盖只对应当前elastic_neck72_v1域。", "",
        "## 22项缺口分类与片上存储", "",
        f"历史assessment（评估分类）互斥：已有模板整帧超容量{hist.get('existing_template_exceeds_count')}种，缺模板{hist.get('missing_template_count')}种，"
        f"并集{hist.get('target_union_count')}种。按纯容量单独统计为{hist.get('physical_capacity_exceeds_count')}种，缺模板与容量超限重叠"
        f"{hist.get('absent_and_capacity_overlap')}种；器件容量{hist.get('available_bram18')}个BRAM18（18Kb块RAM）。", "",
        "新模板把整帧保存在DDR，默认输出块2×32，PW（逐点卷积）PE=4/SIMD=4，其余PE=4/SIMD=1。"
        "卷积输入patch（带邻域的局部块）尺寸为((TILE_H−1)×stride+K)×((TILE_W−1)×stride+K)。"
        "DW（逐通道卷积）、pool、resize、add、精度边界一次只缓冲PE个通道，普通卷积缓冲所有输入通道的一个空间块。", "",
        "下表数组容量包含声明缓冲/权重/bias/参数及已显式统计的累加lane寄存器。ceil(bits/18432)仅为理想BRAM18容量下界，"
        "未包括银行碎片、FIFO、AXI外壳和其他调度寄存器，不能替代综合或布线FIT。", "",
        "| case_id | 类别 | 输入CHW → 输出CHW | 历史整帧BRAM18下界 | 新声明数组bit数 | 理想BRAM18容量下界 |",
        "|---|---|---|---:|---:|---:|"]
    for row in data["rows"]:
        if not (row.get("historical_existing_template_exceeds") or row.get("historical_missing_template")): continue
        bits = row["declared_arrays"].get("total_declared_array_bits")
        label = "缺模板" if row.get("historical_missing_template") else "已有模板整帧超限"
        lines.append(f"| `{row['case_id']}` | {row['family']} / {label} | {_shape(row['shape']['input_shapes'])} → {_shape([row['shape']['output_shape']])} | "
                     f"{_fmt(row['historical_full_frame_bram18_lower_bound'])} | {_fmt(bits)} | {_fmt(math.ceil(bits/18432) if bits is not None else None)} |")
    lines += ["", "## 65项实现与验证矩阵", "",
        "| case_id | 类型/精度I-W-O | native | HLS C仿真 | 综合 | HLS资源FIT | 代表布线 | 代表完整板测 | 代表计时 | 未选板端历史 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for row in data["rows"]:
        p = row["precision"]
        lines.append(f"| `{row['case_id']}` | {row['family']} / {p['input_bits']}-{p['weight_bits'] or '无'}-{p['output_bits']} | "
            f"{row['native']['status']} | {row['hls']['csim']} | {row['hls']['synthesis']} | {row['hls']['resource_fit']} | "
            f"{row['route']['status']} | {row['board']['correctness']} | {row['board']['timing']} | "
            f"{'; '.join(item.get('classification', item['binding']) for item in row['board'].get('unselected_history', [])) or '—'} |")
    lines += ["", "## 未选attempt、probe与隔离历史", "",
        "矩阵中的NOT_RUN只表示尚无已选验收运行；实际执行但未选的probe（探针验证）、失败与DMA隔离如下。"
        "这些记录仅来自本地archive和文件SHA一致的快照；保留调用数量，不计算或合并其性能统计，也不增加PASS计数。", "",
        "| job_id / case_id | 模式 | 快照状态 / 绑定 | 保留的measured记录 | 分类 |",
        "|---|---|---|---|---|"]
    for item in data.get("board_snapshot_history", []):
        if item.get("selected_for_report"):
            continue
        calls = "; ".join(f"{batch['batch_id']}: complete={batch['completed_measured_rows']}, "
            f"incomplete={batch['incomplete_measured_rows']}, iterations={batch['incomplete_iterations']}"
            for batch in item.get("recorded_calls", [])) or "—"
        lines.append(f"| `{item.get('job_id') or item['snapshot_name']}` / `{item.get('case_id') or '未绑定'}` | "
            f"{item.get('mode') or '—'} | {item.get('observed_state') or '—'} / {item['binding']} | "
            f"{calls} | {item.get('classification', 'INVALID_SNAPSHOT_BINDING')} |")
    if not data.get("board_snapshot_history"):
        lines.append("| — | — | 无本地快照 | — | — |")
    for incident in data.get("board_incidents", []):
        lines.append(f"- 事件记录：[{Path(incident['path']).parent.name}]({incident['path']})；"
            f"{incident['binding']}；{incident.get('conclusion') or incident.get('binding_error')}；整run有效={_fmt(incident.get('run_valid'))}。")
    lines += ["", "## 主机计时协议与窗口", "",
        "每个case只采用一个已选运行，报告不跨run合并host（主机）计时。runtime protocol（运行协议）或guard（保护检查）版本不同、"
        "以及LEGACY_UNSPECIFIED（旧记录未声明协议）的call_ns不直接混合比较。新保护检查的软件开销包含在主机时间中；"
        "PL compute/transfer仍对应各自硬件阶段。旧artifact保持原字节与字段。", ""]
    protocols = []
    for row in data["rows"]:
        if row["board"].get("runtime_protocol"):
            protocols.append((row["case_id"], "已选运行", row["board"]["runtime_protocol"]))
    for item in data.get("board_snapshot_history", []):
        if not item.get("selected_for_report") and item.get("runtime_protocol"):
            protocols.append((item["case_id"], item["job_id"] + "（未选）", item["runtime_protocol"]))
    for case_id, label, protocol in protocols:
        lines += [f"- `{case_id}` / {label}：runtime_protocol_id=`{protocol['runtime_protocol_id']}`；"
            f"host_timing_guard_version=`{protocol['host_timing_guard_version']}`。",
            f"  individual_call_ns_scope：{protocol.get('individual_call_ns_scope') or '旧记录未声明'}。",
            f"  group_call_ns_scope / session.call_scope：{protocol.get('group_call_ns_scope') or '旧记录未声明'}。",
            f"  每次invoke窗口内检查：individual={_fmt(protocol.get('individual_invoke_identity_checks_inside_call_ns'))}，"
            f"group_total={_fmt(protocol.get('invocation_identity_checks_inside_group_total_ns'))}；"
            f"fixture选择范围：{protocol.get('fixture_selection_scope') or '旧记录未声明'}。"]
    if not protocols:
        lines.append("尚无已绑定的计时协议记录；未将新源码声明当成已执行运行。")
    lines += ["", "## 13项代表的资源、阶段计时与板端统计", "",
        "HLS资源是核心估计；布线资源是整套shell（硬件外壳），不混用。以下板端median（中位数）、CV（波动系数）直接由完整原始CSV复算，"
        "慢样本不裁剪。compute=phase2片上计算，transfer=phase1加载+phase3写回，call=主机每组总时间/N；固定阶段切换开销已包括且不扣除。", "",
        "| case_id | HLS LUT/DSP/BRAM18 | 布线LUT/DSP/BRAM18 | WNS ns | compute median ms / CV | transfer median ms / CV | call median ms | 原环境条件 |",
        "|---|---|---|---:|---|---|---:|---|"]
    for row in data["rows"]:
        if not row["representative"]: continue
        hls, route, tested = row["hls"], row["route"], row["board"]
        h = hls.get("resources", {}).get("used", {})
        r = route.get("resources", {})
        stats = tested.get("stats") or {}
        compute, transfer, call = (stats.get(name) or {} for name in ("compute", "transfer", "call"))
        lines.append(f"| `{row['case_id']}` | {_fmt(h.get('LUT'))}/{_fmt(h.get('DSP'))}/{_fmt(h.get('BRAM_18K'))} | "
            f"{_fmt(r.get('LUT'))}/{_fmt(r.get('DSP'))}/{_fmt(r.get('BRAM18'))} | {_fmt(route.get('WNS_ns'))} | "
            f"{_fmt(compute.get('median_ns'),1e6)} / {_fmt(compute.get('cv'))} | {_fmt(transfer.get('median_ns'),1e6)} / {_fmt(transfer.get('cv'))} | "
            f"{_fmt(call.get('median_ns'),1e6)} | protocol_conditions_met={_fmt(tested.get('protocol_conditions_met'))}; eligible={_fmt(tested.get('hardware_measurement_eligible'))} |")
    lines += ["", "| case_id | HLS流程总秒 | Vivado流程总秒 | 分阶段记录（秒） | 实际PL频率Hz |", "|---|---:|---:|---|---|"]
    for row in data["rows"]:
        if not row["representative"]: continue
        records = [r for r in data["stage_records"] if r["case_id"] == row["case_id"]]
        envelope = {r["stage"]: r.get("seconds") for r in records if r["stage"] in ("hls_process_envelope", "route_process_envelope")}
        substages = "; ".join(f"{r['stage']}={_fmt(r.get('seconds'))}" for r in records if r["evidence_layer"] in ("HLS", "ROUTE")) or "—"
        lines.append(f"| `{row['case_id']}` | {_fmt(envelope.get('hls_process_envelope'))} | {_fmt(envelope.get('route_process_envelope'))} | "
                     f"{substages} | {row['board'].get('actual_clock_hz') or '—'} |")
    lines += ["", "阶段总时长与子阶段存在包含关系，不相加重复计费；失败构建的已耗时照实记录。host（主机）计时组大小与跨批稳定性见JSON，"
              "未给主机CV追加未经冻结的默认验收门。PL频率来自PS时钟寄存器派生读数；独立物理频率校准未执行。", "", "## 命令与证据文件", "",
              "以下是当前attempt的可复现入口；远程及板操作只有在既定授权与源绑定验收满足后使用。本报告生成只读消费证据，不启动这些动作。", "",
              "```powershell", f"Set-Location '{ROOT.as_posix()}'",
              f"python -m tools.hardware_template_coverage_v1.prepare prepare --output '{data['attempt']}'",
              f"python -m tools.hardware_template_coverage_v1.prepare native --output '{data['attempt']}'",
              f"python -m tools.hardware_template_coverage_v1.remote deploy --attempt {Path(data['attempt']).name}",
              f"python -m tools.hardware_template_coverage_v1.parallel_build launch --attempt {Path(data['attempt']).name} --stage coverage --jobs 3 --limit 0",
              f"python -m tools.hardware_template_coverage_v1.remote fetch --attempt {Path(data['attempt']).name}",
              f"python -m tools.hardware_template_coverage_v1.coverage_report --attempt '{data['attempt']}'", "```", "",
              "prepare拒绝覆盖已存在attempt；复现实施时须指定新attempt。实际执行的Vitis/Vivado argv、cwd和记录路径逐项保存在report.json.recorded_commands。", ""]
    board_commands = [row for row in data["recorded_commands"] if row["kind"] == "RECORDED_BOARD_DRIVER_COMMAND"]
    if board_commands:
        lines += ["已执行板端流程的精确argv（JSON数组，保留路径与参数原值）：", "", "```json",
                  json.dumps(board_commands, indent=2, ensure_ascii=False), "```", ""]
    else:
        lines += ["板端流程命令尚未取得绑定记录，不将计划命令写成已执行命令。", ""]
    for source in data["sources"].values():
        lines.append(f"- [{Path(source['path']).name}]({source['path']})；SHA256 `{source.get('sha256') or source.get('manifest_sha256')}`。")
    for row in data["rows"]:
        tested = row["board"]
        if tested.get("raw_source"):
            lines.append(f"- `{row['case_id']}`：[完整原始CSV]({tested['raw_source']})；[原始summary]({tested['summary_source']})；"
                         f"CSV SHA256 `{tested['raw_csv_sha256']}`。")
    lines += ["", "## 未决项与限制", ""]
    if data["issues"]:
        lines.extend("- " + issue for issue in data["issues"])
    if data["status"] == "INCOMPLETE":
        lines.append("- 尚缺的验收门见65项矩阵；未取得证据的阶段不算通过。")
    lines.extend("- " + item for item in data["evidence_boundaries"])
    return "\n".join(lines) + "\n"


def generate(attempt=DEFAULT, output=None):
    data = _public_report(collect(attempt))
    output = Path(output).resolve() if output else BASE / "reports" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "report.json", data)
    (output / "report.md").write_text(markdown(data), encoding="utf-8")
    result = dict(status=data["status"], counts=data["counts"], output=output.as_posix(),
                  report_json_sha256=sha(output / "report.json"), report_md_sha256=sha(output / "report.md"),
                  formal_cost_eligible=False)
    save_new(output / "generation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.attempt, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
