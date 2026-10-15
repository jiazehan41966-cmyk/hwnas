"""Negative import/replay witnesses using a synthetic, explicitly non-board run."""
from copy import deepcopy
import csv
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from tools.hardware_measurement_pilot import report as legacy_report
from tools.hardware_template_coverage_v1 import board_runner as board
from tools.hardware_template_coverage_v1 import measurement
from tools.hardware_template_coverage_v1 import timing_validation as validator


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synthetic_snapshot(root):
    """No hardware observations: construct valid bindings to test rejection paths."""
    package = root / "case"
    output = root / "snapshot/evidence/output"
    package.mkdir(parents=True)
    output.mkdir(parents=True)
    for name in ("kernel.cpp", "config.h", "phase_timer.v"):
        (package / name).write_text("synthetic-validator-test-source\n")
    contract = {"evidence": "SYNTHETIC_UNIT_TEST_ONLY"}
    write_json(package / "numeric_contract.json", contract)
    fixtures = []
    for index in range(5):
        fid = f"test{index}"
        binary = bytes([index, 128, 127, 0])
        (package / f"{fid}_expected.bin").write_bytes(binary)
        digest = validator.sha(package / f"{fid}_expected.bin")
        fixtures.append(dict(fixture_id=fid, expected_elements=4,
                             expected_output_sha256=digest,
                             files={f"{fid}_expected.bin": digest}))
    case = dict(case_id="synthetic_validator_test", fixtures=fixtures,
                phase_counter_abi="P3T1", phase_contract=board.PHASE_CONTRACT,
                numeric_contract_sha256=board._canonical(contract),
                source_files={p.name: validator.sha(p) for p in package.iterdir()})
    case["case_sha256"] = board._canonical(case)
    write_json(package / "case.json", case)
    bit = package / "vivado_shell/top.bit"
    bit.parent.mkdir(parents=True)
    bit.write_bytes(b"synthetic-no-valid-bitstream")
    syn = package / "hls_project/solution1/syn/report/pilot_kernel_csynth.xml"
    syn.parent.mkdir(parents=True)
    syn.write_text("<synthetic_unit_test />\n")
    trace = package / "phase_evidence/synthetic.json"
    write_json(trace, {"evidence": "SYNTHETIC_UNIT_TEST_ONLY"})
    phase = dict(schema="t6_integrated_phase_validation_v1",
                 status="PASS_INTEGRATED_PHASE_VALIDATION",
                 case_sha256=case["case_sha256"], bitstream_sha256=validator.sha(bit),
                 phase_contract=board.PHASE_CONTRACT, phase_interface="ap_vld_latched",
                 single_clock_domain=True, phases_disjoint=True,
                 source_files={name: case["source_files"][name]
                               for name in ("kernel.cpp", "config.h", "phase_timer.v")},
                 synthesis_report={"path": syn.relative_to(package).as_posix(), "sha256": validator.sha(syn)},
                 evidence_files={trace.relative_to(package).as_posix(): validator.sha(trace)})
    write_json(package / "phase_validation.json", phase)
    write_json(package / "board_bundle.json", {"evidence": "SYNTHETIC_UNIT_TEST_ONLY"})
    shutil.copyfile(Path(measurement.__file__), package / "measurement.py")
    expected = dict(case_id=case["case_id"], case_sha256=case["case_sha256"],
                    numeric_contract_sha256=case["numeric_contract_sha256"],
                    bitstream_sha256=validator.sha(bit), expected_output_sha256=fixtures[0]["expected_output_sha256"],
                    board_id="SYNTHETIC_NO_BOARD", expected_elements=4, clock_hz=187500000.0)
    timing = dict(cases=[expected], batches=[], phase_clock_evidence_verified=True)
    rows = []
    for index in range(1, 4):
        bid = f"batch{index}"
        timing["batches"].append(dict(case_id=case["case_id"], batch_id=bid,
            independence_event="reload_bitstream", independence_id=f"synthetic-load-{index}",
            clock_hz=187500000.0, clock_hz_end=187500000.0,
            cpu_governor="UNAVAILABLE", temperature_c=40.0, temperature_c_end=41.0,
            started_utc=f"2026-10-13T00:0{index}:00+00:00",
            finished_utc=f"2026-10-13T00:0{index}:50+00:00"))
        folder = output / bid
        folder.mkdir()
        environment = dict(cpu=dict(cpu_governor="UNAVAILABLE", policies=[dict(
            path="synthetic-policy", actual="userspace", available=["userspace"],
            condition_deviation="performance unavailable")]))
        write_json(folder / "environment_start.json", environment)
        write_json(folder / "environment_end.json", environment)
        checks = []
        for fixture in fixtures:
            fid = fixture["fixture_id"]
            shutil.copyfile(package / f"{fid}_expected.bin", folder / f"{fid}_board.bin")
            checks.append(dict(fixture_id=fid, compared_elements=4, mismatched_elements=0,
                               observed_output_sha256=fixture["expected_output_sha256"]))
        write_json(folder / "correctness_fixtures.json", checks)
        for phase_name, count in (("warmup", 50), ("measured", 1000)):
            for iteration in range(count):
                row = {key: value for key, value in expected.items() if key != "expected_elements"}
                row.update(batch_id=bid, phase=phase_name, iteration=iteration,
                           compute_ns=1000, transfer_ns=2000, call_ns=4000,
                           compared_elements=4, mismatched_elements=0,
                           observed_output_sha256=expected["expected_output_sha256"],
                           group_id=str(iteration), group_size=1, group_call_total_ns=4000)
                rows.append(row)
    with (output / "timing_raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=measurement.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "timing_manifest.json", timing)
    summary = measurement.summarize_csv(output / "timing_raw.csv", timing)
    summary.update(correctness_fixtures_passed=15, integrated_phase_validation="PASS_BOUND_ARTIFACT",
                   board_bundle_sha256=validator.sha(package / "board_bundle.json"),
                   protocol_status="DRAFT", formal_cost_eligible=False)
    write_json(output / "summary.json", summary)
    session = {key: expected[key] for key in ("case_sha256", "numeric_contract_sha256", "bitstream_sha256")}
    session.update(phase_validation_sha256=validator.sha(package / "phase_validation.json"),
                   board_bundle_sha256=validator.sha(package / "board_bundle.json"))
    write_json(output / "session.json", session)
    write_json(output.parent / "request.json", {"config": dict(mode="timing", case_sha256=case["case_sha256"])})
    snapshot = root / "snapshot"
    rebind_transport(snapshot)
    return snapshot, package


def rebind_transport(snapshot):
    """Model an independently hash-verified transport carrying changed contents."""
    evidence = snapshot / "evidence"
    members = {p.relative_to(evidence).as_posix(): validator.sha(p)
               for p in evidence.rglob("*") if p.is_file() and p.name != "snapshot_manifest.json"}
    write_json(evidence / "snapshot_manifest.json", dict(schema="pilot_board_snapshot_v1", files=members))
    with tarfile.open(snapshot / "snapshot.tar.gz", "w:gz") as archive:
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(evidence).as_posix())
    write_json(snapshot / "download.json", dict(archive_sha256=validator.sha(snapshot / "snapshot.tar.gz")))


class SnapshotValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t6_validator_")
        self.addCleanup(self.temporary.cleanup)
        self.snapshot, self.package = synthetic_snapshot(Path(self.temporary.name))

    def test_exact_new_snapshot_and_raw_cpu_false_are_preserved(self):
        result = validator.validate_timing_snapshot(self.snapshot, self.package)
        self.assertEqual(result["status"], "PASS_IMPORT_AUDIT")
        self.assertEqual(result["numerical_status"], "PASS")
        self.assertIs(result["protocol_conditions_met"], False)
        self.assertIs(result["hardware_measurement_eligible"], False)
        self.assertEqual(result["cpu_governor_observations"]["batch1"]["observed_policy_values"], ["userspace"])
        self.assertEqual(result["runtime_protocol"]["status"], "LEGACY_UNSPECIFIED")

    def add_guarded_protocol(self):
        values = dict(runtime_protocol_id=board.RUNTIME_PROTOCOL_ID,
                      host_timing_guard_version=board.HOST_TIMING_GUARD_VERSION,
                      individual_call_ns_scope=board.INDIVIDUAL_CALL_NS_SCOPE,
                      group_call_ns_scope=board.GROUP_CALL_NS_SCOPE,
                      fixture_selection_scope="outside timed warmup/measured groups",
                      host_timing_comparability=board.HOST_TIMING_COMPARABILITY)
        output = self.snapshot / "evidence/output"
        session = validator.read_json(output / "session.json")
        session.update(**values, call_scope=values["group_call_ns_scope"])
        timing = validator.read_json(output / "timing_manifest.json")
        timing.update(**values, individual_invoke_identity_checks_inside_call_ns=2,
                      invocation_identity_checks_inside_group_total_ns=4)
        write_json(output / "session.json", session)
        write_json(output / "timing_manifest.json", timing)
        rebind_transport(self.snapshot)
        return output

    def test_guarded_runtime_and_both_host_scopes_are_preserved_exactly(self):
        self.add_guarded_protocol()
        result = validator.validate_timing_snapshot(self.snapshot, self.package)
        protocol = result["runtime_protocol"]
        self.assertEqual(protocol["runtime_protocol_id"], board.RUNTIME_PROTOCOL_ID)
        self.assertEqual(protocol["host_timing_guard_version"], board.HOST_TIMING_GUARD_VERSION)
        self.assertEqual(protocol["individual_call_ns_scope"], board.INDIVIDUAL_CALL_NS_SCOPE)
        self.assertEqual(protocol["group_call_ns_scope"], board.GROUP_CALL_NS_SCOPE)
        self.assertEqual(protocol["call_scope"], board.GROUP_CALL_NS_SCOPE)
        self.assertEqual(protocol["individual_invoke_identity_checks_inside_call_ns"], 2)
        self.assertEqual(protocol["invocation_identity_checks_inside_group_total_ns"], 4)
        self.assertIn("NOT_PERFORMED", protocol["host_statistics_pooling"])
        self.assertIs(result["hardware_measurement_eligible"], False)

    def test_legacy_raw_call_scope_is_preserved_without_inventing_guard_version(self):
        path = self.snapshot / "evidence/output/session.json"
        session = validator.read_json(path)
        session["call_scope"] = "synthetic legacy group interval"
        write_json(path, session)
        rebind_transport(self.snapshot)
        protocol = validator.validate_timing_snapshot(self.snapshot, self.package)["runtime_protocol"]
        self.assertEqual(protocol["runtime_protocol_id"], "LEGACY_UNSPECIFIED")
        self.assertEqual(protocol["host_timing_guard_version"], "LEGACY_UNSPECIFIED")
        self.assertEqual(protocol["group_call_ns_scope"], session["call_scope"])
        self.assertIsNone(protocol["individual_call_ns_scope"])

    def test_protocol_difference_or_partial_declaration_fails_even_with_valid_transport(self):
        output = self.add_guarded_protocol()
        path = output / "timing_manifest.json"
        original = validator.read_json(path)
        for field in ("runtime_protocol_id", "host_timing_guard_version", "individual_call_ns_scope",
                      "group_call_ns_scope", "fixture_selection_scope", "host_timing_comparability"):
            changed = deepcopy(original)
            changed[field] += "-changed"
            write_json(path, changed)
            rebind_transport(self.snapshot)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "timing.*differs"):
                validator.validate_timing_snapshot(self.snapshot, self.package)
        changed = deepcopy(original)
        del changed["group_call_ns_scope"]
        write_json(path, changed)
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validator.validate_timing_snapshot(self.snapshot, self.package)

    def test_guard_count_and_session_group_alias_cannot_be_silently_changed(self):
        output = self.add_guarded_protocol()
        path = output / "timing_manifest.json"
        timing = validator.read_json(path)
        timing["invocation_identity_checks_inside_group_total_ns"] = 2
        write_json(path, timing)
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "count differs"):
            validator.validate_timing_snapshot(self.snapshot, self.package)
        timing["invocation_identity_checks_inside_group_total_ns"] = 4
        write_json(path, timing)
        path = output / "session.json"
        session = validator.read_json(path)
        session["call_scope"] = board.INDIVIDUAL_CALL_NS_SCOPE
        write_json(path, session)
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "call_scope must equal"):
            validator.validate_timing_snapshot(self.snapshot, self.package)

    def test_archive_and_member_corruption_fail_before_statistics(self):
        for kind in ("archive", "raw"):
            target = self.snapshot / ("snapshot.tar.gz" if kind == "archive" else "evidence/output/timing_raw.csv")
            original = target.read_bytes()
            target.write_bytes(original + b"corrupt")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validator.validate_timing_snapshot(self.snapshot, self.package)
            target.write_bytes(original)

    def test_case_or_bit_identity_mutation_fails_even_with_valid_transport(self):
        target = self.snapshot / "evidence/output/session.json"
        original = validator.read_json(target)
        for field in ("case_sha256", "numeric_contract_sha256", "bitstream_sha256", "phase_validation_sha256", "board_bundle_sha256"):
            changed = deepcopy(original)
            changed[field] = "0" * 64
            write_json(target, changed)
            rebind_transport(self.snapshot)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.validate_timing_snapshot(self.snapshot, self.package)
        write_json(target, original)

    def test_raw_fixture_mismatch_fails_even_with_valid_transport(self):
        target = self.snapshot / "evidence/output/batch2/test3_board.bin"
        target.write_bytes(bytes([3, 127, 127, 0]))
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "full integer output"):
            validator.validate_timing_snapshot(self.snapshot, self.package)

    def test_incomplete_fixture_record_and_eligibility_promotion_are_rejected(self):
        fixture = self.snapshot / "evidence/output/batch1/correctness_fixtures.json"
        original = validator.read_json(fixture)
        write_json(fixture, original[:-1])
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            validator.validate_timing_snapshot(self.snapshot, self.package)
        write_json(fixture, original)
        summary = self.snapshot / "evidence/output/summary.json"
        changed = validator.read_json(summary)
        changed["protocol_conditions_met"] = changed["hardware_measurement_eligible"] = True
        write_json(summary, changed)
        rebind_transport(self.snapshot)
        with self.assertRaisesRegex(ValueError, "exact value differs"):
            validator.validate_timing_snapshot(self.snapshot, self.package)

    def test_csv_omission_and_changed_measurement_source_are_rejected(self):
        csv_file = self.snapshot / "evidence/output/timing_raw.csv"
        original = csv_file.read_bytes()
        csv_file.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
        rebind_transport(self.snapshot)
        with self.assertRaises(ValueError):
            validator.validate_timing_snapshot(self.snapshot, self.package)
        csv_file.write_bytes(original)
        rebind_transport(self.snapshot)
        with (self.package / "measurement.py").open("a") as stream:
            stream.write("\n# Changed synthetic source\n")
        with self.assertRaisesRegex(ValueError, "measurement source differs"):
            validator.validate_timing_snapshot(self.snapshot, self.package)


class ReplayToleranceTests(unittest.TestCase):
    def test_existing_directory_alias_preserves_import_and_all_gates(self):
        with tempfile.TemporaryDirectory(prefix="t6_path_alias_") as temporary:
            root = Path(temporary)
            (root / "anchor").mkdir()
            snapshot = root / "snapshot"
            output = snapshot / "evidence/output"
            output.mkdir(parents=True)
            replay = dict(snapshot=snapshot.as_posix(), output=output.as_posix(),
                          numerical_status="PASS", hardware_measurement_eligible=False)
            stored = dict(replay, snapshot=(root / "anchor/../snapshot").as_posix(),
                          output=(root / "anchor/../snapshot/evidence/output").as_posix())
            original = deepcopy(stored)
            result = validator.compare_imported_timing_audit(stored, replay)
            self.assertEqual(result["status"], "PASS_REPLAY_EQUIVALENCE")
            self.assertEqual(len(result["verified_local_directory_aliases"]), 2)
            self.assertEqual(stored, original)
            stored["hardware_measurement_eligible"] = True
            with self.assertRaises(ValueError):
                validator.compare_imported_timing_audit(stored, replay)

    def test_different_or_missing_directory_is_not_an_import_alias(self):
        with tempfile.TemporaryDirectory(prefix="t6_path_alias_") as temporary:
            root = Path(temporary)
            (root / "original").mkdir()
            (root / "different").mkdir()
            replay = dict(snapshot=(root / "original").as_posix())
            for name in ("different", "missing"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validator.compare_imported_timing_audit(
                        dict(snapshot=(root / name).as_posix()), replay)

    def test_functions_are_exact_extractions_after_namespace_substitution(self):
        for name in ("compare_replayed_summary", "_cpu_governor_display", "validate_timing_snapshot"):
            old = inspect.getsource(getattr(legacy_report, name))
            target = (validator._validate_timing_snapshot_without_runtime_protocol
                      if name == "validate_timing_snapshot" else getattr(validator, name))
            new = inspect.getsource(target)
            self.assertEqual(old.replace("hardware_measurement_pilot", "hardware_template_coverage_v1"), new)

    def test_old_derived_import_can_keep_its_raw_artifacts_and_new_metadata_is_exact(self):
        checked = dict(status="PASS_IMPORT_AUDIT", case_id="synthetic",
                       runtime_protocol=dict(status="LEGACY_UNSPECIFIED"))
        original_import = {key: value for key, value in checked.items() if key != "runtime_protocol"}
        self.assertEqual(validator.compare_imported_timing_audit(original_import, checked)["status"],
                         "PASS_REPLAY_EQUIVALENCE")
        altered = deepcopy(checked)
        altered["runtime_protocol"]["status"] = "RECORDED_GUARDED_HOST_PROTOCOL"
        with self.assertRaises(ValueError):
            validator.compare_imported_timing_audit(altered, checked)

    def test_eight_ulp_only_for_named_statistics_and_never_gates(self):
        expected = dict(cases=[dict(scopes={"compute": {"mean_ns": 1000.0}},
            batches=[dict(metadata={"clock_hz": 187500000.0})])], status="PASS", cv_limit=0.05,
            protocol_conditions_met=False, hardware_measurement_eligible=False)
        actual = deepcopy(expected)
        for _ in range(8):
            actual["cases"][0]["scopes"]["compute"]["mean_ns"] = math.nextafter(
                actual["cases"][0]["scopes"]["compute"]["mean_ns"], math.inf)
        self.assertEqual(validator.compare_replayed_summary(actual, expected)["tolerated_float_differences"][0]["ulp_distance"], 8)
        actual["cases"][0]["scopes"]["compute"]["mean_ns"] = math.nextafter(
            actual["cases"][0]["scopes"]["compute"]["mean_ns"], math.inf)
        with self.assertRaises(ValueError):
            validator.compare_replayed_summary(actual, expected)
        for target in ("threshold", "clock", "gate", "eligibility", "type"):
            actual = deepcopy(expected)
            if target == "threshold": actual["cv_limit"] = math.nextafter(actual["cv_limit"], math.inf)
            elif target == "clock": actual["cases"][0]["batches"][0]["metadata"]["clock_hz"] += 1.0
            elif target == "gate": actual["status"] = "FAIL"
            elif target == "eligibility": actual["hardware_measurement_eligible"] = True
            else: actual["cases"][0]["scopes"]["compute"]["mean_ns"] = 1000
            with self.subTest(target=target), self.assertRaises(ValueError):
                validator.compare_replayed_summary(actual, expected)


if __name__ == "__main__":
    unittest.main()
