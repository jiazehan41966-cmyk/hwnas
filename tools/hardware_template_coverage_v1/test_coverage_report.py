"""Report failures must stay visible instead of becoming implementation PASS."""
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
import xml.etree.ElementTree as ET

from . import coverage_report as report
from .remote_worker import hls_resource_fit


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class CoverageGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t6_report_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = dict(case_id="synthetic_report_test", output_bits=8, case_sha256="a" * 64,
            fixtures=[dict(fixture_id=f"test{i}", expected_elements=4,
                           expected_output_sha256=hashlib.sha256(bytes([i, 1, 2, 3])).hexdigest()) for i in range(5)])

    def evidence(self, bram):
        used = dict(BRAM_18K=bram, DSP=1, FF=120, LUT=100, URAM=0)
        available = dict(BRAM_18K=624, DSP=1728, FF=460800, LUT=230400, URAM=96)
        body = "<profile><UserAssignments><Part>xczu7ev-ffvc1156-2-e</Part></UserAssignments><AreaEstimates>"
        for tag, values in (("Resources", used), ("AvailableResources", available)):
            body += "<" + tag + ">" + "".join(f"<{key}>{value}</{key}>" for key, value in values.items()) + "</" + tag + ">"
        body += "</AreaEstimates></profile>"
        xml = self.root / "pilot_kernel_csynth.xml"
        xml.write_text(body, encoding="utf-8")
        fit = hls_resource_fit(ET.parse(xml))
        write(self.root / "hls_resource_fit.json", fit)
        witnesses = [dict(fixture=spec["fixture_id"], bytes=4,
            expected_sha256=spec["expected_output_sha256"], actual_sha256=spec["expected_output_sha256"])
            for spec in self.case["fixtures"]]
        write(self.root / "csim_parity.json", dict(status="PASS", fixtures=witnesses))
        write(self.root / "c_result.json", dict(fixtures=5, mismatches=0, elements=20))
        return dict(status="PASS" if fit["status"] == "FIT" else "FAILED",
                    evidence_directory=self.root.as_posix(), record=dict(gate=dict(resource_fit=fit)),
                    bound_files={p.name: report.sha(p) for p in self.root.iterdir()})

    def test_missing_hardware_is_not_pass_and_unselected_route_is_not_measured(self):
        result = report._hls(self.case, dict(status="NOT_RUN", evidence_directory=None), [])
        self.assertEqual(result["csim"], "NOT_RUN")
        self.assertEqual(result["synthesis"], "NOT_RUN")
        self.assertEqual(result["resource_fit"], "NOT_RUN")
        routed = report._route(dict(status="NOT_RUN", evidence_directory=None), result, False, [], self.case["case_id"])
        self.assertEqual(routed["status"], "NOT_REQUIRED_REPRESENTATIVE_ONLY")
        self.assertEqual(routed["bitstream"], "NOT_GENERATED")

    def test_oversized_synthesis_keeps_csim_but_cannot_become_fit(self):
        stage = self.evidence(624)
        issues = []
        result = report._hls(self.case, stage, issues)
        self.assertEqual(result["csim"], "PASS")
        self.assertEqual(result["synthesis"], "PASS")
        self.assertEqual(result["resource_fit"], "OVERSIZED")
        self.assertEqual(result["process_status"], "FAILED")
        self.assertFalse(issues)

    def test_resource_record_is_replayed_from_raw_xml_and_changed_record_rejected(self):
        stage = self.evidence(623)
        self.assertEqual(report._hls(self.case, stage, [])["resource_fit"], "FIT")
        changed = json.loads((self.root / "hls_resource_fit.json").read_text())
        changed["used"]["BRAM_18K"] -= 1
        write(self.root / "hls_resource_fit.json", changed)
        issues = []
        self.assertEqual(report._hls(self.case, stage, issues)["resource_fit"], "INVALID_RESOURCE_FIT")
        self.assertTrue(issues)

    def test_partial_or_unbound_csim_witness_cannot_become_pass(self):
        stage = self.evidence(10)
        parity = json.loads((self.root / "csim_parity.json").read_text())
        parity["fixtures"].pop()
        write(self.root / "csim_parity.json", parity)
        self.assertEqual(report._hls(self.case, stage, [])["csim"], "FAIL_CSIM_EVIDENCE")
        del stage["bound_files"]["csim_parity.json"]
        self.assertEqual(report._hls(self.case, stage, [])["csim"], "INVALID_UNBOUND_EVIDENCE")

    def test_stale_stage_identity_is_not_accepted(self):
        path = self.root / "results/synthetic_report_test/hls/latest.json"
        path.parent.mkdir(parents=True)
        write(path, dict(case_sha256="b" * 64, source_manifest_sha256="c" * 64,
                         evidence_directory="results/synthetic_report_test/hls/run001", status="PASS"))
        issues = []
        relative = path.relative_to(self.root).as_posix()
        stage = report._stage(self.root, {relative: report.sha(path)}, self.case, "hls", "c" * 64, True, issues)
        self.assertEqual(stage["status"], "STALE_OR_INVALID_BINDING")
        self.assertTrue(issues)

    def test_route_substages_do_not_duplicate_prior_hls_timings(self):
        records = [dict(stage="csim", wall_seconds=2, returncode=0),
                   dict(stage="hls_synthesis", wall_seconds=3, returncode=0),
                   dict(stage="ip_export", wall_seconds=4, returncode=0),
                   dict(stage="synthesis", wall_seconds=5, returncode=0),
                   dict(stage="route", wall_seconds=6, returncode=0),
                   dict(stage="bitstream", wall_seconds=7, returncode=0)]
        stage = dict(status="PASS", record=dict(substage_records=records, wall_seconds=40))
        rows = report._stage_rows(self.case["case_id"], dict(status="PASS"),
                                 dict(record={}), stage, {})
        routed = [row for row in rows if row["evidence_layer"] == "ROUTE"]
        self.assertEqual([row["stage"] for row in routed], ["synthesis", "route", "bitstream"])
        envelope = next(row for row in rows if row["stage"] == "route_process_envelope")
        self.assertEqual(envelope["evidence_layer"], "PARENT_ENVELOPE_DO_NOT_SUM_WITH_SUBSTAGES")

    def history_snapshot(self, state="DMA_QUARANTINED", mode="timing", identity=None):
        snapshot = self.root / "board_snapshots" / (mode + "_record")
        evidence = snapshot / "evidence"
        output = evidence / "output"
        output.mkdir(parents=True)
        job = snapshot.name
        write(evidence / "request.json", dict(job_id=job, lock_binding=dict(owner_token="private-test-value"),
            config=dict(case_id=self.case["case_id"], case_sha256=identity or self.case["case_sha256"], mode=mode)))
        write(evidence / "status.json", dict(job_id=job, state=state))
        if mode == "timing":
            (output / "timing_raw.csv").write_text(
                "case_id,case_sha256,batch_id,phase,iteration,compute_ns,transfer_ns,compared_elements,mismatched_elements,observed_output_sha256,expected_output_sha256\n"
                f"{self.case['case_id']},{self.case['case_sha256']},batch1,measured,0,1,2,4,0,expected,expected\n"
                f"{self.case['case_id']},{self.case['case_sha256']},batch1,measured,1,,,,,,expected\n", encoding="utf-8")
            write(output / "failure.json", dict(status="FAIL", error_type="UnsafeDeviceState", error="timeout"))
        files = {path.relative_to(evidence).as_posix(): report.sha(path)
                 for path in evidence.rglob("*") if path.is_file()}
        write(evidence / "snapshot_manifest.json", dict(schema="pilot_board_snapshot_v1", files=files,
                                                       job_id=job, observed_state=state))
        archive = snapshot / "snapshot.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for path in evidence.rglob("*"):
                if path.is_file():
                    stream.add(path, arcname=path.relative_to(evidence).as_posix())
        write(snapshot / "download.json", dict(ok=True, archive_sha256=report.sha(archive),
            archive_bytes=archive.stat().st_size, observed_state=state))
        return snapshot

    def history_case(self):
        return dict(case_id=self.case["case_id"], case_sha256=self.case["case_sha256"],
            board=dict(correctness="NOT_RUN", timing="NOT_RUN", measured_calls=None, stats=None))

    def test_quarantine_and_probe_remain_visible_without_pass_or_aggregate(self):
        self.history_snapshot()
        self.history_snapshot("PASS", "probe")
        case = self.history_case()
        before = {key: case["board"][key] for key in ("correctness", "timing", "measured_calls", "stats")}
        issues = []
        history = report._board_history(self.root, [case], issues)
        self.assertFalse(issues)
        self.assertEqual({row["classification"] for row in history},
                         {"DMA_QUARANTINED_INVALID_RUN", "PROBE_ONLY_NOT_TIMING_ACCEPTED"})
        timing = next(row for row in history if row["mode"] == "timing")
        self.assertEqual(timing["recorded_calls"][0]["completed_measured_rows"], 1)
        self.assertEqual(timing["recorded_calls"][0]["incomplete_iterations"], [1])
        self.assertTrue(all(row["accepted_for_pass_or_aggregate"] is False for row in history))
        self.assertEqual(before, {key: case["board"][key] for key in before})
        self.assertNotIn("private-test-value", json.dumps(history))

    def test_unselected_pass_state_does_not_become_selected_timing_pass(self):
        self.history_snapshot("PASS")
        case = self.history_case()
        history = report._board_history(self.root, [case], [])
        self.assertEqual(history[0]["classification"], "UNSELECTED_RUN_NOT_ACCEPTED")
        self.assertEqual(case["board"]["timing"], "NOT_RUN")
        self.assertIsNone(history[0]["statistics"])

    def test_changed_extracted_manifest_cannot_hide_changed_csv(self):
        snapshot = self.history_snapshot()
        evidence = snapshot / "evidence"
        csv_path = evidence / "output/timing_raw.csv"
        csv_path.write_text(csv_path.read_text(encoding="utf-8").replace(",measured,1,", ",measured,2,"), encoding="utf-8")
        manifest = report.load(evidence / "snapshot_manifest.json")
        manifest["files"]["output/timing_raw.csv"] = report.sha(csv_path)
        write(evidence / "snapshot_manifest.json", manifest)
        issues = []
        history = report._board_history(self.root, [self.history_case()], issues)
        self.assertEqual(history[0]["binding"], "INVALID_SNAPSHOT_BINDING")
        self.assertFalse(history[0]["recorded_calls"])
        self.assertTrue(issues)

    def test_changed_archive_and_wrong_case_identity_are_rejected(self):
        snapshot = self.history_snapshot()
        archive = snapshot / "snapshot.tar.gz"
        with archive.open("ab") as stream:
            stream.write(b"changed")
        self.assertEqual(report._board_history(self.root, [self.history_case()], [])[0]["binding"],
                         "INVALID_SNAPSHOT_BINDING")
        # Another distinct snapshot remains byte-bound but carries a wrong exact identity.
        self.history_snapshot("PASS", "probe", "b" * 64)
        history = report._board_history(self.root, [self.history_case()], [])
        self.assertTrue(all(row["binding"] == "INVALID_SNAPSHOT_BINDING" for row in history))

    def test_selected_metrics_are_unchanged_by_snapshot_history(self):
        snapshot = self.history_snapshot("PASS")
        case = self.history_case()
        case["board"].update(snapshot=snapshot.resolve().as_posix(), correctness="PASS_15_FULL_FIXTURES",
                             timing="PASS", measured_calls=3000, stats=dict(compute=dict(median_ns=123)))
        before = {key: case["board"][key] for key in ("correctness", "timing", "measured_calls", "stats")}
        history = report._board_history(self.root, [case], [])
        self.assertTrue(history[0]["selected_for_report"])
        self.assertFalse(case["board"]["unselected_history"])
        self.assertEqual(before, {key: case["board"][key] for key in before})

    def test_private_lock_owner_fields_are_omitted_recursively(self):
        public = report._public_report(dict(lock=dict(owner_token="secret", job_id="job"),
            nested=[dict(lock_owner_token="also-secret", state="T")]))
        self.assertNotIn("secret", json.dumps(public))
        self.assertNotIn("owner_token", json.dumps(public))
        self.assertEqual(public["lock"], dict(job_id="job"))

    def test_distinct_runtime_protocols_are_kept_per_run_and_never_pooled(self):
        old = report.timing_protocol(dict(call_scope="old host group"), {})
        new_fields = dict(runtime_protocol_id="t6_tiled_p3t1_pl_epoch_guard_v1",
            host_timing_guard_version="p3t1_pl_epoch_guard_v1", individual_call_ns_scope="new individual scope",
            group_call_ns_scope="new group scope", fixture_selection_scope="outside measured groups",
            host_timing_comparability="guarded host timing is separate")
        new = report.timing_protocol(dict(**new_fields, call_scope="new group scope"),
            dict(**new_fields, individual_invoke_identity_checks_inside_call_ns=2,
                 invocation_identity_checks_inside_group_total_ns=4))
        self.assertNotEqual(old["runtime_protocol_id"], new["runtime_protocol_id"])
        for record in (old, new):
            self.assertEqual(record["host_statistics_pooling"], "NOT_PERFORMED_SINGLE_SELECTED_RUN_PER_CASE")
            self.assertIn("DO_NOT_POOL", record["cross_protocol_host_comparison"])


if __name__ == "__main__":
    unittest.main()
