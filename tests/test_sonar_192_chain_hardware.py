"""Hardware interface tests; every generated artifact is diagnostic in tmpdir."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.sonar_192_chain import hardware as hw


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sonar192_DIAGNOSTIC_ONLY_")
        self.root = Path(self.tmp.name)
        self.mapping = dict(PE=4, SIMD=1, tile_h=2, tile_w=32,
                            buffer_profile="diagnostic_tiles", implementation_id="DIAGNOSTIC_TEST_KERNEL")
        self.precision = dict(input_bitwidth=8, output_bitwidth=8, weight_bitwidth=None, accumulator_bitwidth=32)
        self.platform = dict(part="DIAGNOSTIC_TEST_PART", tool_version="TEST_ONLY")

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, folder, name, value):
        path = folder / name
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def receipt(self, name="a", height=40, compute=1000, transfer=2000, host=5000, op="maxpool"):
        folder = self.root / name
        folder.mkdir()
        shape = dict(op=op, operator_type=op, input_shapes=[[4, height, height]],
                     output_shape=[4, height // 2, height // 2], kernel=2, stride=2, padding=0, activation="linear")
        primitive = hw.primitive_record(dict(shape, id="node", precision=self.precision), "W8A8")
        raw = folder / "raw.csv"
        with raw.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["batch_id", "phase", "compute_ns", "transfer_ns", "host_ns", "clock_hz", "valid", "fixture_match"])
            writer.writeheader()
            for batch in range(3):
                for phase in ("warmup", "measure", "measure"):
                    writer.writerow(dict(batch_id=str(batch), phase=phase, compute_ns=compute,
                                         transfer_ns=transfer, host_ns=host, clock_hz=100_000_000,
                                         valid="true", fixture_match="true"))
        protocol = self.save(folder, "protocol.json", dict(purpose="diagnostic", status="DRAFT",
                         scope="DIAGNOSTIC_TEST_FIXTURE", runtime_protocol="DIAGNOSTIC_ONLY_PROTOCOL",
                         minimum_batches=3, minimum_warmup=1, minimum_measured_per_batch=2, maximum_pl_cv=.05))
        plan = self.save(folder, "mapping.json", dict(mapping=self.mapping, status="FROZEN_CONCRETE_MAPPING", purpose="diagnostic"))
        numeric = self.save(folder, "numeric.json", dict(precision=self.precision, purpose="diagnostic"))
        interface = self.save(folder, "interface.json", dict(precision=self.precision,
                        numeric_contract_sha256=hw.digest(hw.read_json(numeric)), purpose="diagnostic"))
        bit, hwh, source = folder / "fixture.bit", folder / "fixture.hwh", folder / "kernel.cpp"
        for path in (bit, hwh, source):
            path.write_text("DIAGNOSTIC TEST FIXTURE - NOT BOARD EVIDENCE\n", encoding="utf-8")
        sources = self.save(folder, "sources.json", dict(purpose="diagnostic", files={"kernel.cpp": hw.bind(source)}))
        route = self.save(folder, "route.json", dict(purpose="diagnostic", stage="POST_ROUTE", part=self.platform["part"],
                         bitstream_sha256=hw.sha256_file(bit), hwh_sha256=hw.sha256_file(hwh), WNS_ns=1,
                         LUT=100 + height, DSP=0, BRAM18=2))
        fixtures = []
        for batch in range(3):
            expected, actual = folder / f"expected{batch}.bin", folder / f"actual{batch}.bin"
            expected.write_bytes(bytes([batch]) * (4 * (height // 2)**2))
            actual.write_bytes(expected.read_bytes())
            fixtures.append(dict(batch_id=str(batch), fixture_id="DIAGNOSTIC_FIXTURE",
                                 expected=hw.bind(expected), actual=hw.bind(actual)))
        correctness = self.save(folder, "correctness.json", dict(purpose="diagnostic", fixtures=fixtures))
        evidence = {key: hw.bind(path) for key, path in dict(raw_csv=raw, protocol=protocol,
                        mapping_plan=plan, numeric_contract=numeric, interface_contract=interface,
                        route_resources=route, bitstream=bit, hwh=hwh, source_manifest=sources, correctness=correctness).items()}
        key = dict(primitive_id=primitive["primitive_id"], mapping=self.mapping, platform=self.platform, profile="W8A8")
        review = self.save(folder, "review.json", dict(status="PASS_SOURCE_BOUND_BOARD_MEASUREMENT", purpose="diagnostic",
                           evidence_sha256=hw.digest(evidence), measurement_key_sha256=hw.digest(key)))
        evidence["review"] = hw.bind(review)
        receipt = dict(schema_version="sonar192_board_operator_receipt_v1", purpose="diagnostic",
                        scope="DIAGNOSTIC_TEST_FIXTURE", record_id=name, shape=shape, precision=self.precision,
                        primitive_id=primitive["primitive_id"], mapping=self.mapping, profile="W8A8",
                        platform=self.platform, nominal_clock_hz=100_000_000,
                        formal_cost_eligible=False, hardware_measurement_eligible=False, protocol_conditions_met=False,
                        evidence=evidence)
        return self.save(folder, "receipt.json", receipt)

    def test_raw_cost_consumption_and_serial_scope(self):
        receipt = self.receipt()
        records = hw.import_board_receipts([receipt], formal=False)
        self.assertEqual(records[0]["metrics"]["compute_us"], 1)
        self.assertEqual(records[0]["measured_calls"], 6)
        graph = dict(nodes=[dict(records[0]["shape"], id="a", precision=self.precision),
                            dict(records[0]["shape"], id="b", precision=self.precision)], space_version=hw.SPACE)
        result = hw.query_graph_cost(graph, records, "W8A8", self.mapping, formal=False,
                                     identity={"space_version": hw.SPACE})
        self.assertEqual(result["status"], "DIAGNOSTIC_LOOKUP")
        self.assertEqual(result["cost"]["compute_us"], 2)
        self.assertEqual(result["cost"]["LUT"], 140)
        self.assertFalse(result["formal_eligible"])
        self.assertIn("not_network_e2e", result["latency_scope"])

    def test_fixture_formal_admission_fails(self):
        with self.assertRaisesRegex(ValueError, "Formal measurement protocol"):
            hw.import_board_receipts([self.receipt()], formal=True)

    def test_self_signed_receipt_review_are_not_independent_formal_registration(self):
        path = self.receipt()
        self.assertEqual(hw.REGISTERED_BOARD_OPERATOR_RECEIPTS, ())
        with self.assertRaisesRegex(ValueError, "Unregistered formal board operator receipt"):
            hw.require_registered_board_receipt(path, path.parent / "review.json")

    def test_receipt_only_relabel_cannot_admit_diagnostic(self):
        path = self.receipt()
        data = hw.read_json(path)
        data.update(purpose="formal", scope="BOARD_OPERATOR_MEASUREMENT", formal_cost_eligible=True,
                    hardware_measurement_eligible=True, protocol_conditions_met=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "purpose differs"):
            hw.import_board_receipts([path], formal=True)

    def test_post_import_mutation_and_underlying_source_drift_rejected(self):
        path = self.receipt()
        records = hw.import_board_receipts([path], formal=False)
        records[0]["formal_cost_eligible"] = True
        with self.assertRaisesRegex(ValueError, "modified"):
            records.verify_unchanged()
        records = hw.import_board_receipts([path], formal=False)
        (path.parent / "kernel.cpp").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            records.verify_unchanged()

    def test_raw_corruption_fails_before_statistics(self):
        path = self.receipt()
        (path.parent / "raw.csv").write_text("invalid raw rows", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            hw.import_board_receipts([path], formal=False)

    def test_false_parity_flag_cannot_replace_full_output_comparison(self):
        path = self.receipt()
        actual = path.parent / "actual0.bin"
        actual.write_bytes(b"x" * actual.stat().st_size)
        correctness_path = path.parent / "correctness.json"
        correctness = hw.read_json(correctness_path)
        correctness["fixtures"][0]["actual"] = hw.bind(actual)
        correctness_path.write_text(json.dumps(correctness), encoding="utf-8")
        receipt = hw.read_json(path)
        receipt["evidence"]["correctness"] = hw.bind(correctness_path)
        review_path = path.parent / "review.json"
        review = hw.read_json(review_path)
        review["evidence_sha256"] = hw.digest({k:v for k,v in receipt["evidence"].items() if k != "review"})
        review_path.write_text(json.dumps(review), encoding="utf-8")
        receipt["evidence"]["review"] = hw.bind(review_path)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not bit-exact"):
            hw.import_board_receipts([path], formal=False)

    def test_unverified_json_rows_and_ambiguous_repeats_rejected(self):
        path = self.receipt()
        records = hw.import_board_receipts([path], formal=False)
        with self.assertRaisesRegex(ValueError, "source-verified"):
            hw.BoardCostBackend({"space_version": hw.SPACE}, None, list(records), formal=False)
        second = self.receipt("b")
        records = hw.import_board_receipts([path, second], formal=False)
        graph = dict(nodes=[dict(records[0]["shape"], id="a", precision=self.precision)])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            hw.query_graph_cost(graph, records, "W8A8", self.mapping, formal=False, identity={"space_version": hw.SPACE})

    def test_missing_exact_shape_returns_null_cost(self):
        records = hw.import_board_receipts([self.receipt()], formal=False)
        node = dict(records[0]["shape"], id="a", precision=self.precision)
        node["input_shapes"], node["output_shape"] = [[4, 80, 80]], [4, 40, 40]
        result = hw.query_graph_cost(dict(nodes=[node]), records, "W8A8", self.mapping,
                    formal=False, identity={"space_version": hw.SPACE})
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["cost"])
        self.assertEqual(len(result["missing"]), 1)

    def test_shape_holdout_no_self_prediction_or_family_fallback(self):
        paths = [self.receipt("a", height=40, compute=1000), self.receipt("b", height=80, compute=2000),
                 self.receipt("c", height=40, compute=1500, op="nearest")]
        records = hw.import_board_receipts(paths, formal=False)
        report = hw.validate_predictor(records)
        self.assertEqual(report["metrics"]["compute_us"]["predicted_count"], 2)
        self.assertEqual(report["metrics"]["compute_us"]["MAE"], 1)
        self.assertEqual(report["metrics"]["compute_us"]["MAPE"], 75)
        self.assertEqual(report["predictions"][2]["status"], "UNSUPPORTED_FAMILY")
        self.assertFalse(report["formal_search_ready"])
        with self.assertRaisesRegex(ValueError, "Ineligible"):
            hw.NearestMeasuredPredictor(records, formal=True)

    def test_repeated_shape_is_held_out_together(self):
        records = hw.import_board_receipts([self.receipt("a"), self.receipt("b")], formal=False)
        report = hw.validate_predictor(records)
        self.assertEqual(report["metrics"]["compute_us"]["predicted_count"], 0)

    def test_blur_constants_and_silu_lut_change_primitive_identity(self):
        shape = dict(op="blur", operator_type="DW", input_shapes=[[4, 40, 40]], output_shape=[4, 20, 20],
                     fixed_kernel=[[1,2,1],[2,4,2],[1,2,1]], denominator=16, id="blur")
        first = hw.primitive_record(shape, "W8A8")
        shape["denominator"] = 8
        self.assertNotEqual(first["primitive_id"], hw.primitive_record(shape, "W8A8")["primitive_id"])
        self.assertFalse(first["legacy_reuse_safe"])

    def test_static_mixed_boundary_shared_once_per_scale(self):
        levels = ["P2", "P3", "P4", "P5"]
        outputs = {level: "neck." + level for level in levels}
        edges = {value: [48, 160 // (2**i), 160 // (2**i)] for i, value in enumerate(outputs.values())}
        nodes = []
        for i, (level, source) in enumerate(outputs.items()):
            for branch in ("cv2", "cv3"):
                nodes.append(dict(id=f"head.{branch}.{i}.first", op="conv", operator_type="Conv", inputs=[source],
                         input_shapes=[edges[source]], output_shape=[48, *edges[source][1:]], Cin=48, Cout=48,
                         kernel=3, stride=1, padding=1, groups=1, activation="linear"))
        graph = dict(nodes=nodes, neck_outputs=outputs, edges=edges, graph_scope="STATIC_PROPOSED_HEAD_AUDIT")
        requirements = hw.graph_requirements(graph, "HEAD_W8A16")
        boundaries = [row for row in requirements if row["shape"]["op"] == "body8_to_head16"]
        self.assertEqual(len(boundaries), 4)
        for row in requirements[:8]:
            self.assertTrue(row["inputs"][0].endswith("input_requant"))
        graph.pop("graph_scope")
        with self.assertRaisesRegex(ValueError, "explicitly exported"):
            hw.graph_requirements(graph, "HEAD_W8A16")

    def test_mixed_numeric_contract_preserves_body_head_boundary_precision(self):
        contract = dict(body={"activation_bits": 8, "weight_bits": 8}, body_activation_bits=8,
                        body_weight_bits=8, head_input_bits=16, head_activation_bits=16,
                        head_weight_bits=16, accumulator_bits=32)
        def primitive(op, input_bits, output_bits, weight_bits):
            return dict(shape={"op": op}, precision=dict(input_bitwidth=input_bits,
                        output_bitwidth=output_bits, weight_bitwidth=weight_bits, accumulator_bitwidth=32))
        for value in (primitive("conv", 8, 8, 8), primitive("conv", 16, 16, 16),
                      primitive("requantize", 8, 16, None), primitive("silu", 16, 16, None)):
            self.assertTrue(hw.numeric_precision_matches(contract, value))
        for value in (primitive("conv", 8, 8, 16), primitive("conv", 16, 16, 8),
                      primitive("requantize", 16, 8, None), primitive("requantize", 8, 16, 8)):
            self.assertFalse(hw.numeric_precision_matches(contract, value))

    def test_unfrozen_policy_queue_does_not_alias_or_promote_observations(self):
        records = hw.import_board_receipts([self.receipt()], formal=False)
        graph = dict(nodes=[dict(records[0]["shape"], id="a", precision=self.precision)])
        inventory = hw.build_inventory({"space_version": hw.SPACE}, graph, profiles=["W8A8"])
        queue = hw.coverage_queue(inventory, records, {})
        self.assertEqual(queue["task_count"], 6)
        self.assertEqual(queue["eligible_task_count"], 0)
        mapped = hw.coverage_queue(inventory, records, {policy: self.mapping for policy in hw.POLICIES})
        self.assertEqual(mapped["task_count"], 1)
        self.assertEqual(mapped["eligible_task_count"], 0)
        self.assertFalse(mapped["formal_search_ready"])


if __name__ == "__main__":
    unittest.main()
