import unittest

from hwnas_fpga.hardware.lookup_table import (
    FormalLutStatusEntry,
    LutEntry,
    LutQueryEngine,
    LutTable,
    OpSpec,
    canonicalize_lut_op_name,
)


class OpSpecImplementationTests(unittest.TestCase):
    def test_opspec_normalizes_hls_operator_aliases(self) -> None:
        self.assertEqual(canonicalize_lut_op_name("conv_bn_relu6"), "conv")
        self.assertEqual(canonicalize_lut_op_name("inverted_residual"), "mbconv")
        self.assertEqual(canonicalize_lut_op_name("stem_conv_k3_s2"), "conv")
        self.assertEqual(canonicalize_lut_op_name("pw_conv"), "conv")

        self.assertEqual(
            OpSpec(
                op="conv_bn_relu6",
                kernel_size=3,
                in_channels=1,
                out_channels=32,
                stride=2,
                input_resolution=(224, 224),
            ).op,
            "conv",
        )
        self.assertEqual(
            OpSpec(
                op="inverted_residual",
                kernel_size=3,
                in_channels=16,
                out_channels=24,
                stride=2,
                expand_ratio=6,
                input_resolution=(112, 112),
            ).op,
            "mbconv",
        )

    def test_opspec_roundtrip_preserves_implementation_fields(self) -> None:
        spec = OpSpec(
            op="conv",
            kernel_size=3,
            in_channels=16,
            out_channels=24,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(56, 56),
            bitwidth=8,
            input_parallelism=2,
            output_parallelism=4,
            unroll_factor=3,
            target_clock_mhz=200.0,
        )

        restored = OpSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, restored)
        self.assertEqual(spec.shape_signature(), restored.shape_signature())
        self.assertEqual(spec.implementation_signature(), restored.implementation_signature())

    def test_query_engine_can_fallback_to_unique_shape_match(self) -> None:
        table = LutTable(
            [
                LutEntry(
                    op_spec=OpSpec(
                        op="conv",
                        kernel_size=1,
                        in_channels=24,
                        out_channels=32,
                        stride=1,
                        groups=1,
                        expand_ratio=1,
                        input_resolution=(56, 56),
                        bitwidth=8,
                        input_parallelism=1,
                        output_parallelism=1,
                        unroll_factor=1,
                        target_clock_mhz=200.0,
                    ),
                    latency_ms=0.5,
                    cycles=100_000,
                    dsp=8,
                    bram=2,
                    lut=1000,
                )
            ]
        )

        query = OpSpec(
            op="conv",
            kernel_size=1,
            in_channels=24,
            out_channels=32,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(56, 56),
        )
        entry = LutQueryEngine(table, enable_interpolation=False).query(query)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.cycles, 100_000)

    def test_query_engine_interpolates_between_out_channels(self) -> None:
        table = LutTable(
            [
                LutEntry(
                    op_spec=OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=32,
                        stride=1,
                        groups=1,
                        expand_ratio=1,
                        input_resolution=(56, 56),
                    ),
                    latency_ms=0.4,
                    cycles=80_000,
                    dsp=8,
                    bram=2,
                    lut=600,
                ),
                LutEntry(
                    op_spec=OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=64,
                        stride=1,
                        groups=1,
                        expand_ratio=1,
                        input_resolution=(56, 56),
                    ),
                    latency_ms=0.8,
                    cycles=160_000,
                    dsp=16,
                    bram=4,
                    lut=1200,
                ),
            ]
        )

        query = OpSpec(
            op="conv",
            kernel_size=3,
            in_channels=16,
            out_channels=48,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(56, 56),
        )
        entry = LutQueryEngine(table, enable_interpolation=True).query(query)

        self.assertIsNotNone(entry)
        if entry is None:
            self.fail("Expected interpolated LUT entry")
        self.assertEqual(entry.cycles, 120_000)
        self.assertEqual(entry.dsp, 12)
        self.assertEqual(entry.bram, 3)
        self.assertEqual(entry.lut, 900)

    def test_query_engine_returns_none_on_lut_miss_without_interpolation(self) -> None:
        table = LutTable(
            [
                LutEntry(
                    op_spec=OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=32,
                        stride=1,
                        groups=1,
                        expand_ratio=1,
                        input_resolution=(56, 56),
                    ),
                    latency_ms=0.4,
                    cycles=80_000,
                    dsp=8,
                    bram=2,
                    lut=600,
                ),
                LutEntry(
                    op_spec=OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=64,
                        stride=1,
                        groups=1,
                        expand_ratio=1,
                        input_resolution=(56, 56),
                    ),
                    latency_ms=0.8,
                    cycles=160_000,
                    dsp=16,
                    bram=4,
                    lut=1200,
                ),
            ]
        )

        query = OpSpec(
            op="conv",
            kernel_size=3,
            in_channels=16,
            out_channels=48,
            stride=1,
            groups=1,
            expand_ratio=1,
            input_resolution=(56, 56),
        )
        entry = LutQueryEngine(table, enable_interpolation=False).query(query)
        self.assertIsNone(entry)

    def test_strict_formal_lut_distinguishes_deferred_from_missing(self) -> None:
        measured_spec = OpSpec(
            op="mbconv_e3_k3",
            kernel_size=3,
            in_channels=16,
            out_channels=24,
            stride=2,
            groups=1,
            expand_ratio=3,
            input_resolution=(112, 112),
        )
        deferred_spec = OpSpec(
            op="mbconv_e3_k3",
            kernel_size=3,
            in_channels=24,
            out_channels=32,
            stride=2,
            groups=1,
            expand_ratio=3,
            input_resolution=(56, 56),
        )
        table = LutTable(
            [
                LutEntry(
                    op_spec=measured_spec,
                    latency_ms=26.2,
                    cycles=5_241_117,
                    dsp=13,
                    bram=56,
                    lut=7828,
                )
            ]
        )
        engine = LutQueryEngine(
            table,
            strict_formal_lut=True,
            formal_status_entries={
                measured_spec: FormalLutStatusEntry(
                    case_name="mbconv_stage2",
                    status="measured",
                    op_type="mbconv_e3_k3",
                    lookup_op="mbconv_e3_k3",
                    op_spec=measured_spec,
                ),
                deferred_spec: FormalLutStatusEntry(
                    case_name="mbconv_stage3",
                    status="defer_current_impl",
                    op_type="mbconv_e3_k3",
                    lookup_op="mbconv_e3_k3",
                    defer_reason="mbconv_e3_stage_ge_3",
                    op_spec=deferred_spec,
                ),
            },
        )

        measured = engine.query_with_status(measured_spec)
        self.assertEqual(measured.status, "measured")
        self.assertIsNotNone(measured.entry)
        self.assertEqual(measured.matched_by, "formal_exact")

        deferred = engine.query_with_status(deferred_spec)
        self.assertEqual(deferred.status, "defer_current_impl")
        self.assertIsNone(deferred.entry)
        self.assertEqual(deferred.status_entry.case_name, "mbconv_stage3")

        missing = engine.query_with_status(
            OpSpec(
                op="mbconv_e3_k3",
                kernel_size=3,
                in_channels=32,
                out_channels=64,
                stride=2,
                groups=1,
                expand_ratio=3,
                input_resolution=(28, 28),
            )
        )
        self.assertEqual(missing.status, "missing")
        self.assertIsNone(missing.entry)


if __name__ == "__main__":
    unittest.main()
