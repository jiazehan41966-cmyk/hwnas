import unittest

from hwnas_fpga.hardware.lookup_table import (
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


if __name__ == "__main__":
    unittest.main()
