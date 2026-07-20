import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.hardware.hls_evidence import (
    candidate_roles,
    load_status_index,
    normalized_op_key,
)
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


class OpKeyTests(unittest.TestCase):
    def test_target_clock_is_not_part_of_query_key(self) -> None:
        spec = {
            "op": "skip",
            "kernel_size": 1,
            "in_channels": 16,
            "out_channels": 16,
            "stride": 1,
            "groups": 1,
            "expand_ratio": 1,
            "input_resolution": [28, 28],
            "bitwidth": 8,
            "input_parallelism": 1,
            "output_parallelism": 1,
            "unroll_factor": 1,
        }
        self.assertEqual(
            normalized_op_key({**spec, "target_clock_mhz": None}),
            normalized_op_key({**spec, "target_clock_mhz": 200}),
        )

    def test_status_index_rejects_ambiguous_key(self) -> None:
        spec = {
            "op": "skip",
            "kernel_size": 1,
            "in_channels": 16,
            "out_channels": 16,
            "stride": 1,
            "groups": 1,
            "expand_ratio": 1,
            "input_resolution": [28, 28],
            "bitwidth": 8,
            "input_parallelism": 1,
            "output_parallelism": 1,
            "unroll_factor": 1,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "status.json"
            path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"case_name": "a", "op_spec": spec},
                            {"case_name": "b", "op_spec": spec},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_status_index(path)


class RoleTests(unittest.TestCase):
    def test_baseline_roles_include_stem_head_gap_classifier(self) -> None:
        space = SearchSpace(
            SearchSpaceConfig(
                input_channels=1,
                image_size=32,
                stem_channels=8,
                stage_strides=(1,),
                stage_channel_choices=((8,),),
                stage_depth_choices=((1,),),
                kernel_choices=(3,),
                expand_choices=(1,),
                op_choices=("mbconv",),
                head_conv_channels=16,
                num_classes=8,
            )
        )
        architecture = space.baseline_architecture()
        roles = candidate_roles(architecture, space)
        names = [role["role"] for role in roles]
        self.assertEqual(names[0], "stem")
        self.assertIn("head_conv", names)
        self.assertEqual(names[-2:], ["global_avg_pool", "classifier"])


if __name__ == "__main__":
    unittest.main()
