import unittest

from hls_lut_builder.board_harness.scripts.run_dynamic_validation import (
    summarize_records,
)


class SummaryTests(unittest.TestCase):
    def test_complete_exact_records_are_claimable(self) -> None:
        records = [
            {
                "sample_id": 1,
                "label": 0,
                "argmax": 0,
                "status": 0,
                "numeric_match": True,
                "cycles": 10,
            },
            {
                "sample_id": 2,
                "label": 1,
                "argmax": 1,
                "status": 0,
                "numeric_match": True,
                "cycles": 12,
            },
        ]
        result = summarize_records(records, expected_sample_ids={1, 2})
        self.assertTrue(result["claimable"])
        self.assertEqual(result["top1"], 1.0)

    def test_missing_or_mismatch_blocks_claim(self) -> None:
        records = [
            {
                "sample_id": 1,
                "label": 0,
                "argmax": 0,
                "status": 0,
                "numeric_match": False,
                "cycles": 10,
            }
        ]
        result = summarize_records(records, expected_sample_ids={1, 2})
        self.assertFalse(result["claimable"])
        self.assertEqual(result["missing_sample_ids"], [2])
        self.assertEqual(result["numeric_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
