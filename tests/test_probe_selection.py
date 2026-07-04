import unittest

from scripts.select_hw_calibration_probes import select_maximin


class ProbeSelectionTests(unittest.TestCase):
    def test_maximin_selects_requested_unique_rows(self) -> None:
        rows = [
            {"features": {"a": float(index), "b": float(index % 2)}}
            for index in range(10)
        ]
        selected = select_maximin(rows, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)


if __name__ == "__main__":
    unittest.main()
