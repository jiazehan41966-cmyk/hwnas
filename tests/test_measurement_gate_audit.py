import unittest

from scripts.audit_measurement_first_gates import g1_status


class MeasurementGateAuditTests(unittest.TestCase):
    def test_missing_g1_runs_are_pending_not_fabricated(self) -> None:
        result = g1_status(__import__("pathlib").Path("Z:/definitely_missing"))
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(result["completed_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
