"""Forecast identity and missing-value guards for the actual UM response format."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_um_points import parse


def response(*rows):
    return "#START7777\n# TMFC TMEF VARN LEVEL VALUS\n" + "\n".join(rows) + "\n#7777END\n"


class UMResponseContract(unittest.TestCase):
    def test_utc_forecast_identity(self):
        rows = parse(response("2024070100 2024070200 4200 16777215 25.9487"), "2024070100")
        self.assertEqual(rows, [{"lead": 24, "variable": "tdsw", "value": 25.9487}])

    def test_absent_forecast_is_not_invented(self):
        rows = parse(response("2024070100 2024070200 6213 16777215 1"), "2024070100")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variable"], "tcam")

    def test_missing_sentinel_stays_missing(self):
        rows = parse(response("2024070100 2024070200 4200 16777215 -9999"), "2024070100")
        self.assertIsNone(rows[0]["value"])

    def test_wrong_initialization_rejected(self):
        with self.assertRaises(AssertionError):
            parse(response("2024070100 2024070200 4200 16777215 25"), "2024070112")

    def test_duplicate_forecast_rejected(self):
        line = "2024070100 2024070200 4200 16777215 25"
        with self.assertRaises(AssertionError):
            parse(response(line, line), "2024070100")

    def test_success_status_cannot_replace_complete_body(self):
        with self.assertRaises(ValueError):
            parse("#START7777\n2024070100 2024070200 4200 16777215 25", "2024070100")


if __name__ == "__main__":
    unittest.main()
