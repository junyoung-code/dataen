"""Only independently reviewed source gaps may unblock the comparison."""
import unittest

from scripts.continue_weather_ablation import reviewed_ens_gap


class ReviewedGapTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"snapshot_id": "frozen", "signature": "config"}
        self.review = {
            "source": "ENS", "status": "retry_failed", "snapshot_id": "frozen",
            "signature": "config", "day": "2024-09-07", "error_type": "ValueError",
            "error": "Missing/nonfinite source field solar: 9/10",
            "variables": {"solar": {"nonfinite_count": 1}},
        }
        self.summary = {
            "status": "failed", "requested_dates": ["2024-09-06", "2024-09-07"],
            "completed": [{"day": "2024-09-06"}],
            "errors": [{"day": "2024-09-07", "type": "ValueError",
                        "error": self.review["error"]}],
        }

    def test_reviewed_source_gap_can_proceed_without_imputation(self):
        self.assertTrue(reviewed_ens_gap(self.summary, self.review, self.dataset))

    def test_unreviewed_errors_or_missing_successes_stop_pipeline(self):
        for changes in ({"errors": []}, {"completed": []},
                        {"errors": self.summary["errors"] + [{"day": "new"}]},
                        {"completed": self.summary["completed"] * 2}):
            with self.subTest(changes=changes):
                self.assertFalse(reviewed_ens_gap(self.summary | changes, self.review, self.dataset))

    def test_wrong_snapshot_config_or_unconfirmed_gap_stops_pipeline(self):
        for changes in ({"snapshot_id": "other"}, {"signature": "other"},
                        {"status": "retry_passed"}, {"variables": {}},
                        {"day": "2024-09-08"}, {"error": "network timeout"}):
            with self.subTest(changes=changes):
                self.assertFalse(reviewed_ens_gap(self.summary, self.review | changes, self.dataset))


if __name__ == "__main__":
    unittest.main()
