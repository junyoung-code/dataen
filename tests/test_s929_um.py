"""Key/mask/capacity mistakes must fail instead of silently changing peer W."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_s929_um import aligned_prediction, direct_cf


class S929AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "input.parquet"
        self.ts = pd.date_range("2024-06-01 09:00", periods=2, freq="h", tz="Asia/Seoul")
        self.d = {"sites": np.array(["a", "b"]), "ts": self.ts,
                  "actual": np.array([[1., 2.], [3., 0.]]),
                  "has": np.array([[True, True], [True, False]]),
                  "cap": np.array([10., 20.]), "in_span": np.array([True, True])}
        self.frame = pd.DataFrame({"site_id": ["a", "a", "b", "b"], "ts": list(self.ts) * 2,
                                  "actual": [1., 2., 3., np.nan], "gen_missing": [0, 0, 0, 1],
                                  "pred": [5., 6., 7., 8.]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_shuffled_keys_align_by_site_and_time(self):
        self.frame.iloc[[3, 0, 2, 1]].to_parquet(self.path)
        np.testing.assert_array_equal(aligned_prediction(self.path, self.d), [[5, 6], [7, 8]])

    def test_duplicate_keys_fail(self):
        pd.concat([self.frame, self.frame.iloc[:1]]).to_parquet(self.path)
        with self.assertRaises(AssertionError):
            aligned_prediction(self.path, self.d)

    def test_changed_observation_mask_fails(self):
        self.frame.loc[0, "gen_missing"] = 1
        self.frame.to_parquet(self.path)
        with self.assertRaises(AssertionError):
            aligned_prediction(self.path, self.d)

    def test_nonfinite_prediction_fails(self):
        self.frame.loc[1, "pred"] = np.nan
        self.frame.to_parquet(self.path)
        with self.assertRaises(AssertionError):
            aligned_prediction(self.path, self.d)

    def test_full_grid_predictions_use_peer_row_support(self):
        # Peer OOF omits whole plants AND hours outside their observation span.
        self.d['actual'] = np.array([[1., 0.], [0., 0.]])
        self.d['has'] = np.array([[True, False], [False, False]])
        self.d['in_span'] = np.array([True, False])
        self.d['prediction_support'] = np.array([[True, False], [False, False]])
        self.frame.loc[1:, 'actual'] = np.nan
        self.frame.loc[1:, 'gen_missing'] = 1
        self.frame.to_parquet(self.path)
        np.testing.assert_array_equal(aligned_prediction(self.path, self.d), [[5., 0.], [0., 0.]])

    def test_missing_unobserved_peer_row_still_fails(self):
        self.d['prediction_support'] = np.ones((2, 2), bool)
        self.frame.iloc[:3].to_parquet(self.path)
        with self.assertRaises(AssertionError):
            aligned_prediction(self.path, self.d)

    def test_direct_is_cf_times_observed_capacity(self):
        frame = pd.DataFrame({"group": ["G1", "G2"], "ts": [self.ts[0]] * 2,
                              "cf": [.5, .4], "cap_has": [10., 20.], "pred": [5., 8.]})
        frame.to_parquet(self.path)
        np.testing.assert_array_equal(direct_cf(self.path, self.d, np.array([0, 1])), [[.5, 0], [.4, 0]])
        frame.loc[1, "cap_has"] = 21.
        frame.to_parquet(self.path)
        with self.assertRaises(AssertionError):
            direct_cf(self.path, self.d, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
