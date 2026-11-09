"""Fallback must preserve baseline predictions and reject unsafe alignment."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_s929_um_submission import UM_COLS, available, fallback, label_calibrated_groups


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.ts=pd.date_range("2026-04-01 09:00",periods=4,freq="h",tz="Asia/Seoul")
        self.raw=pd.DataFrame({"ts":self.ts[:3],**{c:[1.,2.,3.] for c in UM_COLS}})
        self.raw.loc[1,UM_COLS[0]]=np.nan
        self.raw.loc[2,UM_COLS[1]]=np.inf
        self.base=pd.DataFrame({"group":["G1"]*4,"ts":self.ts,"pred":[10.1,20.2,30.3,40.4]})
        self.cand=self.base.assign(pred=99.)

    def test_partial_missing_absent_and_infinite_use_exact_baseline(self):
        result=fallback(self.base,self.cand.iloc[::-1],self.raw)
        np.testing.assert_array_equal(result.pred,[99.,20.2,30.3,40.4])
        self.assertEqual(result.source.tolist(),["UM_raw18","S929","S929","S929"])

    def test_zero_is_observed_value_not_missing(self):
        self.raw.loc[0,UM_COLS]=0
        self.assertTrue(available(self.raw,self.ts)[0])

    def test_duplicate_and_mismatched_keys_rejected(self):
        with self.assertRaises(AssertionError):fallback(self.base,self.cand.iloc[:3],self.raw)
        with self.assertRaises(AssertionError):fallback(self.base,pd.concat([self.cand,self.cand.iloc[:1]]),self.raw)
        with self.assertRaises(AssertionError):available(pd.concat([self.raw,self.raw.iloc[:1]]),self.ts)

    def test_nonfinite_prediction_rejected_even_on_fallback_hour(self):
        self.cand.loc[1,"pred"]=np.nan
        with self.assertRaises(AssertionError):fallback(self.base,self.cand,self.raw)

    def test_calibration_group_codes_match_composition(self):
        comp=pd.DataFrame({"site_id":["a","b"],"group":["G2","G1"]})
        frame=pd.DataFrame({"site_id":["a","b"],"group":[1,0]})
        self.assertEqual(label_calibrated_groups(frame,comp).group.tolist(),["G2","G1"])
        with self.assertRaises(AssertionError):
            label_calibrated_groups(frame.assign(group=[0,1]),comp)


if __name__=="__main__":unittest.main()
