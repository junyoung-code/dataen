"""Submission mistakes must fail locally before consuming a submission slot."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_s929_um_submission import UM_COLS,verify_submission


class SubmissionFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.data=self.root/"data";self.data.mkdir();self.dest=self.root/"submission";self.dest.mkdir()
        sites=[f"P{i:03}" for i in range(253)]
        pd.DataFrame({"site_id":sites,"capacity_kw":200.}).to_csv(self.data/"sites.csv",index=False)
        self.comp=pd.DataFrame({"site_id":sites,"group":["G1"]*126+["G2"]*127})
        self.comp.to_csv(self.dest/"composition.csv",index=False)
        ts=pd.date_range("2025-06-01 01:00",periods=8760,freq="h",tz="Asia/Seoul")
        pd.DataFrame({"ts":ts}).to_csv(self.data/"smp.csv",index=False)
        self.base=pd.DataFrame({"group":np.repeat(["G1","G2"],8760),"ts":list(ts)*2,"pred":0.})
        self.result=self.base.assign(source="S929")
        self.fc=self.base.rename(columns={"pred":"forecast_kwh"})
        self.fc.to_csv(self.dest/"forecast.csv",index=False)
        self.raw=pd.DataFrame({"ts":ts[:1],**{c:[np.nan] for c in UM_COLS}})

    def tearDown(self):self.tmp.cleanup()
    def check(self):return verify_submission(self.dest,self.data,self.result,self.base,self.raw)

    def test_valid_format(self):self.assertEqual(self.check()["rows"],17520)

    def test_missing_hour_rejected(self):
        self.fc.iloc[1:].to_csv(self.dest/"forecast.csv",index=False)
        with self.assertRaises(AssertionError):self.check()

    def test_missing_timezone_rejected(self):
        self.fc["ts"]=self.fc.ts.dt.tz_localize(None)
        self.fc.to_csv(self.dest/"forecast.csv",index=False)
        with self.assertRaises(AssertionError):self.check()

    def test_infeasible_group_rejected(self):
        self.comp["group"]=["G1"]*20+["G2"]*233
        self.comp.to_csv(self.dest/"composition.csv",index=False)
        with self.assertRaises(AssertionError):self.check()

    def test_changed_fallback_value_rejected(self):
        self.fc.loc[0,"forecast_kwh"]=1.
        self.fc.to_csv(self.dest/"forecast.csv",index=False)
        with self.assertRaises(AssertionError):self.check()


if __name__=="__main__":unittest.main()
