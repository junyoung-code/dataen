import unittest
import numpy as np
import pandas as pd
from collect_gefs_b1 import interval_values,hourly,summarize
from features_gefs_b1 import CFG,COLUMNS,attach,load_table
from pathlib import Path

class TestGefs(unittest.TestCase):
    def test_six_hour_products_deaveraged(self):
        x=np.array([10.,15.,30.,35.])[None,:,None,None]
        np.testing.assert_array_equal(interval_values(x,[39,42,45,48]).ravel(),[10,20,30,40])
        with self.assertRaises(ValueError):interval_values(x[:,:1],[42])

    def test_daily_endpoints_and_deadline(self):
        members={p:np.tile(np.arange(9),(31,1)) for p in ['gefs_sw','gefs_cloud']}
        rows=hourly('2023-06-01',members,CFG)
        self.assertEqual(str(rows.ts.iloc[0]),'2023-06-01 01:00:00+09:00')
        self.assertEqual(str(rows.ts.iloc[-1]),'2023-06-02 00:00:00+09:00')
        self.assertEqual(str(rows.init_time.iloc[0]),'2023-05-30 00:00:00+00:00')
        self.assertTrue((rows.deadline-rows.init_time==pd.Timedelta(hours=32)).all())
        np.testing.assert_array_equal(rows.gefs_sw_mean.values,np.repeat(np.arange(1,9),3))
        bad=CFG|{'availability_lag_hours':33}
        with self.assertRaises(ValueError):hourly('2023-06-01',members,bad)

    def test_fixed_region_reduction_and_ensemble(self):
        x=np.ones((31,9,2,2))*np.arange(31)[:,None,None,None]
        m=summarize(x,[32,40],'gefs_sw',CFG['lead_hours'])
        np.testing.assert_allclose(m,np.broadcast_to(np.arange(31)[:,None],(31,9)),atol=1e-12)
        x[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):summarize(x,[32,40],'gefs_sw',CFG['lead_hours'])

    def test_real_trial_join_and_missing_rejection(self):
        table=load_table(Path('outputs/b1_gefs_trial/hourly.parquet'))
        times=table.ts.iloc[:24].tolist()
        frame=pd.DataFrame({'site_id':['synthetic-a']*24+['synthetic-b']*24,'ts':times+times})
        merged=attach(frame,table)
        np.testing.assert_array_equal(merged[COLUMNS].iloc[:24],merged[COLUMNS].iloc[24:])
        with self.assertRaises(ValueError):attach(frame,table.iloc[1:])
        with self.assertRaises(ValueError):attach(frame,pd.concat([table,table.iloc[:1]]))

if __name__=='__main__':unittest.main()
