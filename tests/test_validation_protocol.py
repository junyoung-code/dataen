import unittest
import numpy as np
import pandas as pd
import validation_protocol as p


class ProtocolTests(unittest.TestCase):
    def test_boundaries_and_embargo(self):
        for name in p.WINDOWS:
            a,b,c,d=p.bounds(name)
            frame=pd.DataFrame({'ts':[a-pd.Timedelta(hours=1),a,b-pd.Timedelta(hours=1),b,c-pd.Timedelta(hours=1),c,d-pd.Timedelta(hours=1),d]})
            train,valid=p.split(frame,name)
            self.assertEqual(train.index.tolist(),[1,2])
            self.assertEqual(valid.index.tolist(),[5,6])
            self.assertLess(train.ts.max(),c.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17))

    def test_missing_labels_are_not_zero_and_mask_ignores_grouping(self):
        t=pd.date_range('2025-01-01',periods=2,freq='h',tz=p.TZ)
        rows=pd.DataFrame({'site_id':['a','b','a'],'ts':[t[0],t[0],t[1]],'energy':[0.,2.,3.]})
        self.assertEqual(p.complete_hours(rows,['a','b']).tolist(),[t[0]])
        rows['group']=['G2','G1','G2']
        self.assertEqual(p.complete_hours(rows,['a','b']).tolist(),[t[0]])
        with self.assertRaises(ValueError): p.complete_hours(pd.concat([rows,rows.iloc[:1]]),['a','b'])
        rows.loc[0,'energy']=np.nan
        with self.assertRaises(ValueError): p.complete_hours(rows,['a','b'])

    def fixture(self):
        hours=pd.DatetimeIndex(['2025-01-01 10:00','2025-01-01 11:00'],tz=p.TZ)
        frame=pd.DataFrame({'group':['G1','G1','G2','G2'],'ts':list(hours)*2,'energy':[0.,20.,0.,20.],
                            'prediction':[2.,14.,2.,14.],'smp':[10.,20.,10.,20.]})
        return frame,hours

    def test_parquet_timestamp_precision_is_canonicalized(self):
        hours=pd.date_range('2025-01-01 01:00',periods=2,freq='h',tz=p.TZ)
        rows=pd.DataFrame({'site_id':['a','a'],'ts':hours.as_unit('ms'),'energy':[1.,2.]})
        self.assertTrue(p.complete_hours(rows,['a']).equals(hours))

    def test_weighted_score_hand_calculation(self):
        frame,hours=self.fixture()
        self.assertAlmostEqual(p.weighted_group_mae(frame,hours), (10*2+20*3*6)/(10+20*3))
        self.assertAlmostEqual(p.weighted_group_mae(frame.sample(frac=1,random_state=0),hours),38/7)

    def test_rejects_changed_mask_duplicate_and_invalid_prices(self):
        frame,hours=self.fixture()
        for bad in [frame.iloc[:-1],pd.concat([frame,frame.iloc[:1]])]:
            with self.assertRaises(ValueError): p.weighted_group_mae(bad,hours)
        for price in [-1.,0.,np.nan]:
            bad=frame.copy(); bad['smp']=price
            with self.assertRaises(ValueError): p.weighted_group_mae(bad,hours)
        bad=frame.copy(); bad.loc[0,'smp']=12.
        with self.assertRaises(ValueError): p.weighted_group_mae(bad,hours)

    def test_peak_hour_edges_and_zero_price(self):
        hours=pd.DatetimeIndex(['2025-01-01 10:00','2025-01-01 11:00','2025-01-01 15:00','2025-01-01 16:00'],tz=p.TZ)
        frame=pd.DataFrame({'group':['G1']*4+['G2']*4,'ts':list(hours)*2,'energy':[0.]*8,
                            'prediction':[100.,2.,4.,8.]*2,'smp':[0.,1.,1.,1.]*2})
        self.assertAlmostEqual(p.weighted_group_mae(frame,hours),26/7)


if __name__=='__main__': unittest.main()
