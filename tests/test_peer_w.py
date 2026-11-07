import unittest

import numpy as np
import pandas as pd

from run_peer_w import bounds, peer_labels, score, parameters


class PeerWTests(unittest.TestCase):
    def test_annual_windows_and_first_issuance(self):
        for fold,n in [('F2',8784),('F3',8760)]:
            a,b,c,d=bounds(fold)
            self.assertEqual(len(pd.date_range(c,d,freq='h',inclusive='left')),n)
            self.assertEqual(b,c.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17))
            self.assertEqual(a,c-pd.DateOffset(years=2))
            self.assertEqual(c-b,pd.Timedelta(hours=8))

    def test_peer_invalid_labels_are_excluded_not_zeroed(self):
        y,has=peer_labels([0,105,106,-1,np.nan],[100]*5)
        np.testing.assert_array_equal(has,[True,True,False,False,False])
        self.assertTrue(np.isnan(y[2:]).all())

    def test_group_cancellation_missing_mask_and_peak_hour(self):
        ts=pd.date_range('2024-06-01 10:00',periods=2,freq='h',tz='Asia/Seoul')
        sites=pd.DataFrame({'site_id':['a','b','c'],'current_group':['A','A','B']})
        # a +5 and b -5 cancel in G1; invalid c at 11:00 must contribute nothing.
        rows=pd.DataFrame({'site_id':['a','b','c','c'],'ts':[ts[0],ts[0],ts[0],ts[1]],
                           'energy_raw':[10.,10.,10.,106.], 'energy_peer':[10.,10.,10.,np.nan],
                           'has_peer':[True,True,True,False], 'seen':[True,False,True,True],
                           'ensemble':[15.,5.,12.,999.]})
        prices=pd.DataFrame({'ts':ts,'smp':[100.,100.]})
        m,g=score(rows,sites,prices,ts[0],ts[-1]+pd.Timedelta(hours=1))
        self.assertEqual(m['W'],200.)
        self.assertEqual(m['weight_sum'],800.)
        self.assertEqual(m['weighted_mae_kwh'],.25)
        self.assertEqual(m['observed_site_hours'],3)
        before=rows.copy(deep=True)
        m,_=score(rows,sites,prices,ts[0],ts[-1]+pd.Timedelta(hours=1),seen_only=True)
        self.assertEqual(m['W'],700.)
        pd.testing.assert_frame_equal(rows,before)

    def test_no_silent_duplicate_or_missing_price(self):
        ts=pd.Timestamp('2024-06-01 11:00',tz='Asia/Seoul')
        sites=pd.DataFrame({'site_id':['a'],'current_group':['A']})
        rows=pd.DataFrame({'site_id':['a'],'ts':[ts],'energy_raw':[0.],'energy_peer':[0.],
                           'has_peer':[True],'seen':[True],'ensemble':[1.]})
        prices=pd.DataFrame({'ts':[ts],'smp':[100.]})
        with self.assertRaises(ValueError):
            score(pd.concat([rows,rows]),sites,prices,ts,ts+pd.Timedelta(hours=1))
        with self.assertRaises(ValueError):
            score(rows,sites,prices,ts,ts+pd.Timedelta(hours=2))


if __name__=='__main__':
    unittest.main()
