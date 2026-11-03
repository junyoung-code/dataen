import unittest
import numpy as np
import pandas as pd
from training_weights import make_weights, choose, VARIANTS


class WeightTests(unittest.TestCase):
    def fixture(self):
        times=pd.DatetimeIndex(['2024-01-01 10:00','2024-01-01 11:00','2024-01-01 15:00','2024-01-01 16:00'],tz='Asia/Seoul')
        rows=pd.DataFrame(dict(ts=times,cap=[100.,200.,100.,100.]),index=[7,4,9,1])
        prices=pd.DataFrame(dict(ts=times,smp=[0.,2.,4.,1.]))
        return rows,prices,times[0],times[-1]+pd.Timedelta(hours=1)

    def test_capacity_converts_normalized_absolute_error_to_kwh(self):
        rows,prices,a,b=self.fixture();w,info=make_weights(rows,prices,'capacity',a,b)
        normalized_error=np.array([.1,.2,.3,.4])
        self.assertAlmostEqual(float(np.sum(w*normalized_error)*info['raw_mean']),float(np.sum(rows.cap*normalized_error)))
        self.assertAlmostEqual(float(w.mean()),1.)

    def test_peak_boundaries_zero_price_and_order(self):
        rows,prices,a,b=self.fixture()
        w,info=make_weights(rows,prices.sample(frac=1,random_state=0),'capacity_peak_smp',a,b)
        raw=np.array([0.,1200.,1200.,100.]);np.testing.assert_allclose(w,raw/raw.mean())
        self.assertEqual(info['zero_rows'],1)
        # UTC representation must still use KST hour boundaries.
        rows.ts=rows.ts.dt.tz_convert('UTC')
        other,_=make_weights(rows,prices,'capacity_peak_smp',a,b);np.testing.assert_allclose(w,other)

    def test_future_prices_cannot_change_training_weights(self):
        rows,prices,a,b=self.fixture();w,_=make_weights(rows,prices,'capacity_peak_smp',a,b)
        future=pd.DataFrame(dict(ts=[b,b],smp=[np.nan,-999.]))
        other,_=make_weights(rows,pd.concat([prices,future]),'capacity_peak_smp',a,b)
        np.testing.assert_array_equal(w,other)
        rows.loc[7,'ts']=b
        with self.assertRaises(ValueError):make_weights(rows,prices,'capacity_peak_smp',a,b)

    def test_invalid_weights_and_prices_fail(self):
        rows,prices,a,b=self.fixture()
        for bad in [prices.iloc[:-1],pd.concat([prices,prices.iloc[:1]]),prices.assign(smp=0.),prices.assign(smp=-1.)]:
            with self.assertRaises(ValueError):make_weights(rows,bad,'capacity_peak_smp',a,b)
        rows.loc[7,'cap']=np.nan
        with self.assertRaises(ValueError):make_weights(rows,prices,'capacity',a,b)

    def test_uniform_and_selection_guards(self):
        rows,prices,a,b=self.fixture();w,_=make_weights(rows,prices,'uniform',a,b);self.assertIsNone(w)
        scores=[dict(variant=k,weighted_group_mae=100.) for k in VARIANTS]
        self.assertEqual(choose(scores)['variant'],'l2_uniform')
        scores[-1]['weighted_group_mae']=90.
        self.assertEqual(choose(scores)['variant'],'l1_capacity_peak_smp')
        for bad in [scores[:-1],scores[:-1]+[scores[0]]]:
            with self.assertRaises(ValueError):choose(bad)
        scores[0]['weighted_group_mae']=np.nan
        with self.assertRaises(ValueError):choose(scores)


if __name__=='__main__':unittest.main()
