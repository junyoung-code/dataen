import unittest
import numpy as np
import pandas as pd
import train_smoke as original
import run_p3
from features_p3 import build, extra_columns


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.b=original.load_baseline()

    def fixture(self):
        t=pd.Timestamp('2025-01-02 12:00',tz='Asia/Seoul')
        sites=pd.DataFrame(dict(site_id=['a','b'],current_group=['A','B'],capacity_kw=[100.,200.],
                                lat=[35.,36.],lon=[127.,128.],wa_station=['a','b'],wb_station=['a','b'],
                                wa_dist=[100.,200.],wb_dist=[300.,400.]))
        gen=pd.DataFrame(dict(site_id=['b','a'],ts=[t,t],energy=[20.,10.]),index=[7,3])
        forecasts=[]
        for station,cols in [('wa_station',self.b.WA_COLS),('wb_station',self.b.WB_COLS)]:
            f=pd.DataFrame({station:['a','a','b','b'],'round':[10,17,10,17],'ts':[t]*4})
            for c in cols:f[c]=[2.,5.,10.,7.]
            forecasts.append(f.sample(frac=1,random_state=2))
        return gen,sites,*forecasts

    def test_revision_keys_sign_and_label_independence(self):
        gen,sites,fa,fb=self.fixture()
        d,cols=build(self.b,gen,sites,fa,fb,10,'revision')
        np.testing.assert_array_equal(d.revision_wa_cloud,[-3.,3.])
        self.assertEqual(len(cols),54)
        changed=gen.copy();changed.energy=[-999.,999.]
        future=fa.copy();future.ts+=pd.Timedelta(days=1);future.wa_cloud=999.
        e,_=build(self.b,changed,sites,pd.concat([fa,future]),fb,10,'revision')
        pd.testing.assert_frame_equal(d[cols],e[cols])

    def test_absent_issue_stays_nan_and_duplicate_keys_fail(self):
        gen,sites,fa,fb=self.fixture()
        fa=fa.loc[~((fa.wa_station=='a')&(fa['round']==17))]
        d,_=build(self.b,gen,sites,fa,fb,10,'revision')
        self.assertTrue(np.isnan(d.loc[1,'revision_wa_cloud']))
        self.assertEqual(d.loc[1,'revision_wa_cloud_missing'],1)
        with self.assertRaises(ValueError):build(self.b,gen,sites,pd.concat([fa,fa.iloc[:1]]),fb,10,'revision')

    def test_provider_units_and_distances(self):
        gen,sites,fa,fb=self.fixture();fa['wa_wind_speed']=10.;fb['wb_wind_speed']=36.
        d,cols=build(self.b,gen,sites,fa,fb,17,'provider_gap')
        np.testing.assert_allclose(d.gap_wind_ms,0.)
        self.assertEqual(len(cols),32)
        d,cols=build(self.b,gen,sites,fa,fb,17,'station_distance')
        np.testing.assert_array_equal(d.wa_dist,[200.,100.]);self.assertEqual(len(cols),26)
        sites.loc[0,'wa_dist']=-1
        with self.assertRaises(ValueError):build(self.b,gen,sites,fa,fb,17,'station_distance')

    def test_baseline_features_and_no_label_columns(self):
        gen,sites,fa,fb=self.fixture()
        expected=self.b.features(gen,sites,fa,fb,None,10)
        d,cols=build(self.b,gen,sites,fa,fb,10,'baseline')
        pd.testing.assert_frame_equal(d[cols],expected[self.b.FEATS])
        for bundle in run_p3.BUNDLES:
            self.assertNotIn('energy',self.b.FEATS+extra_columns(self.b,bundle))

    def test_selection_favors_baseline_on_tie_and_rejects_bad_grid(self):
        rows=[dict(bundle=b,feature_count=n,weighted_group_mae=100.)
              for b,n in zip(run_p3.BUNDLES,[24,54,32,26])]
        self.assertEqual(run_p3.choose(rows)['bundle'],'baseline')
        rows[1]['weighted_group_mae']=90.
        self.assertEqual(run_p3.choose(rows)['bundle'],'revision')
        for bad in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(ValueError):run_p3.choose(bad)
        rows[0]['weighted_group_mae']=np.nan
        with self.assertRaises(ValueError):run_p3.choose(rows)


if __name__=='__main__':unittest.main()
