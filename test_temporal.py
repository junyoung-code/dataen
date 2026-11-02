"""Guard against temporal leakage and misleading missing-label metrics."""
import unittest
import numpy as np
import pandas as pd

import evaluate_temporal as e


class TemporalTests(unittest.TestCase):
    def test_split_excludes_embargo_and_future(self):
        ts = pd.to_datetime(['2023-06-01 00:00', '2023-06-01 01:00',
                             '2024-05-31 00:00', '2024-05-31 01:00',
                             '2024-06-01 00:00', '2024-06-01 01:00',
                             '2025-06-01 00:00', '2025-06-01 01:00']).tz_localize('Asia/Seoul')
        train, valid = e.split_data(pd.DataFrame({'ts': ts, 'row': range(8)}))
        self.assertEqual(train.row.tolist(), [1, 2])
        self.assertEqual(valid.row.tolist(), [5, 6])

    def test_metrics_use_raw_labels_and_correct_denominators(self):
        m = e.metrics(np.array([0., 20.]), np.array([2., 14.]), np.array([10., 10.]))
        self.assertEqual(m['mae_kwh'], 4.)
        self.assertAlmostEqual(m['rmse_kwh'], np.sqrt(20.))
        self.assertEqual(m['nmae_capacity_pct'], 40.)
        self.assertEqual(m['wape_pct'], 40.)
        self.assertEqual(m['bias_kwh'], -2.)

    def test_zero_actual_and_empty_metrics_do_not_divide_by_zero(self):
        self.assertIsNone(e.metrics(np.array([0.]), np.array([0.]), np.array([10.]))['wape_pct'])
        self.assertEqual(e.metrics(np.array([]), np.array([]), np.array([]))['rows'], 0)

    def test_missing_site_does_not_become_complete_group_or_zero_label(self):
        ts = pd.Timestamp('2024-06-01 12:00', tz='Asia/Seoul')
        sites = pd.DataFrame({'site_id': ['a','b'], 'current_group': ['A','A'], 'capacity_kw': [10.,20.]})
        rows = pd.DataFrame({'site_id': ['a'], 'ts': [ts], 'energy': [4.], 'pred': [5.]})
        result = e.aggregate_groups(rows, sites, ['pred'])
        self.assertFalse(bool(result.complete.iloc[0]))
        self.assertEqual(result.energy.iloc[0], 4.)
        self.assertEqual(result.observed_capacity_kw.iloc[0], 10.)
        both = pd.concat([rows, pd.DataFrame({'site_id':['b'], 'ts':[ts], 'energy':[8.], 'pred':[9.]})])
        complete = e.aggregate_groups(both, sites, ['pred'])
        self.assertTrue(bool(complete.complete.iloc[0]))
        self.assertEqual(complete.energy.iloc[0], 12.)
        self.assertEqual(complete.pred.iloc[0], 14.)

    def test_climatology_falls_back_using_only_training_labels(self):
        train = pd.DataFrame({'site_id':['a','a','b'], 'month':[6,7,6], 'hour':[12,12,12],
                              'energy':[2.,4.,12.], 'capacity_kw':[10.,10.,20.]})
        valid = pd.DataFrame({'site_id':['a','a','c','c'], 'month':[6,8,6,8], 'hour':[12,12,12,3],
                              'energy':[999.,999.,999.,999.], 'capacity_kw':[10.,10.,10.,10.]})
        pred = e.climatology_predict(train, valid)
        np.testing.assert_allclose(pred, [2.,3.,4.,4.])


if __name__ == '__main__':
    unittest.main()
