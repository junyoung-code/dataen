import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HAS_CATBOOST = importlib.util.find_spec('catboost') is not None
if HAS_CATBOOST:
    import run_catboost as runner
    from catboost import CatBoostRegressor


@unittest.skipUnless(HAS_CATBOOST, 'Install requirements-catboost.txt for CatBoost checks')
class CatBoostTests(unittest.TestCase):
    def rows(self):
        return [dict(window='tune', variant=name, weighted_group_mae=600-i)
                for i, name in enumerate(runner.CANDIDATES)]

    def test_selection_uses_only_complete_tune_candidates(self):
        rows = self.rows()
        self.assertEqual(runner.choose(rows)['variant'], rows[-1]['variant'])
        with self.assertRaises(ValueError):
            runner.choose(rows[:-1])
        with self.assertRaises(ValueError):
            runner.choose(rows + [rows[0]])
        rows[0]['window'] = 'confirmation'
        with self.assertRaises(ValueError):
            runner.choose(rows)

    def test_nonfinite_selection_rejected(self):
        for value in (float('nan'), float('inf')):
            rows = self.rows()
            rows[0]['weighted_group_mae'] = value
            with self.assertRaises(ValueError):
                runner.choose(rows)

    def test_frozen_selection_detects_configuration_or_group_changes(self):
        variant = next(iter(runner.CANDIDATES))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sites = root/'sites.csv'
            sites.write_text('site_id,current_group\nPV001,A\n')
            selection = dict(selection_window='tune', selected_variant=variant,
                             config=runner.CANDIDATES[variant], reference=runner.REFERENCE,
                             weight_17=0.75, feature_bundle='station_distance',
                             sites_sha256=runner.protocol.digest(sites))
            path = root/'selection.json'
            path.write_text(json.dumps(selection))
            with patch.object(runner.original, 'DATA', root):
                self.assertEqual(runner.read_selection(path)['selected_variant'], variant)
                selection['weight_17'] = 0.5
                path.write_text(json.dumps(selection))
                with self.assertRaises(ValueError):
                    runner.read_selection(path)
                selection['weight_17'] = 0.75
                path.write_text(json.dumps(selection))
                sites.write_text('site_id,current_group\nPV001,B\n')
                with self.assertRaises(ValueError):
                    runner.read_selection(path)

    def test_weighted_mae_training_missing_values_and_reload(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(256, 3)), columns=['cloud', 'temp', 'distance'])
        y = (0.5 + 0.2*x.cloud).clip(0, 1).to_numpy()
        x.loc[::9, 'temp'] = np.nan
        weights = np.linspace(0.5, 2, len(x))
        params = runner.model_parameters(dict(iterations=12, depth=3, learning_rate=0.1), 2)
        params['verbose'] = False
        model = CatBoostRegressor(**params).fit(x, y, sample_weight=weights)
        self.assertEqual(model.tree_count_, 12)
        self.assertEqual(model.get_all_params()['loss_function'], 'MAE')
        self.assertEqual(model.get_all_params()['leaf_estimation_method'], 'Gradient')
        self.assertEqual(model.feature_names_, list(x))
        prediction = model.predict(x)
        self.assertTrue(np.isfinite(prediction).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.cbm'
            model.save_model(str(path))
            restored = CatBoostRegressor().load_model(str(path))
            np.testing.assert_allclose(restored.predict(x), prediction, rtol=1e-12, atol=1e-12)

    def test_sample_weights_affect_fitted_predictions(self):
        x = pd.DataFrame({'x': np.repeat([0., 1.], 60)})
        y = np.tile(np.repeat([0.1, 0.9], [40, 20]), 2)
        params = runner.model_parameters(dict(iterations=30, depth=2, learning_rate=0.1), 2)
        params['verbose'] = False
        uniform = CatBoostRegressor(**params).fit(x, y)
        weighted = CatBoostRegressor(**params).fit(x, y, sample_weight=np.where(y > 0.5, 10., 1.))
        self.assertGreater(float(np.mean(weighted.predict(x)-uniform.predict(x))), 0.4)
        weighted_error = np.average(abs(weighted.predict(x)-y), weights=np.where(y > 0.5, 10., 1.))
        self.assertLess(weighted_error, 0.2)


if __name__ == '__main__':
    unittest.main()
