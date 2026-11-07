"""Checks for radiation units, hour boundaries and forecast availability."""
import unittest
import numpy as np
import pandas as pd
from weather_ablation_features import interval_flux, time_axis, expand, spatial_stats


class ForecastFeatureTests(unittest.TestCase):
    def test_accumulated_and_running_mean_recover_interval_energy(self):
        leads = np.array([36, 42, 48])
        # Integral up to lead36 is3600 Wh/m2; next intervals are100 then400 W/m2.
        wh = np.array([[3600., 7200.], [4200., 8400.], [6600., 13200.]])
        expected = np.array([[100., 200.], [400., 800.]])
        np.testing.assert_allclose(interval_flux(wh * 3600, leads, "accumulated_joules"), expected)
        np.testing.assert_allclose(interval_flux(wh / leads[:, None], leads, "since_init_mean"), expected)
        np.testing.assert_allclose(expected.sum(0) * 6, wh[-1] - wh[0])

    def test_ens_means_are_not_deaveraged_and_hour_end_uses_left_interval(self):
        leads = np.arange(39, 64, 3)
        times, target = time_axis("2024-04-03", "2024-04-01T00:00Z", leads, 2, 24)
        raw = np.arange(9.)[:, None]
        hourly = expand(interval_flux(raw, leads, "previous_interval_mean"), times, target, interval=True)
        np.testing.assert_array_equal(hourly[:, 0], np.repeat(np.arange(1, 9), 3))
        self.assertEqual(str(target[-1]), "2024-04-04 00:00:00+09:00")

    def test_statistic_is_computed_after_radiation_difference(self):
        # Two cells exchange radiation: accumulated SD is zero at the end, interval SD is10.
        accumulated = np.array([[0., 20.], [20., 20.]]) * 3600
        flux = interval_flux(accumulated, [1, 2], "accumulated_joules")
        np.testing.assert_allclose(spatial_stats(flux, [1, 1]), [[10., 10.]])

    def test_late_forecast_and_wrong_run_are_rejected(self):
        with self.assertRaises(AssertionError):
            time_axis("2024-04-03", "2024-04-02T00:00Z", np.arange(39,64,3), 2, 24)
        with self.assertRaises(AssertionError):
            time_axis("2024-04-03", "2024-04-01T00:00Z", np.arange(39,64,3), 2, 48)

    def test_tigge_temperature_interpolates_with_no_extrapolation(self):
        times, target = time_axis("2025-01-04", "2025-01-01T00:00Z", np.arange(60,91,6), 3, 48)
        values = np.arange(0., 36., 6.)[:, None]
        out = expand(values, times, target)
        np.testing.assert_allclose(out[:, 0], np.arange(4., 28.))


if __name__ == "__main__": unittest.main()
