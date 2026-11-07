"""Collection failures must not turn missing forecasts into usable observations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import xarray as xr
    from collect_spatial_weather import collect_day, timing
    WEATHER_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name not in {"xarray", "icechunk"}:
        raise
    WEATHER_AVAILABLE = False


@unittest.skipUnless(WEATHER_AVAILABLE, "Install requirements/weather.txt for public weather collection checks")
class CollectionTests(unittest.TestCase):
    def fixture(self, values):
        name = "downward_short_wave_radiation_flux_surface"
        ds = xr.Dataset({name: (("init_time", "lead_time", "latitude", "longitude"),
                      np.asarray(values, dtype="float32").reshape(1, 2, 1, 1))},
             coords={"init_time": [np.datetime64("2024-05-30")],
                     "lead_time": pd.to_timedelta([42, 54], unit="h"),
                     "latitude": [35.0], "longitude": [127.0]})
        return ds, {"variables": [name], "lead_hours": [42, 54]}

    def test_all_zero_daytime_radiation_is_not_a_valid_download(self):
        # Removing the sanity guard would silently accept an unwritten zero-filled archive chunk.
        ds, cfg = self.fixture([0, 0])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                collect_day(ds, "2024-06-01", Path(temp), cfg, "test", {}, {}, 24)
            self.assertFalse(list(Path(temp).rglob("*.npz")))

    def test_valid_cache_is_reused_and_corruption_rejected(self):
        ds, cfg = self.fixture([10, 300])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = collect_day(ds, "2024-06-01", root, cfg, "test", {}, {}, 24)
            second = collect_day(ds, "2024-06-01", root, cfg, "test", {}, {}, 24)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            (root / "days/2024-06-01.npz").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                collect_day(ds, "2024-06-01", root, cfg, "test", {}, {}, 24)

    def test_deadline_is_previous_day_in_korea(self):
        init, deadline, available = timing("2024-06-01", 24)
        self.assertEqual(init.isoformat(), "2024-05-30T00:00:00+00:00")
        self.assertEqual(deadline.isoformat(), "2024-05-31T17:00:00+09:00")
        self.assertLess(available, deadline)
        with self.assertRaises(ValueError):
            timing("2024-06-01", 33)


if __name__ == "__main__":
    unittest.main()
