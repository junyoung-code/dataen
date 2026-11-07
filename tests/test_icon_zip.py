"""Check native-cell stitching and refusal of absent ICON source chunks."""
import io
import json
from pathlib import Path
import sys
import unittest
import warnings
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from collect_icon_zip import subset
    READY = True
except ModuleNotFoundError as error:
    if error.name != "numcodecs":
        raise
    READY = False


@unittest.skipUnless(READY, "Install requirements/weather.txt for ICON ZIP checks")
class IconZipTests(unittest.TestCase):
    def source(self, omit=False, dtype="float32"):
        data = np.arange(24, dtype=dtype).reshape(4, 6)
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, "w") as archive:
            archive.writestr("solar/.zarray", json.dumps({"shape": [4, 6], "chunks": [2, 3],
                "dtype": data.dtype.str, "order": "C", "compressor": None, "filters": None}))
            archive.writestr("solar/.zattrs", json.dumps({"units": "W m-2"}))
            for row in range(2):
                for col in range(2):
                    if omit and row == 1 and col == 1:
                        continue
                    archive.writestr(f"solar/{row}.{col}", data[row*2:row*2+2, col*3:col*3+3].tobytes())
        memory.seek(0)
        return zipfile.ZipFile(memory), data

    def test_noncontiguous_cells_across_chunks_keep_requested_order(self):
        archive, expected = self.source()
        with archive:
            actual, attrs = subset(archive, "solar", [np.array([3, 0]), np.array([5, 1, 3])], {})
        np.testing.assert_array_equal(actual, expected[np.ix_([3, 0], [5, 1, 3])])
        self.assertEqual(attrs["units"], "W m-2")

    def test_absent_chunks_are_rejected_instead_of_filled(self):
        archive, _ = self.source(omit=True)
        with archive, self.assertRaises(KeyError):
            subset(archive, "solar", [np.array([3]), np.array([5])], {})

    def test_integer_time_coordinates_do_not_require_nan_fill(self):
        archive, expected = self.source(dtype="int64")
        with archive, warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            actual, _ = subset(archive, "solar", [np.array([3]), np.array([5])], {})
        np.testing.assert_array_equal(actual, expected[np.ix_([3], [5])])


if __name__ == "__main__":
    unittest.main()
