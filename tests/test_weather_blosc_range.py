"""Partial compressed-block reads must match full decompression exactly."""
from pathlib import Path
import sys
import unittest
import io
import json
import types
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import blosc2
    READY = True
except ModuleNotFoundError:
    READY = False


@unittest.skipUnless(READY, "Install requirements/weather-icon.txt for compressed range checks")
class CompressedRangeTests(unittest.TestCase):
    def test_selected_blocks_match_full_decode_and_transfer_less(self):
        from weather_blosc_range import selected_values
        data = np.random.default_rng(37).normal(size=400000).astype("float32")
        compressed = blosc2.compress2(data, codec=blosc2.Codec.ZSTD, typesize=4, blocksize=65536)
        requests = []
        def read(start, length):
            requests.append((start, length))
            return compressed[start:start + length]
        chosen = np.array([300004, 300001, 300004, 399999, 0])
        actual = selected_values(read, len(compressed), chosen, np.dtype("float32"))
        np.testing.assert_array_equal(actual, data[chosen])
        self.assertLess(sum(n for _, n in requests), len(compressed) / 2)

    def test_special_zero_chunk_and_bad_indices(self):
        from weather_blosc_range import selected_values
        data = np.zeros(100000, dtype="float32")
        compressed = blosc2.compress2(data, typesize=4)
        read = lambda start, length: compressed[start:start + length]
        np.testing.assert_array_equal(selected_values(read, len(compressed), np.array([0, 99999]), data.dtype), [0, 0])
        with self.assertRaises(ValueError):
            selected_values(read, len(compressed), np.array([100000]), data.dtype)

    def test_filter_typesize_can_differ_from_array_dtype(self):
        from weather_blosc_range import selected_values
        data = np.random.default_rng(42).normal(size=400000).astype("float32")
        compressed = blosc2.compress2(data, typesize=8, codec=blosc2.Codec.ZSTD, blocksize=65536)
        received = []
        def read(start, length):
            received.append(length)
            return compressed[start:start+length]
        chosen = np.array([300001, 300002, 399999, 0])
        np.testing.assert_array_equal(selected_values(read, len(compressed), chosen, data.dtype), data[chosen])
        self.assertLess(sum(received), len(compressed) / 2)

    def test_zip_offsets_and_partial_subset_match_original_array(self):
        from collect_icon_zip import subset
        data = np.random.default_rng(83).normal(size=(200, 500)).astype("float32")
        compressed = blosc2.compress2(data, typesize=4, blocksize=65536, codec=blosc2.Codec.ZSTD)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("field/.zarray", json.dumps({"shape": [200, 500], "chunks": [200, 500],
                "compressor": {"id": "blosc2"}, "filters": None, "dtype": "<f4", "order": "C"}))
            archive.writestr("field/.zattrs", "{}")
            archive.writestr("field/0.0", compressed)
        blob = buffer.getvalue()
        reader = types.SimpleNamespace(size=len(blob), read_at=lambda start, length: blob[start:start+length])
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            selected, _ = subset(archive, "field", [np.array([1, 99, 199]), np.array([0, 250, 499])], {}, reader)
        np.testing.assert_array_equal(selected, data[np.ix_([1, 99, 199], [0, 250, 499])])


if __name__ == "__main__":
    unittest.main()
