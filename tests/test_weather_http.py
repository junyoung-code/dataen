"""Large weather archives must never be downloaded when a Range request is ignored."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import weather_http


class BoundedHTTPTests(unittest.TestCase):
    def setUp(self):
        weather_http._LOCAL.connections = {}

    def run_response(self, status, length, expected_range, body=b"abcd"):
        response = MagicMock(status=status)
        headers = {"Content-Length": str(length), "Content-Range": expected_range}
        response.getheader.side_effect = headers.get
        response.getheaders.return_value = list(headers.items())
        response.read.return_value = body
        connection = MagicMock()
        connection.getresponse.return_value = response
        return response, connection

    def test_range_ignored_is_rejected_before_reading_body(self):
        response, connection = self.run_response(200, 40 * 1024**3, None)
        with patch.object(weather_http.http.client, "HTTPSConnection", return_value=connection):
            with self.assertRaises(ValueError):
                weather_http.get_bytes("https://example.test/large.zip", status=206, max_bytes=4)
        response.read.assert_not_called()
        connection.close.assert_called_once()

    def test_wrong_range_is_rejected_before_reading_body(self):
        response, connection = self.run_response(206, 4, "bytes 4-7/100")
        with patch.object(weather_http.http.client, "HTTPSConnection", return_value=connection):
            with self.assertRaises(ValueError):
                weather_http.get_bytes("https://example.test/large.zip", status=206,
                    expected_headers={"Content-Range": "bytes 0-3/100"}, max_bytes=4)
        response.read.assert_not_called()

    def test_valid_bounded_response_and_truncated_response(self):
        response, connection = self.run_response(206, 4, "bytes 0-3/100")
        with patch.object(weather_http.http.client, "HTTPSConnection", return_value=connection):
            data, _ = weather_http.get_bytes("https://example.test/large.zip", status=206,
                expected_headers={"Content-Range": "bytes 0-3/100"}, max_bytes=4)
            self.assertEqual(data, b"abcd")
            response.read.return_value = b"abc"
            with self.assertRaises(OSError):
                weather_http.get_bytes("https://example.test/large.zip", status=206, max_bytes=4)


if __name__ == "__main__":
    unittest.main()
