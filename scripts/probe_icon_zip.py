"""Inspect public ICON ZIP indexes with byte ranges, without downloading whole archives."""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import threading
from weather_http import get_bytes


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.transferred = 0
        self.etag = None
        self._lock = threading.Lock()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        data = self.read_at(self.position, end - self.position)
        self.position = end
        return data

    def read_at(self, start, count):
        end = start + count
        if start < 0 or count < 1 or end > self.size:
            raise ValueError("Invalid remote byte range")
        headers = {"Range": f"bytes={start}-{end - 1}"}
        if self.etag:
            headers["If-Match"] = self.etag
        expected = f"bytes {start}-{end - 1}/{self.size}"
        data, response_headers = get_bytes(self.url, headers=headers, status=206,
            expected_headers={"Content-Range": expected}, max_bytes=count)
        etag = response_headers.get("etag")
        if len(data) != count:
            raise ValueError("Truncated range response")
        with self._lock:
            if self.etag and etag and self.etag != etag:
                raise ValueError("Remote archive changed while reading")
            self.etag = etag or self.etag
            self.transferred += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reader = HTTPRangeReader(a.url, a.size)
    with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=128 * 1024)) as archive:
        names = archive.namelist()
        print(json.dumps({"entry_count": len(names), "arrays": [n for n in names if n.endswith(".zarray")]}), flush=True)
        wanted = {"aswdir_s", "aswdifd_s", "clct", "tcc", "clat", "clon", "latitude", "longitude", "step", "time"}
        selected = [x for x in names if (x.endswith(".zarray") or x.endswith(".zattrs"))
                    and ("/" not in x or x.split("/")[-2] in wanted)]
        metadata = {name: json.loads(archive.read(name)) for name in selected}
        result = {"url": a.url, "size": a.size, "entry_count": len(names),
                  "metadata": metadata, "first_names": names[:30], "transferred_bytes": reader.transferred}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "metadata"}))
        print(json.dumps(metadata, indent=2)[:16000])


if __name__ == "__main__":
    main()
