"""Collect only fixed-area solar/cloud chunks from public OCF ICON ZIP mirrors."""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sys
import struct
import time
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import numcodecs

from probe_icon_zip import HTTPRangeReader
from weather_blosc_range import selected_values

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                 for name in ("collect_icon_zip.py", "weather_blosc_range.py", "probe_icon_zip.py", "weather_http.py")}
NAMES = {"aswdir_s": "mean_surface_direct_short_wave_radiation_flux",
         "aswdifd_s": "mean_surface_diffuse_short_wave_radiation_flux", "clct": "total_cloud_cover"}
LEADS = np.array([36, 42, 48, 54, 60, 66])
BLOCK_READERS = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def decode(blob, meta):
    compressor = meta["compressor"]
    if compressor is None:
        raw = blob
    elif compressor["id"] == "blosc2":
        import blosc2
        raw = blosc2.decompress2(blob, nthreads=1)
    else:
        raw = numcodecs.get_codec(compressor).decode(blob)
    if meta.get("filters"):
        raise ValueError("Unsupported additional Zarr filters")
    shape = meta["chunks"]
    return np.frombuffer(raw, dtype=meta["dtype"]).reshape(shape, order=meta["order"])


def subset(archive, name, selections, used, ranged=None):
    meta = json.loads(archive.read(name + "/.zarray"))
    attrs = json.loads(archive.read(name + "/.zattrs"))
    if len(meta["shape"]) != len(selections):
        raise ValueError("Source dimensions changed")
    result = np.empty(tuple(len(s) for s in selections), dtype=meta["dtype"])
    covered = np.zeros(result.shape, dtype=bool)
    for indices in selections:
        if not len(indices):
            raise ValueError("Empty selection")
    groups = [sorted(set((s // c).tolist())) for s, c in zip(selections, meta["chunks"])]
    import itertools
    for key in itertools.product(*groups):
        entry = name + "/" + meta.get("dimension_separator", ".").join(map(str, key))
        info = archive.getinfo(entry)  # A missing chunk is an error, never a zero.
        output = [np.flatnonzero(s // c == k) for s, c, k in zip(selections, meta["chunks"], key)]
        local = [s[o] - c * k for s, o, c, k in zip(selections, output, meta["chunks"], key)]
        partial = ranged is not None and (meta.get("compressor") or {}).get("id") == "blosc2" and info.compress_type == zipfile.ZIP_STORED
        if partial:
            prefix = ranged.read_at(info.header_offset, min(32768, ranged.size - info.header_offset))
            header = prefix[:30]
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            offset = info.header_offset + 30 + name_length + extra_length
            def read(start, count):
                relative = offset - info.header_offset + start
                if relative + count <= len(prefix):
                    return prefix[relative:relative + count]
                return ranged.read_at(offset + start, count)
            def read_many(ranges):
                return list(BLOCK_READERS.map(lambda pair: read(*pair), ranges))
            flat = np.ravel_multi_index(np.meshgrid(*local, indexing="ij"), meta["chunks"]).ravel()
            selected = selected_values(read, info.file_size, flat, np.dtype(meta["dtype"]), read_many)
            result[np.ix_(*output)] = selected.reshape(tuple(len(x) for x in local))
        else:
            block = decode(archive.read(entry), meta)
            result[np.ix_(*output)] = block[np.ix_(*local)]
        covered[np.ix_(*output)] = True
        used[entry] = {"zip_crc32_advertised": info.CRC, "bytes": info.file_size,
                       "read_mode": "blosc_blocks" if partial else "full_chunk_crc_checked"}
    if not covered.all() or not np.isfinite(result).all():
        raise ValueError("Nonfinite decoded values")
    return result, attrs


def collect(record, out, coords, read_mode="full", endpoint="s3"):
    key = record["key"]
    init = dt.datetime.strptime(Path(key).name[:11], "%Y%m%d_%H").replace(tzinfo=dt.timezone.utc)
    if init.hour != 0:
        raise ValueError("Only fixed 00 UTC runs are supported")
    day = (init + dt.timedelta(days=2)).date().isoformat()
    deadline = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), ZoneInfo("Asia/Seoul")) - dt.timedelta(hours=7)
    assumed = init + dt.timedelta(hours=24)
    if assumed > deadline:
        raise ValueError("Forecast availability assumption exceeds deadline")
    folder = out / ("icon_raw_blocks" if read_mode == "blocks" else "icon_raw")
    path = folder / "days" / (day + ".npz")
    audit = path.with_suffix(".json")
    if path.exists() and audit.exists():
        previous = json.loads(audit.read_text())
        if previous["archive_key"] != key or previous["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Cache mismatch")
        return previous | {"reused": True}
    started = time.monotonic()
    base_url = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
                if endpoint == "s3" else "https://data.source.coop/")
    url = base_url + key
    reader = HTTPRangeReader(url, record["size"])
    ranged = HTTPRangeReader(url, record["size"]) if read_mode == "blocks" else None
    used, values, attributes = {}, {}, {}
    with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=128 * 1024)) as archive:
        time_meta = json.loads(archive.read("time/.zarray"))
        time_attrs = json.loads(archive.read("time/.zattrs"))
        if time_attrs["units"] != "nanoseconds since 1970-01-01":
            raise ValueError("Unexpected init time unit")
        actual_time = np.datetime64(int(decode(archive.read("time/0"), time_meta)), "ns")
        if actual_time != np.datetime64(init.replace(tzinfo=None)):
            raise ValueError("Init time does not match archive name")
        step_meta = json.loads(archive.read("step/.zarray"))
        steps, step_attrs = subset(archive, "step", [np.arange(step_meta["shape"][0])], used)
        units = step_attrs["units"]
        steps = steps / (3600 * 1e9) if units == "nanoseconds" else steps if units == "hours" else None
        if steps is None:
            raise ValueError("Unexpected lead time unit")
        chosen = np.array([np.flatnonzero(steps == h).item() for h in LEADS])
        native = coords["native_cell_index"]
        latitude, _ = subset(archive, "latitude", [native], used)
        longitude, _ = subset(archive, "longitude", [native], used)
        longitude = (longitude + 180) % 360 - 180
        if not (np.allclose(latitude, coords["latitude"], rtol=0, atol=1e-5)
                and np.allclose(longitude, coords["longitude"], rtol=0, atol=1e-5)):
            raise ValueError("Native grid changed; do not reuse old cell indices")
        for raw_name, name in NAMES.items():
            values[name], attributes[name] = subset(archive, raw_name, [chosen, native], used, ranged)
    if not any((a > 0).any() for name, a in values.items() if "short_wave" in name):
        raise ValueError("No daytime solar signal")
    if ranged and reader.etag and ranged.etag and reader.etag != ranged.etag:
        raise ValueError("Archive changed between metadata and block reads")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)
    result = {"day": day, "init_time": init.isoformat(), "archive_key": key, "url": url,
              "deadline": deadline.isoformat(), "available_at_assumed": assumed.isoformat(),
              "archive_bytes": record["size"], "downloaded_bytes": reader.transferred + (ranged.transferred if ranged else 0),
              "read_mode": read_mode,
              "implementation_sha256": SOURCE_HASHES,
              "source_etag": reader.etag,
              "publication_verified": False, "mirror_last_modified": record["last_modified_mirror"],
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
              "variables": {n: {"shape": list(a.shape), "min": float(a.min()), "max": float(a.max()),
                                 "attrs": attributes[n]} for n, a in values.items()},
              "source_entries": used, "seconds": time.monotonic() - started, "reused": False}
    dump(audit, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-dates", help="Comma-separated YYYYMMDD; otherwise inventory excluding existing Icechunk days")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-mode", choices=["full", "blocks"], default="full")
    parser.add_argument("--exclude-init-dates", default="", help="Runs handled by a separate in-progress sample job")
    parser.add_argument("--endpoint", choices=["s3", "mirror"], default="s3")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    sys.path.insert(0, str(out / ".runtime/icon-codecs"))
    records = json.loads((out / "metadata/icon_raw_inventory.json").read_text())
    existing = {p.stem for p in (out / "icon/days").glob("*.npz")}
    if args.init_dates:
        wanted = set(args.init_dates.split(","))
        records = [r for r in records if Path(r["key"]).name[:8] in wanted]
        if len(records) != len(wanted):
            raise ValueError("Requested run absent from inventory")
    else:
        records = [r for r in records if (dt.datetime.strptime(Path(r["key"]).name[:8], "%Y%m%d")
                   + dt.timedelta(days=2)).date().isoformat() not in existing]
    excluded = set(args.exclude_init_dates.split(",")) - {""}
    records = [r for r in records if Path(r["key"]).name[:8] not in excluded]
    records.sort(key=lambda r: r["key"])
    coords = dict(np.load(out / "icon/coordinates.npz", allow_pickle=False))
    folder = out / ("icon_raw_blocks" if args.read_mode == "blocks" else "icon_raw")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "coordinates.npz", **coords)
    dump(folder / "dataset.json", {"region": [30, 42, 120, 134], "leads": LEADS.tolist(),
         "collector_implementation_sha256": SOURCE_HASHES,
         "source": "OCF ICON Global via Source Cooperative public raw ZIP mirror",
         "source_page": "https://source.coop/bkr/icon", "variables": NAMES,
         "transformation": "Native cell and lead subset, no interpolation or deaveraging",
         "publication_status": "Not certified from archive availability alone",
         "explicitly_excluded_initializations": sorted(excluded),
         "existing_icechunk_days_excluded": sorted(existing) if not args.init_dates else []})
    done, errors = [], []
    started = time.monotonic()
    def task(record):
        for attempt in range(3):
            try:
                return collect(record, out, coords, args.read_mode, args.endpoint)
            except (ValueError, KeyError):
                raise
            except Exception as error:
                print(json.dumps({"retry": attempt + 1, "key": record["key"],
                                  "error": f"{type(error).__name__}: {error}"}), flush=True)
                if attempt == 2: raise
                time.sleep(2 * (attempt + 1))
    def report(status):
        data = {"status": status, "expected_days": len(records), "completed_days": len(done),
                "sample_only": bool(args.init_dates), "errors": errors,
                "bytes": sum(r["bytes"] for r in done), "seconds": time.monotonic() - started,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump(folder / ("sample_summary.json" if args.init_dates else "collection_summary.json"), data)
        print(json.dumps(data), flush=True)
    report("running")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(task, r): r for r in records}
        for job in concurrent.futures.as_completed(jobs):
            try: done.append(job.result())
            except Exception as e: errors.append({"key": jobs[job]["key"], "error": f"{type(e).__name__}: {e}"})
            if len(records) < 10 or (len(done) + len(errors)) % 10 == 0: report("running")
    report("complete" if not errors else "partial")
    if errors: raise SystemExit(2)


if __name__ == "__main__":
    main()
