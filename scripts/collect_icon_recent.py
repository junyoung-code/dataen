"""Collect archived 2026 ICON GRIB forecasts for a fixed, plant-independent area."""
import argparse
import bz2
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo
from weather_http import get_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
NAMES = {"aswdir_s": "mean_surface_direct_short_wave_radiation_flux",
         "aswdifd_s": "mean_surface_diffuse_short_wave_radiation_flux", "clct": "total_cloud_cover"}
LEADS = [36, 42, 48, 54, 60, 66]
DECODE_LOCK = threading.Lock()
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(key, path):
    audit = path.with_suffix(path.suffix + ".json")
    if path.exists() and audit.exists():
        saved = json.loads(audit.read_text())
        if saved["key"] != key or saved["sha256"] != sha(path):
            raise ValueError("Existing GRIB cache identity/checksum mismatch")
        return saved
    if shutil.disk_usage(path.parent).free < 8 * 1024**3:
        raise OSError("Less than 8 GiB free disk space")
    for attempt in range(3):
        try:
            content, response_headers = get_bytes(BASE + key)
            info = {"key": key, "url": BASE + key, "bytes": len(content),
                    "etag": response_headers.get("etag"),
                    "mirror_last_modified": response_headers.get("last-modified"),
                    "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(content).hexdigest()}
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(content); tmp.replace(path)
            dump(audit, info)
            return info
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == 2:
                raise
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(attempt + 1)


def decode(path, init, lead, raw_name, grid, indices):
    import eccodes
    import numpy as np
    with DECODE_LOCK:
        handle = eccodes.codes_new_from_message(bz2.decompress(path.read_bytes()))
        try:
            keys = ["dataDate", "dataTime", "startStep", "endStep", "stepType", "shortName",
                    "units", "numberOfDataPoints", "numberOfMissing", "uuidOfHGrid"]
            meta = {k: eccodes.codes_get(handle, k) for k in keys}
            if (str(meta["dataDate"]) != init.strftime("%Y%m%d") or meta["dataTime"] != 0
                    or meta["endStep"] != lead or meta["shortName"].lower() != raw_name):
                raise ValueError("GRIB initialization, lead or variable mismatch")
            if (meta["uuidOfHGrid"] != grid["latitude"]["uuidOfHGrid"]
                    or meta["numberOfDataPoints"] != grid["native_cells"] or meta["numberOfMissing"]):
                raise ValueError("GRIB grid changed or contains missing points")
            if raw_name == "clct":
                if meta["stepType"] != "instant" or meta["units"] != "%":
                    raise ValueError("Unexpected cloud semantics")
            elif meta["stepType"] != "avg" or meta["startStep"] != 0 or meta["units"] != "W m-2":
                raise ValueError("Unexpected accumulated-average radiation semantics")
            values = eccodes.codes_get_array(handle, "values")[indices].copy()
        finally:
            eccodes.codes_release(handle)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite GRIB values")
    return values, meta


def prepare_grid(out):
    """Recompute the plant-independent mask from the archived coordinate GRIBs."""
    import eccodes
    import numpy as np
    folder = out / "icon_recent"
    records = json.loads((out / "metadata/icon_recent_files.json").read_text())
    coordinates, metadata, provenance = [], [], []
    for record in records[2:4]:
        item = next(r for r in record["files"] if r["key"].endswith(".grib2.bz2"))
        path = folder / "probe" / Path(item["key"]).name
        path.parent.mkdir(parents=True, exist_ok=True)
        # The earlier acquisition probe saved these exact public objects.
        if path.exists() and not path.with_suffix(path.suffix + ".json").exists():
            if path.stat().st_size != item["size"]:
                raise ValueError("Coordinate probe size differs from source inventory")
            bz2.decompress(path.read_bytes())
            dump(path.with_suffix(path.suffix + ".json"), {"key": item["key"], "url": BASE + item["key"],
                 "bytes": item["size"], "mirror_last_modified": item["last_modified"], "sha256": sha(path),
                 "provenance": "Reused earlier coordinate download in this run; ETag was not recorded."})
        provenance.append(fetch(item["key"], path))
        handle = eccodes.codes_new_from_message(bz2.decompress(path.read_bytes()))
        try:
            meta = {k: eccodes.codes_get(handle, k) for k in ["shortName", "units", "uuidOfHGrid", "numberOfDataPoints"]}
            values = eccodes.codes_get_array(handle, "values")
        finally:
            eccodes.codes_release(handle)
        expected_unit = "Degree N" if not coordinates else "Degree E"
        if meta["units"] != expected_unit:
            raise ValueError("Unexpected native coordinate units")
        coordinates.append(values); metadata.append(meta)
    lat, lon = coordinates
    lon = (lon + 180) % 360 - 180
    if len(lat) != len(lon) or metadata[0]["uuidOfHGrid"] != metadata[1]["uuidOfHGrid"]:
        raise ValueError("Latitude and longitude grids differ")
    indices = np.flatnonzero((lat >= 30) & (lat <= 42) & (lon >= 120) & (lon <= 134))
    selected = {"latitude": lat[indices], "longitude": lon[indices], "native_cell_index": indices,
                "lead_hours": np.array(LEADS)}
    target = folder / "coordinates.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as previous:
            if not all(np.array_equal(previous[k], v) for k, v in selected.items()):
                raise ValueError("Existing selected coordinates changed")
    else:
        np.savez_compressed(target, **selected)
    dump(folder / "grid.json", {"latitude": metadata[0], "longitude": metadata[1], "native_cells": len(lat),
         "selected_cells": len(indices), "sources": provenance, "coordinates_sha256": sha(target)})

    # Reuse the already decoded first solar field, retaining its acquisition provenance.
    probe = folder / "probe/icon_global_icosahedral_single-level_2026031200_036_ASWDIR_S.grib2.bz2"
    if probe.exists():
        record = next(r for r in records[1]["files"] if Path(r["key"]).name == probe.name)
        dest = folder / "grib/20260312" / probe.name
        if not dest.exists():
            if probe.stat().st_size != record["size"]:
                raise ValueError("Solar probe size differs from inventory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.hardlink_to(probe)
            dump(dest.with_suffix(dest.suffix + ".json"), {"key": record["key"], "url": BASE + record["key"],
                 "bytes": record["size"], "mirror_last_modified": record["last_modified"], "sha256": sha(dest),
                 "provenance": "Reused earlier first-field download; ETag was not recorded."})


def collect(stamp, folder, grid, indices):
    import numpy as np
    started = time.monotonic()
    init = dt.datetime.strptime(stamp[:8], "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    day = (init + dt.timedelta(days=2)).date().isoformat()
    deadline = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), ZoneInfo("Asia/Seoul")) - dt.timedelta(hours=7)
    path = folder / "days" / (day + ".npz")
    if path.exists() and path.with_suffix(".json").exists():
        saved = json.loads(path.with_suffix(".json").read_text())
        if saved["sha256"] != sha(path) or saved["capture_folder"] != stamp:
            raise ValueError("Existing subset identity/checksum mismatch")
        return saved | {"reused": True}
    if init + dt.timedelta(hours=24) > deadline:
        raise ValueError("Assumed availability exceeds deadline")
    raw = folder / "grib" / init.strftime("%Y%m%d")
    raw.mkdir(parents=True, exist_ok=True)
    arrays, sources = {}, []
    for raw_name, name in NAMES.items():
        pieces = []
        for lead in LEADS:
            filename = f"icon_global_icosahedral_single-level_{init:%Y%m%d}00_{lead:03d}_{raw_name.upper()}.grib2.bz2"
            key = f"bkr/icon/icon-global-raw/{stamp}/opendata.dwd.de/weather/nwp/icon/grib/00/{raw_name}/{filename}"
            info = fetch(key, raw / filename)
            values, meta = decode(raw / filename, init, lead, raw_name, grid, indices)
            pieces.append(values); sources.append(info | {"grib": meta})
        arrays[name] = np.stack(pieces)
    if not any((a > 0).any() for name, a in arrays.items() if "short_wave" in name):
        raise ValueError("No daytime solar signal")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays); tmp.replace(path)
    result = {"day": day, "init_time": init.isoformat(), "capture_folder": stamp,
              "deadline": deadline.isoformat(), "available_at_assumed": (init + dt.timedelta(hours=24)).isoformat(),
              "publication_verified": False, "publication_note": "Mirror upload times are not original DWD publication evidence.",
              "sha256": sha(path), "bytes": path.stat().st_size,
              "coordinates_sha256": sha(folder / "coordinates.npz"), "implementation_sha256": CODE_SHA,
              "http_implementation_sha256": sha(ROOT / "scripts/weather_http.py"),
              "variables": {n: {"shape": list(a.shape), "dtype": str(a.dtype), "min": float(a.min()), "max": float(a.max())}
                            for n, a in arrays.items()},
              "sources": sources, "seconds": time.monotonic() - started, "reused": False}
    dump(path.with_suffix(".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--init-dates", help="Optional comma-separated YYYYMMDD sample dates")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    sys.path.insert(0, str(out / ".runtime/eccodes"))
    import numpy as np
    folder = out / "icon_recent"
    prepare_grid(out)
    grid = json.loads((folder / "grid.json").read_text())
    with np.load(folder / "coordinates.npz", allow_pickle=False) as c:
        indices = c["native_cell_index"]
        if not ((c["latitude"] >= 30).all() and (c["latitude"] <= 42).all()
                and (c["longitude"] >= 120).all() and (c["longitude"] <= 134).all()
                and np.array_equal(c["lead_hours"], LEADS)):
            raise ValueError("Coordinate selection does not match fixed region/leads")
    listing = json.loads((out / "metadata/icon_recent_files.json").read_text())[0]
    if listing["truncated"] != "false":
        raise ValueError("Source folder inventory is incomplete")
    stamps = sorted({s.split("/")[-2] for s in listing["folders"]
                     if "20260312" <= s.split("/")[-2][:8] <= "20260529" and s.split("/")[-2].endswith("08")})
    if args.init_dates:
        selected = set(args.init_dates.split(",")); stamps = [s for s in stamps if s[:8] in selected]
        if len(stamps) != len(selected):
            raise ValueError("Requested init date is not in the verified 00 UTC inventory")
    dump(folder / "inventory.json", {"capture_folders": stamps, "sample_only": bool(args.init_dates),
        "selection": "00 UTC forecasts captured in 08 UTC folders; fixed init range 20260312..20260529",
        "transformation": "Spatial/lead subset only; ecCodes float64 values retained; no deaveraging or interpolation."})
    completed, errors = [], []
    start = time.monotonic()
    def report(status):
        value = {"status": status, "expected_days": len(stamps), "completed_days": len(completed),
                 "errors": errors, "sample_only": bool(args.init_dates), "bytes": sum(x["bytes"] for x in completed),
                 "seconds": time.monotonic() - start, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump(folder / ("sample_summary.json" if args.init_dates else "collection_summary.json"), value)
        print(json.dumps(value), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, s, folder, grid, indices): s for s in stamps}
        for future in concurrent.futures.as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append({"capture_folder": futures[future], "error": f"{type(error).__name__}: {error}"})
            if (len(completed) + len(errors)) % 5 == 0 or len(completed) + len(errors) == len(stamps):
                report("running")
    report("complete" if not errors else "partial")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
