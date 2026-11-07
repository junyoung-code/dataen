"""Resumable, read-only public forecast acquisition; never reads competition data."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import shutil
import time
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timing(day, lag):
    day = pd.Timestamp(day)
    init = (day - pd.Timedelta(days=2)).tz_localize("UTC")
    deadline = day.tz_localize("Asia/Seoul") - pd.Timedelta(hours=7)
    assumed = init + pd.Timedelta(hours=lag)
    if assumed > deadline:
        raise ValueError("Forecast availability assumption exceeds previous-day 17 KST")
    return init, deadline, assumed


def collect_day(ds, day, folder, cfg, signature, coords, selection, lag):
    dest = folder / "days" / f"{day}.npz"
    audit = dest.with_suffix(".json")
    if audit.exists() and dest.exists():
        saved = json.loads(audit.read_text())
        if saved["signature"] != signature or saved["sha256"] != sha(dest):
            raise ValueError("Existing cache identity/checksum mismatch")
        return saved | {"reused": True}
    if shutil.disk_usage(folder).free < 8 * 1024**3:
        raise OSError("Less than 8 GiB free disk space")
    started = time.monotonic()
    init, deadline, assumed = timing(day, lag)
    time_dim = "step" if "step" in ds.dims else "lead_time"
    sub = ds[cfg["variables"]].sel(init_time=init.tz_localize(None).to_numpy(),
         **{time_dim: pd.to_timedelta(cfg["lead_hours"], unit="h")})
    sub = sub.isel(**selection) if "values" in ds.dims else sub.sel(**selection)
    sub = sub.load()
    arrays, stats = {}, {}
    for name in cfg["variables"]:
        arr = np.asarray(sub[name].values, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.all():
            raise ValueError(f"Missing/nonfinite source field {name}: {int(finite.sum())}/{arr.size}")
        arrays[name] = arr
        stats[name] = {"shape": list(arr.shape), "dimensions": list(sub[name].dims),
                       "min": float(arr.min()), "max": float(arr.max())}
    solar = [a for name, a in arrays.items()
             if "short_wave_radiation_flux" in name and "upward" not in name]
    # The fixed Korea/upstream region and requested >24 h horizon include daylight.
    # Some incomplete ICON mirror chunks are absent but decode as finite zeros.
    if solar and all(not np.any(a > 0) for a in solar):
        raise ValueError("No daytime solar signal: possible missing zero-filled source chunks")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(dest)
    result = {"day": day, "init_time": init.isoformat(), "deadline": deadline.isoformat(),
              "available_at_assumed": assumed.isoformat(), "publication_verified": False,
              "signature": signature, "sha256": sha(dest), "bytes": dest.stat().st_size,
              "selected_uncompressed_bytes": sum(a.nbytes for a in arrays.values()),
              "variables": stats, "seconds": time.monotonic() - started, "reused": False}
    dump(audit, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["gfs", "gefs", "icon"], required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/weather_collection_20261106.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dates", help="Comma-separated sample target days, otherwise collect full configured period")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    settings = json.loads(args.config.read_text())
    cfg = settings["sources"][args.source]
    meta = json.loads((out / "metadata" / cfg["metadata_file"]).read_text())
    signature = hashlib.sha256(json.dumps({"config": settings, "source": args.source,
            "snapshot": meta["snapshot_id"]}, sort_keys=True).encode()).hexdigest()
    folder = out / args.source
    folder.mkdir(parents=True, exist_ok=True)
    previous_metadata = folder / "dataset.json"
    if previous_metadata.exists() and json.loads(previous_metadata.read_text())["signature"] != signature:
        raise ValueError("Existing dataset identity differs; use a new output directory")
    repo = icechunk.Repository.open(icechunk.http_storage(meta["url"]))
    session = repo.readonly_session(snapshot_id=meta["snapshot_id"])
    ds = xr.open_zarr(session.store, chunks=None, consolidated=False)
    r = settings["region"]
    if "values" in ds.dims:
        lat = ds.latitude.values
        lon = (ds.longitude.values + 180) % 360 - 180
        indices = np.flatnonzero((lat >= r["latitude_min"]) & (lat <= r["latitude_max"]) &
                                (lon >= r["longitude_min"]) & (lon <= r["longitude_max"]))
        if not len(indices):
            raise ValueError("No native grid cells inside the fixed region")
        selection = {"values": indices}
        coords = {"latitude": lat[indices], "longitude": lon[indices], "native_cell_index": indices}
    else:
        selection = {"latitude": slice(r["latitude_max"], r["latitude_min"]),
                     "longitude": slice(r["longitude_min"], r["longitude_max"])}
        local = ds.sel(**selection)
        coords = {"latitude": local.latitude.values, "longitude": local.longitude.values}
        if "ensemble_member" in ds.dims:
            coords["ensemble_member"] = ds.ensemble_member.values
    coords["lead_hours"] = np.array(cfg["lead_hours"])
    np.savez_compressed(folder / "coordinates.npz", **coords)
    dump(folder / "dataset.json", {"signature": signature, "snapshot_id": meta["snapshot_id"],
          "url": meta["url"], "source": args.source, "config": settings,
          "variable_metadata": {v: meta["variables"][v] for v in cfg["variables"]},
          "transformation": "Spatial/lead subset only. No interpolation, deaveraging, normalization or imputation.",
          "publication_status": settings["availability_status"]})
    days = args.dates.split(",") if args.dates else pd.date_range(cfg["start"], cfg["end"]).strftime("%Y-%m-%d").tolist()
    completed, errors = [], []
    started = time.monotonic()
    def task(day):
        for attempt in range(3):
            try:
                return collect_day(ds, day, folder, cfg, signature, coords, selection,
                                   settings["availability_lag_hours_assumed"])
            except ValueError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    def report(status):
        result = {"status": status, "source": args.source, "signature": signature,
                  "sample_only": bool(args.dates), "expected_days": len(days), "completed_days": len(completed),
                  "reused_days": sum(x["reused"] for x in completed), "errors": errors,
                  "bytes": sum(x["bytes"] for x in completed), "seconds": time.monotonic() - started,
                  "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump(folder / ("sample_summary.json" if args.dates else "collection_summary.json"), result)
        print(json.dumps({k: v for k, v in result.items() if k != "errors"}), flush=True)
    report("running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, day): day for day in days}
        for future in concurrent.futures.as_completed(futures):
            day = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                item = {"day": day, "type": type(error).__name__, "error": str(error)[:1200]}
                errors.append(item)
                print(json.dumps(item), flush=True)
            count = len(completed) + len(errors)
            if count % 25 == 0 or count == len(days):
                report("running")
    report("complete" if not errors else "partial")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
