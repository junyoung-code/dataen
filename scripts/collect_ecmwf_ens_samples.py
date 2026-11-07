"""Collect explicitly selected ECMWF ENS sample days from a frozen public snapshot.

Uses no competition data or credentials. Dates are target days in Korea; each
sample retains the D-2 00 UTC run and every ensemble member on a fixed region.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collect_spatial_weather import collect_day, dump, sha

VARIABLES = ["downward_short_wave_radiation_flux_surface", "temperature_2m",
             "dew_point_temperature_2m"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dates", required=True, help="Explicit comma-separated target dates; no automatic full-period mode")
    parser.add_argument("--variables", default=",".join(VARIABLES))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    dates = args.dates.split(",")
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate dates would race on cache files")
    for day in dates:
        if pd.Timestamp(day).strftime("%Y-%m-%d") != day:
            raise ValueError("Dates must be YYYY-MM-DD")
    variables = args.variables.split(",")
    if not variables or not set(variables).issubset(VARIABLES):
        raise ValueError("Only documented sample variables are supported")
    meta = json.loads((out / "metadata/ecmwf_ens.metadata.json").read_text())
    cfg = {"variables": variables, "lead_hours": list(range(39, 64, 3)),
           "region": {"north": 42, "south": 30, "west": 120, "east": 134},
           "availability_lag_hours_assumed": 24}
    signature = hashlib.sha256(json.dumps({"config": cfg, "snapshot": meta["snapshot_id"]}, sort_keys=True).encode()).hexdigest()
    folder = out / "ecmwf_ens" / ("solar_probe" if len(variables) == 1 else "seasonal_samples")
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / "dataset.json"
    if old.exists() and json.loads(old.read_text())["signature"] != signature:
        raise ValueError("Existing sample configuration differs; use a new output directory")
    repo = icechunk.Repository.open(icechunk.http_storage(meta["url"]))
    session = repo.readonly_session(snapshot_id=meta["snapshot_id"])
    ds = xr.open_zarr(session.store, chunks=None, consolidated=False)
    selection = {"latitude": slice(42, 30), "longitude": slice(120, 134)}
    local = ds.sel(**selection)
    if local.sizes["latitude"] != 49 or local.sizes["longitude"] != 57 or local.sizes["ensemble_member"] != 51:
        raise ValueError("Unexpected native grid or ensemble size")
    coords = {k: local[k].values for k in ("latitude", "longitude", "ensemble_member")}
    coords["lead_hours"] = np.array(cfg["lead_hours"])
    np.savez_compressed(folder / "coordinates.npz", **coords)
    dump(old, {"signature": signature, "snapshot_id": meta["snapshot_id"], "url": meta["url"],
               "config": cfg, "variable_metadata": {v: meta["variables"][v] for v in variables},
               "transformation": "Native spatial/lead subset only; all 51 members. Solar flux already averaged over previous forecast interval, not since initialization.",
               "publication_status": "24-hour lag is a provisional assumption, not verified historical availability.",
               "licence": "CC BY 4.0 and ECMWF Terms of Use; see source_review.json"})
    dump(folder / "code_provenance.json", {"files": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), ROOT / "src/collect_spatial_weather.py")}})
    records, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(collect_day, ds, day, folder, cfg, signature, coords, selection, 24): day for day in dates}
        for job in concurrent.futures.as_completed(jobs):
            day = jobs[job]
            try:
                record = job.result()
                records.append(record)
                print(json.dumps({"day": day, "status": "reused" if record["reused"] else "saved", "bytes": record["bytes"]}), flush=True)
            except Exception as exc:
                errors.append({"day": day, "type": type(exc).__name__, "error": str(exc)[:1000]})
                print(json.dumps(errors[-1]), flush=True)
            dump(folder / "collection_summary.json", {"requested_dates": dates, "completed": records, "errors": errors,
                   "status": "running", "scope": "Explicit requested dates", "workers": args.workers})
    dump(folder / "collection_summary.json", {"requested_dates": dates, "completed": records, "errors": errors,
           "status": "failed" if errors else "complete", "scope": "Explicit requested dates", "workers": args.workers})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
