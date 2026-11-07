"""Inspect public archive metadata without reading competition data or credentials."""
import argparse
import datetime as dt
import json
from pathlib import Path
import urllib.request

import icechunk
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    sources = {}
    for name in ("noaa-gfs-forecast", "noaa-gefs-forecast-35-day"):
        url = f"https://stac.dynamical.org/{name}/collection.json"
        with urllib.request.urlopen(url, timeout=40) as response:
            catalog = json.load(response)
        save(out / f"{name}.stac.json", catalog)
        assets = {k: v["href"] for k, v in catalog["assets"].items()}
        sources[name] = assets["icechunk-https"]
        print(name, "assets", assets, flush=True)
    sources["icon_global"] = "https://data.source.coop/bkr/icon/icon_global.icechunk"
    report = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sources": {}}
    for name, url in sources.items():
        try:
            repo = icechunk.Repository.open(icechunk.http_storage(url))
            session = repo.readonly_session("main")
            ds = xr.open_zarr(session.store, chunks=None, consolidated=False)
            item = {"url": url, "snapshot_id": session.snapshot_id,
                    "attrs": ds.attrs, "dimensions": dict(ds.sizes),
                    "variables": {v: {"dims": ds[v].dims, "attrs": ds[v].attrs,
                                      "chunks": ds[v].encoding.get("chunks")} for v in ds.data_vars},
                    "coords": {v: {"shape": ds[v].shape,
                                   "first": str(ds[v].values.flat[0]),
                                   "last": str(ds[v].values.flat[-1])}
                               for v in ds.coords if ds[v].ndim == 1}, "status": "accessible"}
            print(name, "snapshot", session.snapshot_id, "sizes", dict(ds.sizes), flush=True)
            print(name, "variables", list(ds.data_vars), flush=True)
            print(name, "coords", item["coords"], flush=True)
            save(out / f"{name}.metadata.json", item)
            report["sources"][name] = item
        except Exception as error:
            report["sources"][name] = {"status": "failed", "error": str(error)[:1500], "url": url}
            print(name, "FAILED", type(error).__name__, str(error)[:400], flush=True)
    save(out / "probe.json", report)


if __name__ == "__main__":
    main()
