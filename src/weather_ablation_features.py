"""Frozen forecast-only regional inputs for the G315 weather ablation.

Hour labels are interval END times in Asia/Seoul. Missing runs stay missing.
Radiation differences are taken per grid cell before any nonlinear statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
ICON_NAMES = {
    "mean_surface_direct_short_wave_radiation_flux": "direct",
    "mean_surface_diffuse_short_wave_radiation_flux": "diffuse",
    "total_cloud_cover": "cloud",
}
ENS_NAMES = {"downward_short_wave_radiation_flux_surface": "solar",
             "temperature_2m": "temp", "dew_point_temperature_2m": "dew"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def target_times(day):
    return pd.date_range(pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=1), periods=24, freq="h")


def time_axis(day, init, leads, delay_days, lag_hours):
    init = pd.Timestamp(init)
    assert init.tzinfo is not None
    expected = (pd.Timestamp(day) - pd.Timedelta(days=delay_days)).tz_localize("UTC")
    assert init == expected, (day, init, expected)
    deadline = pd.Timestamp(day, tz=TZ) - pd.Timedelta(hours=7)
    assert init + pd.Timedelta(hours=lag_hours) <= deadline
    times = (init + pd.to_timedelta(leads, unit="h")).tz_convert(TZ)
    target = target_times(day)
    assert times[0] < target[0] and times[-1] >= target[-1]
    return times, target


def interval_flux(values, leads, kind):
    """Return intervals (lead[i-1], lead[i]], discarding the baseline field."""
    a = np.asarray(values, dtype=np.float64)
    h = np.asarray(leads, dtype=np.float64)
    assert a.shape[0] == len(h) and np.all(np.diff(h) > 0)
    scale = h.reshape((-1,) + (1,) * (a.ndim - 1))
    width = np.diff(h).reshape((-1,) + (1,) * (a.ndim - 1))
    if kind == "since_init_mean": return np.diff(a * scale, axis=0) / width
    if kind == "accumulated_joules": return np.diff(a, axis=0) / (width * 3600)
    if kind == "previous_interval_mean": return a[1:]
    raise ValueError(kind)


def expand(values, times, target, interval=False):
    a = np.asarray(values)
    x, q = times.asi8, target.asi8
    if interval:
        assert len(a) == len(times) - 1
        ix = np.searchsorted(x, q, side="left") - 1
        assert np.all(ix >= 0) and np.all(ix < len(a))
        return a[ix]
    assert len(a) == len(times)
    assert q.min() >= x.min() and q.max() <= x.max()
    return np.column_stack([np.interp(q, x, a[:, j]) for j in range(a.shape[1])])


def spatial_stats(values, weights):
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    a = np.asarray(values, dtype=np.float64)
    assert a.shape[-1] == len(w) and np.isfinite(a).all()
    mean = np.sum(a * w, axis=-1)
    sd = np.sqrt(np.sum((a - mean[..., None]) ** 2 * w, axis=-1))
    return np.stack([mean, sd], axis=-1)


def checked_npz(path, delay, lag):
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["day"] == path.stem and sha(path) == record["sha256"]
    # Earliest ZIP probes predate these two audit fields. Recompute from the
    # retained initialization, and require equality when the fields exist.
    assumed = pd.Timestamp(record["init_time"]) + pd.Timedelta(hours=lag)
    deadline = pd.Timestamp(record["day"], tz=TZ) - pd.Timedelta(hours=7)
    assert assumed <= deadline
    if "available_at_assumed" in record: assert pd.Timestamp(record["available_at_assumed"]) == assumed
    if "deadline" in record: assert pd.Timestamp(record["deadline"]) == deadline
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert all(np.isfinite(a).all() for a in arrays.values())
    return record, arrays


def npz_day(path, source):
    record, arrays = checked_npz(path, 2, 24)
    folder = path.parent.parent
    with np.load(folder / "coordinates.npz", allow_pickle=False) as z:
        coords = dict(z)
    leads = coords["lead_hours"]
    expected = np.arange(36, 67, 6) if source == "icon" else np.arange(39, 64, 3)
    assert np.array_equal(leads, expected)
    times, target = time_axis(record["day"], record["init_time"], leads, 2, 24)
    columns, diagnostics = {"ts": target}, {}
    names = ICON_NAMES if source == "icon" else ENS_NAMES
    assert set(arrays) == set(names)
    if source == "ens":
        meta = json.loads((folder / "dataset.json").read_text())["variable_metadata"]
        assert meta["temperature_2m"]["attrs"]["units"] == "degree_Celsius"
        assert meta["dew_point_temperature_2m"]["attrs"]["units"] == "degree_Celsius"
        assert meta["downward_short_wave_radiation_flux_surface"]["attrs"]["comment"] == "Average flux since the previous forecast step."
    for raw, short in names.items():
        a = arrays[raw].astype(np.float64)
        radiation = short in {"direct", "diffuse", "solar"}
        if source == "icon":
            assert a.shape == (6, len(coords["latitude"]))
            if radiation: a = interval_flux(a, leads, "since_init_mean")
            # ICON triangular native cells have approximately equal areas.
            summary = spatial_stats(a, np.ones(a.shape[-1])); suffixes = ["mean", "spatial_sd"]
        else:
            assert a.shape == (9, 51, len(coords["latitude"]), len(coords["longitude"]))
            if radiation: a = interval_flux(a, leads, "previous_interval_mean")
            weights = np.broadcast_to(np.cos(np.deg2rad(coords["latitude"]))[:, None], a.shape[-2:]).ravel()
            member_means = spatial_stats(a.reshape(a.shape[:2] + (-1,)), weights)[..., 0]
            summary = np.stack([member_means.mean(1), member_means.std(1),
                                np.quantile(member_means, .1, axis=1),
                                np.quantile(member_means, .9, axis=1)], axis=-1)
            suffixes = ["mean", "member_sd", "p10", "p90"]
        if radiation:
            diagnostics[short] = {"negative_cells": int((a < 0).sum()), "min_flux": float(a.min()),
                                  "max_flux": float(a.max())}
            assert a.min() > -30 and a.max() < 1800, diagnostics[short]
        hourly = expand(summary, times, target, interval=radiation)
        for j, suffix in enumerate(suffixes): columns[f"wx_{source}_{short}_{suffix}"] = hourly[:, j]
    return pd.DataFrame(columns), {"file": str(path), "sha256": record["sha256"], "radiation": diagnostics}


def tigge_file(path):
    import eccodes as ec
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["status"] == "verified" and sha(path) == record["verification"]["sha256"]
    origin = record["request"]["origin"]
    fields, grids = {}, {}
    with path.open("rb") as f:
        while (h := ec.codes_grib_new_from_file(f)) is not None:
            try:
                init = str(ec.codes_get(h, "dataDate")); param = ec.codes_get(h, "paramId")
                lead = int(ec.codes_get(h, "endStep"))
                assert ec.codes_get(h, "dataTime") == 0 and ec.codes_get(h, "perturbationNumber") == 0
                assert not ec.codes_get(h, "numberOfMissing")
                a = ec.codes_get_array(h, "values")
                lat = ec.codes_get_array(h, "latitudes"); lon = ec.codes_get_array(h, "longitudes")
                assert np.isfinite(a).all()
                key = (init, param)
                if key not in fields:
                    fields[key] = {}; grids[key] = (lat, lon, ec.codes_get(h, "gridType"))
                assert lead not in fields[key]
                assert np.array_equal(grids[key][0], lat) and np.array_equal(grids[key][1], lon)
                fields[key][lead] = a
            finally: ec.codes_release(h)
    frames, diagnostics = [], []
    params = {167: "temp", 228164: "cloud"}
    if origin != "eccc": params[176] = "net_solar"
    for init in sorted({key[0] for key in fields}):
        init_time = pd.Timestamp(init, tz="UTC"); day = (init_time + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        leads = np.arange(60, 91, 6)
        times, target = time_axis(day, init_time, leads, 3, 48)
        columns = {"ts": target}
        for param, short in params.items():
            key = (init, param)
            assert sorted(fields[key]) == list(leads)
            a = np.stack([fields[key][lead] for lead in leads])
            lat, _, gridtype = grids[key]
            # Reduced Gaussian points are nearly equal area, regular latitude grids are not.
            weights = np.ones(len(lat)) if gridtype.startswith("reduced") else np.cos(np.deg2rad(lat))
            if param == 167: a = a - 273.15
            if param == 176:
                a = interval_flux(a, leads, "accumulated_joules")
                diagnostics.append({"day": day, "origin": origin, "negative_cells": int((a < 0).sum()),
                                    "min_flux": float(a.min()), "max_flux": float(a.max())})
                assert a.min() > -30 and a.max() < 1800
            summary = spatial_stats(a, weights)
            hourly = expand(summary, times, target, interval=param == 176)
            for j, suffix in enumerate(["mean", "spatial_sd"]):
                columns[f"wx_tigge_{origin}_{short}_{suffix}"] = hourly[:, j]
        frames.append(pd.DataFrame(columns))
    return pd.concat(frames), {"file": str(path), "sha256": record["verification"]["sha256"], "radiation": diagnostics}


def build(root, source, start, end, output):
    records, frames = [], []
    if source in {"icon", "ens"}:
        folders = ([root / "weather_collection_20261106" / s for s in ["icon_raw", "icon_raw_blocks", "icon"]]
                   if source == "icon" else [root / "additional_weather_20261106/ecmwf_ens/seasonal_samples"])
        paths = {}
        for folder in folders:
            for p in sorted((folder / "days").glob("*.npz")):
                if start <= p.stem <= end and p.with_suffix(".json").exists(): paths[p.stem] = p
        for day, p in sorted(paths.items()):
            frame, audit = npz_day(p, source); frames.append(frame); records.append(audit)
        assert frames, "no verified daily files"
        result = pd.concat(frames).sort_values("ts")
    else:
        groups = {}
        for p in sorted((root / "tigge_period").glob("*.grib")):
            frame, audit = tigge_file(p); records.append(audit)
            groups.setdefault(p.name.split("_")[0], []).append(frame)
        assert set(groups) == {"ecmwf", "ukmo", "eccc"}, "all three providers required"
        tables = []
        for origin, parts in groups.items():
            t = pd.concat(parts).set_index("ts").sort_index()
            assert t.index.is_unique
            tables.append(t)
        result = pd.concat(tables, axis=1).reset_index()
        result = result[(result.ts >= target_times(start)[0]) & (result.ts <= target_times(end)[-1])]
    assert not result.ts.duplicated().any()
    features = [c for c in result if c != "ts"]
    assert len(features) == {"icon": 6, "ens": 12, "tigge": 16}[source]
    full = pd.DataFrame({"ts": pd.date_range(target_times(start)[0], target_times(end)[-1], freq="h")})
    result = full.merge(result, on="ts", how="left", validate="one_to_one")
    assert np.isfinite(result[features].dropna().to_numpy()).all()
    complete = result[features].notna().all(axis=1)
    days = (result.ts - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
    counts = complete.groupby(days).sum()
    available = counts[counts == 24].index.tolist()
    # TIGGE's ablation is one provider bundle: incomplete days are wholly missing.
    result.loc[~days.isin(available), features] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{source}.parquet"; temp = path.with_suffix(".tmp.parquet")
    result.to_parquet(temp, index=False); temp.replace(path)
    info = {"source": source, "start": start, "end": end, "features": features,
            "available_days": available, "days": len(available), "missing_days": counts[counts < 24].index.tolist(),
            "training_days_before_F3": sum(d < "2024-06-01" for d in available),
            "F3_days": sum("2024-06-01" <= d <= "2025-05-31" for d in available),
            "sha256": sha(path), "code_sha256": sha(__file__), "inputs": records,
            "publication_status": "lag assumptions; not historical per-file certification",
            "radiation": "piecewise constant interval means, without clipping or solar reshaping",
            "statistics": "fixed regional summaries; ICON near equal-area, regular grids cosine latitude weights"}
    save_json(output / f"{source}.json", info)
    print(json.dumps({k: v for k, v in info.items() if k not in {"inputs", "available_days", "missing_days"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--source", choices=["icon", "ens", "tigge"], required=True)
    ap.add_argument("--start", default="2023-03-23")
    ap.add_argument("--end", default="2025-05-31")
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(); build(a.runs_root, a.source, a.start, a.end, a.output_dir)
