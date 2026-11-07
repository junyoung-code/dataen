"""Audit downloaded public grids independently of downloader completion counters."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import bz2
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(out):
    result = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(), "sources": {}}
    for source in ("gfs", "gefs", "icon"):
        folder = out / source
        meta = json.loads((folder / "dataset.json").read_text())
        config = meta["config"]["sources"][source]
        region = meta["config"]["region"]
        coords = dict(np.load(folder / "coordinates.npz", allow_pickle=False))
        errors, days, total, hashes = [], [], 0, {}
        lat, lon = coords["latitude"], coords["longitude"]
        assert (lat >= region["latitude_min"]).all() and (lat <= region["latitude_max"]).all()
        assert (lon >= region["longitude_min"]).all() and (lon <= region["longitude_max"]).all()
        assert np.array_equal(coords["lead_hours"], config["lead_hours"])
        if source != "icon":
            assert np.all(np.diff(lat) < 0) and np.all(np.diff(lon) > 0)
        for audit in sorted((folder / "days").glob("*.json")):
            day = audit.stem
            file = audit.with_suffix(".npz")
            try:
                a = json.loads(audit.read_text())
                digest = sha(file)
                assert a["sha256"] == digest, "Checksum mismatch"
                assert a["signature"] == meta["signature"], "Configuration mismatch"
                assert a["day"] == day, "Day mismatch"
                init = pd.Timestamp(day).tz_localize("UTC") - pd.Timedelta(days=2)
                deadline = pd.Timestamp(day).tz_localize("Asia/Seoul") - pd.Timedelta(hours=7)
                assert pd.Timestamp(a["init_time"]) == init
                assert pd.Timestamp(a["deadline"]) == deadline
                assert pd.Timestamp(a["available_at_assumed"]) <= deadline
                with np.load(file, allow_pickle=False) as data:
                    assert set(data.files) == set(config["variables"]), "Variable mismatch"
                    for name in data.files:
                        arr = data[name]
                        dims = a["variables"][name]["dimensions"]
                        expected = {"lead_time": len(config["lead_hours"]), "step": len(config["lead_hours"]),
                                    "latitude": len(lat), "longitude": len(lon), "values": len(lat),
                                    "ensemble_member": 31}
                        assert arr.shape == tuple(expected[d] for d in dims), "Shape mismatch"
                        assert np.isfinite(arr).all(), "Nonfinite values"
                        assert arr.min() == a["variables"][name]["min"]
                        assert arr.max() == a["variables"][name]["max"]
                    radiation = [data[name] for name in data.files if "short_wave_radiation_flux" in name]
                    assert any((a > 0).any() for a in radiation), "Zero-filled solar radiation"
                days.append(day); total += file.stat().st_size; hashes[day] = digest
            except Exception as error:
                errors.append({"day": day, "error": f"{type(error).__name__}: {error}"})
        expected_days = pd.date_range(config["start"], config["end"]).strftime("%Y-%m-%d").tolist()
        missing = sorted(set(expected_days) - set(days))
        result["sources"][source] = {"validated_days": len(days), "start": min(days, default=None),
            "end": max(days, default=None), "bytes": total, "configured_days": len(expected_days),
            "missing_days": missing, "errors": errors, "daily_sha256": hashes,
            "status": "invalid" if errors else "complete" if not missing else "partial",
            "historical_publication_certified": False}
    return result


def compare_existing_gefs(out):
    """Independently reproduce the old cos(latitude)-weighted member summaries."""
    folder = out / "gefs"
    old = out / "reused/gefs_b1_full/days"
    with np.load(folder / "coordinates.npz") as c:
        lat, lon, leads = c["latitude"], c["longitude"], c["lead_hours"]
    la, lo = (lat >= 32) & (lat <= 40), (lon >= 124) & (lon <= 132)
    w = np.cos(np.deg2rad(lat[la]))[None, None, :, None]
    result = []
    names = {"downward_short_wave_radiation_flux_surface": "gefs_sw", "total_cloud_cover_atmosphere": "gefs_cloud"}
    for day in ("2021-06-01", "2024-06-01"):
        with np.load(folder / "days" / f"{day}.npz") as new, np.load(old / f"{day}.members.npz") as previous:
            for name, prefix in names.items():
                raw = new[name][:, :, la, :][:, :, :, lo]
                values = raw.astype("float64")
                for j, lead in enumerate(leads):
                    if lead % 6 == 0:
                        values[:, j] = 2 * raw[:, j] - raw[:, j - 1]
                values = np.clip(values, 0, 100) if prefix == "gefs_cloud" else np.maximum(values, 0)
                means = (values * w).sum(axis=(2, 3)) / (w.sum() * values.shape[3])
                delta = float(np.max(np.abs(means - previous[prefix])))
                result.append({"day": day, "variable": prefix, "max_absolute_difference": delta,
                               "allclose_atol_1e_6": bool(np.allclose(means, previous[prefix], atol=1e-6, rtol=0))})
    return result


def verify_icon_raw(out):
    reports = {}
    for kind in ("icon_raw", "icon_raw_blocks"):
        folder = out / kind
        days, errors, total = [], [], 0
        for file in sorted((folder / "days").glob("*.npz")):
            try:
                a = json.loads(file.with_suffix(".json").read_text())
                assert sha(file) == a["sha256"]
                assert file.stem == a["day"]
                init = pd.Timestamp(a["init_time"])
                assert init == pd.Timestamp(file.stem).tz_localize("UTC") - pd.Timedelta(days=2)
                deadline = pd.Timestamp(file.stem).tz_localize("Asia/Seoul") - pd.Timedelta(hours=7)
                assert init + pd.Timedelta(hours=24) <= deadline
                with np.load(file, allow_pickle=False) as data:
                    assert len(data.files) == 3
                    for name in data.files:
                        arr = data[name]
                        assert arr.shape == (6, 9716) and np.isfinite(arr).all()
                        assert arr.min() == a["variables"][name]["min"]
                        assert arr.max() == a["variables"][name]["max"]
                    assert any((data[n] > 0).any() for n in data.files if "short_wave" in n)
                days.append(file.stem); total += file.stat().st_size
            except Exception as error:
                errors.append({"file": str(file.relative_to(out)), "error": str(error)})
        reports[kind] = {"validated_days": len(days), "days": days, "bytes": total, "errors": errors}
    comparison = []
    for day in sorted(set(reports["icon_raw"]["days"]) & set(reports["icon_raw_blocks"]["days"])):
        with np.load(out / "icon_raw/days" / (day + ".npz")) as full, np.load(out / "icon_raw_blocks/days" / (day + ".npz")) as partial:
            for name in full.files:
                comparison.append({"day": day, "variable": name,
                                   "array_equal": bool(np.array_equal(full[name], partial[name]))})
    reports["full_vs_block_reads"] = comparison
    unique = {p.stem for p in (out / "icon/days").glob("*.npz")}
    unique.update(reports["icon_raw"]["days"]); unique.update(reports["icon_raw_blocks"]["days"])
    reports["unique_combined_days"] = len(unique)
    return reports


def verify_icon_recent(out):
    folder = out / "icon_recent"
    if not (folder / "coordinates.npz").exists():
        return {"validated_days": 0, "days": [], "errors": [], "status": "not_collected"}
    sys.path.insert(0, str(out / ".runtime/eccodes"))
    import eccodes
    coords = dict(np.load(folder / "coordinates.npz", allow_pickle=False))
    indices = coords["native_cell_index"]
    grid = json.loads((folder / "grid.json").read_text())
    days, errors, total, raw_bytes = [], [], 0, 0
    names = {"aswdir_s": "mean_surface_direct_short_wave_radiation_flux",
             "aswdifd_s": "mean_surface_diffuse_short_wave_radiation_flux", "clct": "total_cloud_cover"}
    for file in sorted((folder / "days").glob("*.npz")):
        try:
            audit = json.loads(file.with_suffix(".json").read_text())
            assert sha(file) == audit["sha256"]
            assert sha(folder / "coordinates.npz") == audit["coordinates_sha256"]
            init = pd.Timestamp(file.stem).tz_localize("UTC") - pd.Timedelta(days=2)
            deadline = pd.Timestamp(file.stem).tz_localize("Asia/Seoul") - pd.Timedelta(hours=7)
            assert pd.Timestamp(audit["init_time"]) == init
            assert pd.Timestamp(audit["deadline"]) == deadline
            assert pd.Timestamp(audit["available_at_assumed"]) <= deadline
            assert audit["publication_verified"] is False
            assert len(audit["sources"]) == 18
            with np.load(file, allow_pickle=False) as data:
                assert set(data.files) == set(names.values())
                for name in data.files:
                    arr = data[name]
                    assert arr.shape == (6, len(indices)) and arr.dtype == np.float64
                    assert np.isfinite(arr).all()
                    assert arr.min() == audit["variables"][name]["min"]
                    assert arr.max() == audit["variables"][name]["max"]
                assert any((data[name] > 0).any() for name in data.files if "short_wave" in name)
                actual_fields = set()
                for source in audit["sources"]:
                    raw_file = folder / "grib" / init.strftime("%Y%m%d") / Path(source["key"]).name
                    assert sha(raw_file) == source["sha256"]
                    assert raw_file.stat().st_size == source["bytes"]
                    raw_bytes += raw_file.stat().st_size
                    m = source["grib"]
                    name = m["shortName"].lower(); lead = m["endStep"]
                    actual_fields.add((name, lead))
                    assert str(m["dataDate"]) == init.strftime("%Y%m%d") and m["dataTime"] == 0
                    assert m["numberOfMissing"] == 0 and m["uuidOfHGrid"] == grid["latitude"]["uuidOfHGrid"]
                    # Decode all three fields at one lead independently for every day.
                    if lead == 48:
                        handle = eccodes.codes_new_from_message(bz2.decompress(raw_file.read_bytes()))
                        try:
                            assert eccodes.codes_get(handle, "endStep") == lead
                            original = eccodes.codes_get_array(handle, "values")[indices]
                        finally:
                            eccodes.codes_release(handle)
                        assert np.array_equal(original, data[names[name]][2]), "Raw GRIB/subset mismatch"
                assert actual_fields == {(n, int(h)) for n in names for h in coords["lead_hours"]}
            days.append(file.stem); total += file.stat().st_size
        except Exception as error:
            errors.append({"file": str(file.relative_to(out)), "error": f"{type(error).__name__}: {error}"})
    return {"validated_days": len(days), "days": days, "bytes": total, "raw_grib_bytes": raw_bytes,
            "errors": errors, "raw_check": "All original hashes; three fields at lead48 redecoded for every day",
            "historical_publication_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under project outputs")
    report = verify(out)
    report["gefs_existing_cache_comparison"] = compare_existing_gefs(out)
    report["icon_raw"] = verify_icon_raw(out)
    report["icon_recent"] = verify_icon_recent(out)
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({source: {k: v for k, v in r.items() if k not in ("missing_days", "daily_sha256")}
                      for source, r in report["sources"].items()}, indent=2))
    print(json.dumps(report["gefs_existing_cache_comparison"], indent=2))
    if any(r["errors"] for r in report["sources"].values()):
        raise SystemExit(2)
    if not all(r["allclose_atol_1e_6"] for r in report["gefs_existing_cache_comparison"]):
        raise SystemExit(3)
    if any(report["icon_raw"][name]["errors"] for name in ("icon_raw", "icon_raw_blocks")):
        raise SystemExit(4)
    if not all(r["array_equal"] for r in report["icon_raw"]["full_vs_block_reads"]):
        raise SystemExit(5)
    if report["icon_recent"]["errors"]:
        raise SystemExit(6)
