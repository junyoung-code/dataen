"""Independently decode TIGGE GRIB samples and validate their request identity."""
import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path

import eccodes as ec
import numpy as np

PARAMETERS = {"2_m_temperature": (167, "2t", "K", "instant"),
              "total_cloud_cover": (228164, "tcc", "%", "instant"),
              "surface_net_solar_radiation": (176, "ssr", "J m**-2", "accum")}


def verify(file):
    audit = json.loads(file.with_suffix(".json").read_text())
    request = audit["request"]
    assert audit["status"] == "downloaded", "Download incomplete"
    assert hashlib.sha256(file.read_bytes()).hexdigest() == audit["sha256"], "Checksum mismatch"
    expected = set(itertools.product([PARAMETERS[v][0] for v in request["variable"]], map(int, request["leadtime_hour"])))
    found, rows = set(), []
    lat_ref = lon_ref = None
    with file.open("rb") as stream:
        while (handle := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                row = {k: ec.codes_get(handle, k) for k in
                       ["dataDate", "dataTime", "shortName", "paramId", "startStep", "endStep", "stepType",
                        "units", "gridType", "numberOfDataPoints", "numberOfMissing", "centre",
                        "typeOfProcessedData", "perturbationNumber"]}
                values = ec.codes_get_array(handle, "values")
                lat = ec.codes_get_array(handle, "latitudes")
                lon = ec.codes_get_array(handle, "longitudes")
            finally:
                ec.codes_release(handle)
            identity = (row["paramId"], row["endStep"])
            assert identity in expected and identity not in found, "Unexpected or duplicate field"
            found.add(identity)
            assert row["dataDate"] == int(request["year"] + request["month"] + request["day"])
            assert row["dataTime"] == int(request["time"].replace(":", ""))
            assert request["forecast_type"] == "control_forecast"
            assert row["typeOfProcessedData"] == "cf" and row["perturbationNumber"] == 0, "Not a control forecast"
            assert not row["numberOfMissing"] and np.isfinite(values).all(), "Missing values"
            north, west, south, east = request["area"]
            assert lat.min() >= south - 1e-7 and lat.max() <= north + 1e-7
            assert lon.min() >= west - 1e-7 and lon.max() <= east + 1e-7
            if lat_ref is None:
                lat_ref, lon_ref = lat, lon
            else:
                assert np.array_equal(lat, lat_ref) and np.array_equal(lon, lon_ref), "Grid changes within request"
            parameter = next(p for p in PARAMETERS.values() if p[0] == row["paramId"])
            assert (row["shortName"], row["units"], row["stepType"]) == parameter[1:]
            if row["shortName"] == "2t":
                assert values.min() > 180 and values.max() < 340, "Temperature outside sanity bounds"
            elif row["shortName"] == "tcc":
                assert values.min() >= -0.01 and values.max() <= 100.01, "Cloud range mismatch"
            else:
                assert row["startStep"] == 0 and values.min() >= 0 and values.max() > 0
            row.update(min=float(values.min()), max=float(values.max()),
                       lat_min=float(lat.min()), lat_max=float(lat.max()),
                       lon_min=float(lon.min()), lon_max=float(lon.max()))
            rows.append(row)
    assert found == expected, "Requested fields missing"
    target = dt.date.fromisoformat(audit["target_day_kst"])
    deadline = dt.datetime.combine(target - dt.timedelta(days=1), dt.time(17),
                                   tzinfo=dt.timezone(dt.timedelta(hours=9)))
    assert dt.datetime.fromisoformat(audit["available_at_schedule"]) <= deadline
    audit.update(data_validation="passed", independently_verified_at=dt.datetime.now(dt.timezone.utc).isoformat())
    file.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    return {"file": file.name, "status": "passed", "bytes": file.stat().st_size,
            "fields": len(rows), "grid_points_per_field": rows[0]["numberOfDataPoints"],
            "requested_combinations_complete": True, "missing_values": 0,
            "current_48h_schedule_before_cutoff": True, "historical_publication_verified": False,
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    reports, errors = [], []
    for file in sorted((out / "tigge").glob("*.grib")):
        try:
            reports.append(verify(file))
        except Exception as exc:
            errors.append({"file": file.name, "error": f"{type(exc).__name__}: {exc}"})
    result = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Downloaded GRIB files only; pending requests are not counted as verified samples.",
              "status": "passed" if reports and not errors else "failed", "samples": reports, "errors": errors}
    (out / "tigge_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "samples": len(reports),
                      "fields": sum(r["fields"] for r in reports), "errors": errors}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
