"""Resume sequential monthly TIGGE requests for a bounded comparison period."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import time

import eccodes as ec
import numpy as np
from ecmwf.datastores import Client

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"2_m_temperature": (167, "K", "instant"),
          "total_cloud_cover": (228164, "%", "instant"),
          "surface_net_solar_radiation": (176, "J m**-2", "accum")}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def request_for(origin, dates):
    assert origin in {"ecmwf", "ukmo", "eccc"}
    assert len({(x.year, x.month) for x in dates}) == 1
    variables = ["2_m_temperature", "total_cloud_cover"]
    if origin != "eccc": variables.append("surface_net_solar_radiation")
    return {"origin": origin, "year": [str(dates[0].year)], "month": [f"{dates[0].month:02d}"],
            "day": [f"{x.day:02d}" for x in dates], "time": "00:00", "level_type": "single_level",
            "variable": variables, "forecast_type": "control_forecast", "data_format": "grib",
            "leadtime_hour": [str(x) for x in range(60, 91, 6)], "area": [42, 120, 30, 134]}


def verify(path, request):
    dates = [int(request["year"][0] + request["month"][0] + x) for x in request["day"]]
    parameters = {PARAMS[v][0]: PARAMS[v][1:] for v in request["variable"]}
    expected = set(itertools.product(dates, parameters, map(int, request["leadtime_hour"])))
    seen = set()
    with path.open("rb") as stream:
        while (h := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                day, param, lead = [ec.codes_get(h, k) for k in ("dataDate", "paramId", "endStep")]
                key = (day, param, lead)
                assert key in expected and key not in seen, f"unexpected/duplicate {key}"
                assert ec.codes_get(h, "dataTime") == 0
                assert ec.codes_get(h, "typeOfProcessedData") == "cf"
                assert ec.codes_get(h, "perturbationNumber") == 0
                assert (ec.codes_get(h, "units"), ec.codes_get(h, "stepType")) == parameters[param]
                values = ec.codes_get_array(h, "values")
                lat = ec.codes_get_array(h, "latitudes"); lon = ec.codes_get_array(h, "longitudes")
                assert not ec.codes_get(h, "numberOfMissing") and np.isfinite(values).all()
                assert lat.min() >= 30-1e-7 and lat.max() <= 42+1e-7
                assert lon.min() >= 120-1e-7 and lon.max() <= 134+1e-7
                if param == 167: assert values.min() > 180 and values.max() < 340
                elif param == 228164: assert values.min() >= -0.01 and values.max() <= 100.01
                else: assert ec.codes_get(h, "startStep") == 0 and values.min() >= 0
                seen.add(key)
            finally:
                ec.codes_release(h)
    assert seen == expected, f"missing {len(expected-seen)} fields"
    return {"status": "passed", "dates": len(dates), "fields": len(seen), "sha256": sha(path),
            "bytes": path.stat().st_size, "missing_values": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--start", required=True, help="Target date in Korea")
    ap.add_argument("--end", required=True)
    args = ap.parse_args(); out = args.output_dir.resolve()
    assert out.is_relative_to(ROOT / "outputs")
    lo, hi = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    assert lo <= hi and (hi-lo).days <= 900
    init_dates = [lo + dt.timedelta(days=i-3) for i in range((hi-lo).days+1)]
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k,v = line.split("=",1); env[k.strip()] = v.strip().strip("\"'")
    assert env["ECDSAPI_URL"] == "https://ecds.ecmwf.int/api"
    client = Client(url=env["ECDSAPI_URL"], key=env["ECDSAPI_KEY"], progress=False,
                    timeout=60, maximum_tries=3, retry_after=15, cleanup=False)
    client.check_authentication()
    completed = []
    for month in sorted({(x.year, x.month) for x in init_dates}):
        dates = [x for x in init_dates if (x.year,x.month) == month]
        for origin in ("ecmwf", "ukmo", "eccc"):
            request = request_for(origin, dates)
            label = f"{origin}_{month[0]}{month[1]:02d}_control"
            audit = out / f"{label}.json"; target = out / f"{label}.grib"
            signature = hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
            record = json.loads(audit.read_text()) if audit.exists() else {}
            assert not record or record["signature"] == signature
            record.update(request=request, signature=signature,
                          source_code_sha256=sha(Path(__file__)), availability_lag_hours=48)
            try:
                if record.get("status") == "verified":
                    assert target.exists() and sha(target) == record["verification"]["sha256"]
                else:
                    if not record.get("request_id"):
                        remote = client.submit("tigge-forecasts",request)
                        record["request_id"] = remote.request_id
                        save(audit, record)
                    else: remote = client.get_remote(record["request_id"])
                    previous = None
                    while True:
                        status = remote.status
                        record.update(status=status, checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
                        save(audit,record)
                        if status != previous:
                            print(json.dumps({"label":label,"status":status}),flush=True); previous=status
                        if status == "successful": break
                        if status not in {"accepted","running"}: remote.get_results(); raise RuntimeError(status)
                        time.sleep(30)
                    part = target.with_suffix(".part")
                    remote.download(str(part))
                    verification = verify(part,request)
                    part.replace(target)
                    record.update(status="verified",verification=verification)
                    save(audit,record)
                completed.append({"label":label,**record["verification"]})
                save(out/"collection_summary.json",{"status":"running","completed":completed,
                     "target_start":str(lo),"target_end":str(hi),"full_training":"not run"})
                print(json.dumps({"label":label,**record["verification"]}),flush=True)
            except Exception as exc:
                record.update(error_type=type(exc).__name__,error=str(exc).replace(env["ECDSAPI_KEY"],"[REDACTED]")[:1500])
                save(audit,record)
                print(json.dumps({"label":label,"error_type":type(exc).__name__,"error":record["error"]}),flush=True)
                raise SystemExit(1)
    save(out/"collection_summary.json",{"status":"complete","completed":completed,
         "target_start":str(lo),"target_end":str(hi),"full_training":"not run"})


if __name__ == "__main__": main()
