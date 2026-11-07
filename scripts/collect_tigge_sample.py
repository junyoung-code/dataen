"""Submit or resume one explicit TIGGE sample request; credentials stay local."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time

from ecmwf.datastores import Client

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs") or not re.fullmatch(r"[a-z0-9_]+", args.label):
        raise ValueError("Use a project outputs directory and a simple sample label")
    request = json.loads(args.request_file.read_text())
    if request["origin"] not in {"ecmwf", "ukmo", "eccc", "dwd", "kma", "ncep"}:
        raise ValueError("Only the reviewed CC BY centres are supported")
    if request.get("area") != [42, 120, 30, 134]:
        raise ValueError("This sampler uses the preselected common geographical region")
    folder = out / "tigge"
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / f"{args.label}.json"
    target = folder / f"{args.label}.grib"
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    record = json.loads(status_path.read_text()) if status_path.exists() else {}
    if record and record.get("signature") != signature:
        raise ValueError("Existing request identity differs")
    if record.get("status") == "downloaded":
        if not target.is_file() or sha(target) != record["sha256"]:
            raise ValueError("Existing sample checksum mismatch")
        print(json.dumps({"label": args.label, "status": "reused", "bytes": target.stat().st_size}), flush=True)
        return
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
    token = env["ECDSAPI_KEY"]
    url = env["ECDSAPI_URL"]
    if url != "https://ecds.ecmwf.int/api":
        raise ValueError("Unexpected ECDS API destination")
    client = Client(url=url, key=token, progress=False, timeout=60,
                    maximum_tries=3, retry_after=10, sleep_max=15, cleanup=False)
    init = dt.datetime(int(request["year"]), int(request["month"]), int(request["day"]),
                       int(request["time"].split(":")[0]), tzinfo=dt.timezone.utc)
    record.update({"signature": signature, "request": request, "dataset": "tigge-forecasts",
                   "init_time": init.isoformat(), "target_day_kst": (init + dt.timedelta(days=3)).date().isoformat(),
                   "available_at_schedule": (init + dt.timedelta(hours=48)).isoformat(),
                   "availability_note": "Current official 48-hour TIGGE delay; year-specific historical applicability not yet independently verified.",
                   "source_code_sha256": sha(Path(__file__)), "data_validation": "pending"})
    try:
        client.check_authentication()
        record["authentication_verified"] = True
        if record.get("request_id"):
            remote = client.get_remote(record["request_id"])
        else:
            record["status"] = "submitting"
            save(status_path, record)
            remote = client.submit("tigge-forecasts", request)
            record["request_id"] = remote.request_id
            record["status"] = "submitted"
            save(status_path, record)
        started = time.monotonic()
        previous = None
        while True:
            status = remote.status
            record.update(status=status, checked_at=dt.datetime.now(dt.timezone.utc).isoformat())
            save(status_path, record)
            if status != previous:
                print(json.dumps({"label": args.label, "status": status, "request_id": remote.request_id}), flush=True)
                previous = status
            if status == "successful":
                temp = target.with_suffix(".grib.part")
                remote.download(str(temp))
                with temp.open("rb") as file:
                    if file.read(4) != b"GRIB":
                        raise ValueError("Downloaded payload is not a raw GRIB file")
                temp.replace(target)
                record.update(status="downloaded", bytes=target.stat().st_size, sha256=sha(target))
                save(status_path, record)
                print(json.dumps({"label": args.label, "status": "downloaded", "bytes": target.stat().st_size}), flush=True)
                return
            if status not in {"accepted", "running"}:
                remote.get_results()
                raise RuntimeError(f"Unexpected terminal request status: {status}")
            if time.monotonic() - started >= args.wait_seconds:
                print(json.dumps({"label": args.label, "status": status, "resume": "rerun identical command; existing request_id will be reused"}), flush=True)
                raise SystemExit(3)
            time.sleep(15)
    except Exception as exc:
        record.update(error_type=type(exc).__name__, error=str(exc).replace(token, "[REDACTED]")[:1800])
        save(status_path, record)
        print(json.dumps({"label": args.label, "error_type": record["error_type"], "error": record["error"]}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
