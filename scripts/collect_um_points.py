"""Collect explicitly approved UM point time series with a persistent daily budget.

Only fixed meteorological points are sent to the official KMA API. Competition
data is never read. Credentials are read locally and never written to evidence.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from probe_kma_api import ROOT, OfficialRedirect
from kma_shared_budget import SharedBudget

ENDPOINT = "https://apihub.kma.go.kr/api/typ06/url/um_grib_pt_tmfc.php"
KST = timezone(timedelta(hours=9))
VARS = {4200: "tdsw", 6213: "tcam"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def parse(text, initialization):
    """Validate the service's five-column rows, preserving missing forecasts."""
    if not text.startswith("#START7777") or "#7777END" not in text:
        raise ValueError("Not a complete UM point response")
    init = datetime.strptime(initialization, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    rows, seen = [], set()
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Unexpected point response columns")
        tmfc, tmef, varn, level, value = fields
        valid = datetime.strptime(tmef, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        lead = (valid - init).total_seconds() / 3600
        varn, level, value = int(varn), int(level), float(value)
        assert tmfc == initialization and lead.is_integer() and 16 <= lead <= 39
        assert varn in VARS and level == 16777215
        identity = (int(lead), varn)
        assert identity not in seen
        seen.add(identity)
        missing = value <= -900 or value >= 1e19 or not (-float("inf") < value < float("inf"))
        if not missing:
            assert -1e-3 <= value <= (1700 if varn == 4200 else 1.001), (varn, value)
        rows.append({"lead": int(lead), "variable": VARS[varn], "value": None if missing else value})
    return rows


class Budget:
    def __init__(self, path, requests, nbytes):
        self.path, self.requests, self.nbytes = path, requests, nbytes
        self.lock = threading.Lock()
        self.state = json.loads(path.read_text()) if path.exists() else {}

    def reserve(self):
        with self.lock:
            today = datetime.now(KST).date().isoformat()
            # Headroom includes earlier sample requests made outside this collector.
            usage = self.state.setdefault(today, {"requests": 256, "bytes": 100 * 1024**2})
            if usage["requests"] >= self.requests or usage["bytes"] + 1024**2 > self.nbytes:
                raise RuntimeError("Daily collection budget reached; do not bypass or change credentials")
            usage["requests"] += 1
            save(self.path, self.state)
            return today

    def record_bytes(self, day, size):
        with self.lock:
            self.state[day]["bytes"] += size
            save(self.path, self.state)


def collect(args):
    out = args.output_dir.resolve()
    assert out.is_relative_to(ROOT / "outputs")
    out.mkdir(parents=True, exist_ok=True)
    points_bytes = args.points.read_bytes()
    points = json.loads(points_bytes)["points"]
    assert len(points) == 9 and len({p["label"] for p in points}) == 9
    signature = digest(points_bytes)
    key = next(line.split("=", 1)[1].strip().strip("\"'") for line in (ROOT / ".env").read_text().splitlines()
               if line.startswith("KMA_API_KEY="))
    assert key
    budget = (SharedBudget(args.shared_budget, "UM") if args.shared_budget
              else Budget(out / "daily_budget.json", args.max_daily_requests, args.max_daily_bytes))
    stop = threading.Event()
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    assert 0 < len(days) <= 1611 and 1 <= args.workers <= 8
    config = {"endpoint": ENDPOINT, "points_sha256": signature, "start": str(args.start), "end": str(args.end),
              "initialization": "target day minus 1, 00UTC", "leads": [args.lead_start, args.lead_end, 1], "variables": VARS}
    if (out / "config.json").exists():
        assert json.loads((out / "config.json").read_text()) == json.loads(json.dumps(config))
    else:
        save(out / "config.json", config)
    (out / "responses").mkdir(exist_ok=True)
    (out / "days").mkdir(exist_ok=True)

    def fetch(day, point, refresh=False):
        init = (day - timedelta(days=1)).strftime("%Y%m%d00")
        params = {"group": "UMKR", "nwp": "N512", "data": "U", "varn": "4200,6213", "tmfc": init,
                  "ef": f"{args.lead_start},{args.lead_end},1", "X": str(point["X"]), "Y": str(point["Y"]), "help": "1"}
        dest = out / "responses" / f"{day}_{point['label']}.txt"
        meta = dest.with_suffix(".json")
        prior = None
        if dest.exists() and meta.exists():
            old = json.loads(meta.read_text())
            assert old["params"] == params and digest(dest.read_bytes()) == old["sha256"]
            if old["status"] == "verified":
                if not refresh:
                    return parse(dest.read_text(), init), old
                assert not old.get("partial_recheck_attempted"), "Only one partial recheck is allowed"
                prior = old
                attempts = out / "attempts"
                attempts.mkdir(exist_ok=True)
                backup = attempts / (dest.stem + "_initial.txt")
                backup.write_bytes(dest.read_bytes())
                save(backup.with_suffix(".json"), old)
            else:
                raise RuntimeError(f"Previous failed response needs review: {meta.name}")
        for attempt in range(3):
            if stop.is_set():
                raise RuntimeError("Collection stopped after another request failed")
            charge_day = budget.reserve()
            while charge_day is None:
                if stop.is_set():
                    raise RuntimeError("Collection stopped while waiting for quota")
                time.sleep(30)
                charge_day = budget.reserve()
            if args.shared_budget:
                budget.export_owner(out / "daily_budget.json")
            url = ENDPOINT + "?" + urllib.parse.urlencode({**params, "authKey": key})
            before = time.monotonic()
            try:
                with urllib.request.build_opener(OfficialRedirect()).open(url, timeout=45) as response:
                    payload = response.read(1024**2 + 1)
                    code = response.status
                budget.record_bytes(charge_day, len(payload))
                if args.shared_budget:
                    budget.export_owner(out / "daily_budget.json")
                assert len(payload) <= 1024**2
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    text = payload.decode("cp949", errors="replace")
                text = text.replace(key, "[REDACTED]")
                dest.write_text(text)
                record = {"status": "response_saved", "params": params, "sha256": digest(dest.read_bytes()),
                          "http_status": code, "seconds": time.monotonic() - before}
                rows = parse(text, init)
                assert all(args.lead_start <= row["lead"] <= args.lead_end for row in rows)
                record.update(status="verified", rows=len(rows))
                if prior:
                    record["partial_recheck_attempted"] = True
                    record["previous_response"] = {"file": str(backup.relative_to(out)), "sha256": prior["sha256"]}
                    # Preserve both original API replies, even if the second is worse.
                    fresh = attempts / (dest.stem + "_recheck.txt")
                    fresh.write_bytes(dest.read_bytes())
                    old_rows = parse(backup.read_text(), init)
                    old_by_key = {(x["lead"], x["variable"]): x["value"] for x in old_rows}
                    for row in rows:
                        previous = old_by_key.get((row["lead"], row["variable"]))
                        if previous is not None and row["value"] is not None:
                            assert abs(previous - row["value"]) <= 1e-5 * max(1, abs(previous))
                    record["recheck_response"] = {"file": str(fresh.relative_to(out)), "sha256": digest(fresh.read_bytes())}
                    if len(rows) < len(old_rows):
                        dest.write_bytes(backup.read_bytes())
                        record.update(sha256=prior["sha256"], rows=len(old_rows), selected_response="initial")
                        rows = old_rows
                    else:
                        record["selected_response"] = "recheck"
                save(meta, record)
                return rows, record
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                code = getattr(exc, "code", None)
                if code in (403, 429) or attempt == 2:
                    stop.set()
                    save(meta, {"status": "failed", "params": params, "http_status": code,
                                "error_type": type(exc).__name__})
                    raise RuntimeError(f"Official API request failed ({code}, {type(exc).__name__})") from None
                time.sleep(2**attempt)
            except Exception as exc:
                stop.set()
                save(meta, {"status": "failed", "params": params, "error_type": type(exc).__name__})
                raise RuntimeError(f"Response validation failed: {meta.name} ({type(exc).__name__})") from None

    def one_day(day):
        result, hashes = {}, {}
        for point in points:
            rows, meta = fetch(day, point)
            result[point["label"]] = rows
            hashes[point["label"]] = meta["sha256"]
        expected = (args.lead_end - args.lead_start + 1) * 2
        # Let the first pass warm the archive before a bounded partial recheck.
        for point in points:
            label = point["label"]
            meta = json.loads((out / "responses" / f"{day}_{label}.json").read_text())
            if len(result[label]) < expected and not meta.get("partial_recheck_attempted"):
                result[label], meta = fetch(day, point, refresh=True)
                hashes[label] = meta["sha256"]
        save(out / "days" / f"{day}.json", {"target_day": str(day), "points_sha256": signature,
             "response_sha256": hashes, "points": result})
        return str(day)

    started = time.monotonic()
    completed = []
    with (out / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one_day, day) for day in days]
                for future in cf.as_completed(futures):
                    completed.append(future.result())
                    state = {"status": "collecting", "completed_days": len(completed), "expected_days": len(days),
                             "elapsed_seconds": time.monotonic() - started, "last_completed_day": completed[-1]}
                    save(out / "status.json", state)
                    if len(completed) % 25 == 0:
                        print(json.dumps(state), flush=True)
        except Exception as exc:
            stop.set()
            save(out / "status.json", {"status": "failed", "error": str(exc), "completed_days": len(completed)})
            raise
        save(out / "status.json", {"status": "collected", "completed_days": len(completed),
                                  "expected_days": len(days), "elapsed_seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lead-start", type=int, default=16)
    parser.add_argument("--lead-end", type=int, default=39)
    parser.add_argument("--max-daily-requests", type=int, default=19000)
    parser.add_argument("--max-daily-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--shared-budget", type=Path)
    args = parser.parse_args()
    assert args.max_daily_requests <= 19000 and args.max_daily_bytes <= 4 * 1024**3
    assert 16 <= args.lead_start <= args.lead_end <= 39
    collect(args)


if __name__ == "__main__":
    main()
