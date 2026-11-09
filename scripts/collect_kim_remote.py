"""KIM archive collector: local quota coordinator, remote HTTPS downloader.

Only fixed meteorological bounds are sent to KMA. The SSH worker receives the
key in memory through stdin. No competition files or credentials are copied.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://apihub.kma.go.kr/api/typ06/url/nwp_nc_vars_down.php"
KST = dt.timezone(dt.timedelta(hours=9))
MAX_BODY = 1024 ** 2
VARIABLES = ("dswrsfc", "tcld")


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def parse_grid(text, params):
    if "# ERROR" in text:
        if "파일 없음" in text:
            return {"status": "file_missing"}
        raise ValueError("API body error")
    for name in ("vars", "group", "data", "tmfc", "hf"):
        if not re.search(r"\b" + name + "=" + re.escape(str(params[name])) + r"(?=\s|$)", text):
            raise ValueError("Response identity mismatch")
    match = re.search(r"# nx=(\d+)\s+ny=(\d+)", text)
    if not match:
        raise ValueError("No grid shape")
    nx, ny = map(int, match.groups())
    source = re.search(r"# fname=(\S+)", text)
    if not source or params["tmfc"] not in source.group(1) or ("ft%03d." % int(params["hf"])) not in source.group(1):
        raise ValueError("Source time mismatch")
    values = [float(v) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#") for v in line.split()]
    if len(values) != nx * ny or not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Truncated or nonfinite grid")
    missing = [v <= -900 or v >= 1e19 for v in values]
    valid = [v for v, m in zip(values, missing) if not m]
    upper = 1.001 if params["vars"] == "tcld" else 1700
    if any(v < -0.001 or v > upper for v in valid):
        raise ValueError("Variable range mismatch")
    return {"status": "verified" if valid else "all_values_missing", "shape": [ny, nx],
            "values": len(values), "missing_values": sum(missing),
            "minimum": min(valid) if valid else None, "maximum": max(valid) if valid else None,
            "source": source.group(1)}


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlsplit(newurl)
        if u.scheme != "https" or u.hostname != "apihub.kma.go.kr":
            raise ValueError("Nonofficial redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def worker(out):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = sys.stdin.readline().strip()
    if not key:
        raise ValueError("Missing key")
    (out / "responses").mkdir(exist_ok=True)
    (out / "days").mkdir(exist_ok=True)
    print(json.dumps({"status": "ready"}), flush=True)
    opener = urllib.request.build_opener(OfficialRedirect())
    for line in sys.stdin:
        job = json.loads(line)
        if job["op"] == "status":
            save(out / "status.json", job["value"])
            print('{"status":"saved"}', flush=True)
            continue
        if job["op"] == "day":
            save(out / "days" / (job["day"] + ".json"), job["value"])
            print('{"status":"saved"}', flush=True)
            continue
        if job["op"] == "config":
            config_path = out / "config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != job["value"]:
                raise ValueError("Configuration differs")
            save(config_path, job["value"])
            print('{"status":"saved"}', flush=True)
            continue
        p = job["params"]
        label = p["tmfc"] + "_" + p["vars"] + "_h" + p["hf"]
        if not re.fullmatch(r"\d{10}_(dswrsfc|tcld)_h\d{2}", label):
            raise ValueError("Invalid request")
        path = out / "responses" / (label + ".txt")
        meta = path.with_suffix(".json")
        rec = {"params": p, "checked_at": dt.datetime.now(KST).isoformat(), "endpoint": ENDPOINT}
        if meta.exists():
            old = json.loads(meta.read_text())
            if old["params"] != p:
                raise ValueError("Cached identity differs")
            if old["status"] in ("verified", "file_missing", "all_values_missing"):
                if digest(path.read_bytes()) != old["sha256"]:
                    raise ValueError("Cached hash mismatch")
                print(json.dumps(dict(old, reused=True)), flush=True)
                continue
        try:
            url = ENDPOINT + "?" + urllib.parse.urlencode(dict(p, authKey=key))
            with opener.open(url, timeout=45) as response:
                body = response.read(MAX_BODY + 1)
                rec["http_status"] = response.status
            rec["bytes"] = len(body)
            if len(body) > MAX_BODY:
                raise ValueError("Response too large")
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                text = body.decode("cp949", errors="replace")
            text = text.replace(key, "[REDACTED]")
            path.write_text(text)
            rec["sha256"] = digest(path.read_bytes())
            rec.update(parse_grid(text, p))
        except Exception as exc:
            rec.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                rec["http_status"] = exc.code
        save(meta, rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)


def coordinate(args):
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    if root / "outputs" not in out.parents:
        raise ValueError("Output outside project outputs")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Wait locally before opening SSH: a sleeping/unreachable Mac mini must
    # not prevent the requested UM-first sequence from remaining queued.
    if args.wait_for_um:
        previous = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
        previous.pop("error_type", None)
        while True:
            um_state = json.loads(args.wait_for_um.read_text()) if args.wait_for_um.exists() else {}
            if um_state.get("status") == "collected":
                break
            previous.update(status="waiting_for_um_collection", updated_at=dt.datetime.now(KST).isoformat(),
                            um_completed_days=um_state.get("completed_days"), um_status=um_state.get("status", "unknown"))
            save(out / "status.json", previous)
            time.sleep(30)
    key = next(line.split("=", 1)[1].strip().strip("\"'") for line in (root / ".env").read_text().splitlines() if line.startswith("KMA_API_KEY="))
    remote_script = args.remote_dir.rstrip("/") + "/collect_kim_remote.py"
    proc = subprocess.Popen(["ssh", "-o", "ServerAliveInterval=30", args.host,
                             "python3 -u " + shlex.quote(remote_script) + " --worker --output-dir " + shlex.quote(args.remote_dir)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc.stdin.write(key + "\n")
    proc.stdin.flush()
    if json.loads(proc.stdout.readline())["status"] != "ready":
        raise RuntimeError("Worker not ready")
    del key

    def send(job):
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("SSH worker disconnected; explicit resume required")
        return json.loads(line)

    config = {"start_target": args.start, "end_target": args.end, "variables": list(VARIABLES),
              "leads": list(range(24, 33)), "initialization": "target day minus one, 00UTC",
              "bounds": [125, 34, 129, 38], "endpoint": ENDPOINT,
              "gate": "Both variables at +24h absent: remaining leads untested, day not proven wholly absent",
              "training_ready": False, "release_time_verified": False}
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("Local configuration differs")
    save(out / "config.json", config)
    send({"op": "config", "value": config})
    from kma_shared_budget import SharedBudget
    if not args.shared_budget:
        raise ValueError("Explicit shared account ledger required")
    budget = SharedBudget(args.shared_budget, "KIM")
    ledger_path = out / "daily_budget.json"
    records = out / "records"
    records.mkdir(exist_ok=True)
    days_dir = out / "days"
    days_dir.mkdir(exist_ok=True)
    state = {"status": "collecting", "device": "Mac mini", "coordinator": "MacBook", "completed_days": 0,
             "full_days": 0, "partial_days": 0, "gate_missing_days": 0, "training_ready": False}

    def status(**updates):
        state.update(updates, updated_at=dt.datetime.now(KST).isoformat())
        save(out / "status.json", state)
        send({"op": "status", "value": state})

    def fetch(target, var, lead):
        init = (target - dt.timedelta(days=1)).strftime("%Y%m%d00")
        p = {"group": "KIMG", "data": "U", "vars": var, "tmfc": init, "hf": str(lead),
             "lon1": "125", "lat1": "34", "lon2": "129", "lat2": "38", "dataType": "TEXT", "help": "1"}
        cache = records / (init + "_" + var + "_h" + str(lead) + ".json")
        if cache.exists():
            old = json.loads(cache.read_text())
            if old["params"] != p:
                raise ValueError("Local cached identity differs")
            if old["status"] in ("verified", "file_missing", "all_values_missing"):
                return old
        for attempt in range(3):
            while True:
                if args.wait_for_um:
                    um_state = json.loads(args.wait_for_um.read_text()) if args.wait_for_um.exists() else {}
                    if um_state.get("status") != "collected":
                        status(status="waiting_for_um_collection", last_target=str(target),
                               um_completed_days=um_state.get("completed_days"),
                               um_status=um_state.get("status", "unknown"))
                        time.sleep(30)
                        continue
                ticket = budget.reserve()
                if ticket is not None:
                    break
                status(status="waiting_for_shared_daily_budget", last_target=str(target))
                time.sleep(60)
            # Reserve full response size before sending; replace with actual bytes
            # only on an acknowledged response. A crash remains conservatively charged.
            budget.export_owner(ledger_path)
            rec = send({"op": "fetch", "params": p})
            budget.record_bytes(ticket, rec.get("bytes", MAX_BODY))
            budget.export_owner(ledger_path)
            save(cache.with_name(cache.stem + "_attempt" + str(attempt) + ".json"), rec)
            save(cache, rec)
            if rec["status"] != "failed":
                return rec
            if rec.get("http_status") in (401, 403, 429) or rec.get("error_type") == "ValueError":
                raise RuntimeError("API access/rate/validation error; review required")
            time.sleep(2 ** attempt)
        raise RuntimeError("Three request attempts failed")

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    # Start at the known successful era; older requested history follows, with
    # a daily two-variable existence gate instead of 18 repeated absent calls.
    days.sort(key=lambda d: (d < dt.date(2025, 6, 1), d))
    state["expected_days"] = len(days)
    try:
        status(status="collecting")
        for day in days:
            results = [fetch(day, var, 24) for var in VARIABLES]
            gate_missing = all(r["status"] == "file_missing" for r in results)
            if not gate_missing:
                results += [fetch(day, var, lead) for lead in range(25, 33) for var in VARIABLES]
            good = sum(r["status"] == "verified" and r.get("missing_values", 0) == 0 for r in results)
            quality = "gate_missing" if gate_missing else "full" if good == 18 else "partial"
            record = {"target_day": str(day), "quality": quality, "verified_fields": good,
                      "expected_fields": 18, "untested_fields": 16 if gate_missing else 0,
                      "records": results, "training_ready": False}
            save(days_dir / (str(day) + ".json"), record)
            send({"op": "day", "day": str(day), "value": record})
            state[quality + "_days"] += 1
            state["completed_days"] += 1
            status(status="collecting", last_target=str(day))
        status(status="collection_finished_with_coverage_report")
    except Exception as exc:
        status(status="failed", error_type=type(exc).__name__)
        raise RuntimeError("KIM collection stopped; inspect sanitized status and request metadata") from None
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="macmini")
    parser.add_argument("--remote-dir")
    parser.add_argument("--start", default="2021-01-19")
    parser.add_argument("--end", default="2026-11-08")
    parser.add_argument("--shared-budget", type=Path)
    parser.add_argument("--wait-for-um", type=Path, help="Wait until this UM status reports collected")
    args = parser.parse_args()
    if args.worker:
        worker(args.output_dir)
    else:
        coordinate(args)
