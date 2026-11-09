"""Run a small explicit KMA request plan; keep credentials out of saved evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://apihub.kma.go.kr"
ENDPOINTS = {"nwp_vars_down.php", "nwp_nc_vars_down.php"}
MAX_BYTES = 24 * 1024 * 1024


def payload_format(payload: bytes, content_type: str | None) -> str:
    if payload.startswith(b"GRIB"):
        return "grib"
    # The UM download API prepends the source filename and a newline.
    offset = payload.find(b"GRIB", 0, 512)
    if offset > 0 and re.fullmatch(rb"[A-Za-z0-9_.-]+\r?\n", payload[:offset]):
        return "grib_wrapped"
    if payload.startswith((b"CDF", b"\x89HDF")):
        return "netcdf"
    if payload.startswith(b"PK\x03\x04"):
        return "zip"
    if b"\x00" in payload[:1024] or (content_type or "").startswith("application/octet-stream"):
        return "binary"
    return "text"


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https" or target.hostname != "apihub.kma.go.kr":
            raise ValueError("Refusing a redirect outside the official KMA HTTPS host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be inside project outputs")
    plan = json.loads(args.plan.read_text())
    if not 1 <= len(plan) <= 50:
        raise ValueError("Use a bounded plan of at most 50 small requests")
    # Validate the entire plan before reading the key or sending any requests.
    for item in plan:
        if item["endpoint"] not in ENDPOINTS or not re.fullmatch(r"[a-zA-Z0-9_-]+", item["label"]):
            raise ValueError("Use an approved endpoint and a simple unique label")
        if any("key" in k.lower() for k in item["params"]):
            raise ValueError("Credentials must never appear in the request plan")
    if len({i["label"] for i in plan}) != len(plan):
        raise ValueError("Duplicate request labels")
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("\"'")
    key = env["KMA_API_KEY"]
    if not key:
        raise ValueError("KMA key is empty")
    out.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(OfficialRedirect())
    records = []
    for item in plan:
        label = item["label"]
        meta_path = out / f"{label}.json"
        if meta_path.exists():
            record = json.loads(meta_path.read_text())
            if record["request"] != item:
                raise ValueError("Existing request identity differs")
            records.append(record)
            print(json.dumps({"label": label, "status": "existing_evidence_reused"}), flush=True)
            continue
        started = time.monotonic()
        record = {"request": item, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "endpoint": HOST + "/api/typ06/url/" + item["endpoint"]}
        params = {**item["params"], "authKey": key}
        url = record["endpoint"] + "?" + urllib.parse.urlencode(params)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dataen-kma-small-probe/1.0"})
            with opener.open(request, timeout=60) as response:
                payload = response.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    raise ValueError("Response exceeds the small-probe size limit")
                record.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                              last_modified=response.headers.get("Last-Modified"))
            kind = payload_format(payload, record["content_type"])
            if kind == "text":
                # Some error pages echo the request URL. Redact before saving.
                encoding = "utf-8"
                try:
                    text = payload.decode(encoding)
                except UnicodeDecodeError:
                    encoding = "cp949"
                    text = payload.decode(encoding, errors="replace")
                record["source_text_encoding"] = encoding
                text = text.replace(key, "[REDACTED]")
                payload = text.encode("utf-8")
                record["response_excerpt"] = text[:700]
            elif kind == "grib_wrapped":
                record["filename_prefix"] = payload[:payload.index(b"GRIB")].decode("ascii").strip()
            path = out / f"{label}.{kind}"
            path.write_bytes(payload)
            record.update(status="response_saved", format=kind, bytes=len(payload),
                          sha256=hashlib.sha256(payload).hexdigest(), file=path.name)
        except Exception as exc:
            # Never print raw urllib exceptions: they may embed the authKey URL.
            record.update(status="request_failed", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.HTTPError):
                record["http_status"] = exc.code
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        records.append(record)
        print(json.dumps({k: record[k] for k in ["status", "http_status", "format", "bytes", "elapsed_seconds"] if k in record} | {"label": label}), flush=True)
    (out / "responses.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
