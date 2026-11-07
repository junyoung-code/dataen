"""Run verified dependencies as they finish, without restarting collectors."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weather_ablation_features import save_json, sha


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def reviewed_ens_gap(summary, review, dataset):
    """Accept exactly one independently re-read source gap; leave that day absent."""
    if not (review.get("source") == "ENS" and review.get("status") == "retry_failed"
            and summary.get("status") == "failed"
            and review.get("snapshot_id") == dataset.get("snapshot_id")
            and review.get("signature") == dataset.get("signature")
            and review.get("signature")
            and review.get("error_type") == "ValueError"
            and review.get("error", "").startswith("Missing/nonfinite source field ")
            and any(v.get("nonfinite_count", 0) > 0 for v in review.get("variables", {}).values())):
        return False
    error = {"day": review.get("day"), "type": review["error_type"], "error": review["error"]}
    wanted = summary.get("requested_dates", [])
    completed = [r.get("day") for r in summary.get("completed", [])]
    return (summary.get("errors") == [error] and error["day"] in wanted
            and len(wanted) == len(set(wanted)) and len(completed) == len(set(completed))
            and set(completed) == set(wanted) - {error["day"]})


def main():
    os.chdir(ROOT)
    logs = ROOT / "outputs/runs/logs"
    out = ROOT / "outputs/runs/ablation"
    out.mkdir(parents=True, exist_ok=True)
    state = out / "pipeline_status.json"
    env = os.environ | {"PYTHONPATH": str(ROOT / "outputs/weather_deps") + ":" + str(ROOT / "outputs/deps") + ":" + str(ROOT / "src"),
                       "OIBC_WORK_DIR": str(ROOT / "outputs/runs/work"), "OIBC_DATA_DIR": str(ROOT / "data"),
                       "OIBC_N_JOBS": "4"}
    env["DYLD_LIBRARY_PATH"] = str(ROOT.parent / "oibc_mac_mini/.venv/lib/python3.12/site-packages/lightgbm/lib")

    def status(phase, **extra):
        save_json(state, {"status": phase, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "pid": os.getpid(), **extra})

    def execute(label, args):
        status("running", step=label)
        print(label, flush=True)
        with (logs / (label + ".log")).open("w") as f:
            subprocess.run([sys.executable, "-u", *args], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)

    def ready(source):
        r = read(logs / f"{source}.status.json")
        if r.get("status") == "download_process_complete": return True
        if r.get("status") == "needs_review":
            if source == "ens":
                folder = ROOT / "outputs/runs/additional_weather_20261106/ecmwf_ens/seasonal_samples"
                review = read(out / "ens_source_gap_review.json")
                if reviewed_ens_gap(read(folder / "collection_summary.json"), review,
                                    read(folder / "dataset.json")):
                    # A source-defective day must never enter the feature builder.
                    assert not (folder / "days" / f"{review['day']}.npz").exists()
                    return True
            if source == "icon":
                c = read(ROOT / "outputs/runs/weather_collection_20261106/icon_raw_blocks/collection_summary.json")
                errors = c.get("errors", [])
                defects = read(out / "icon_archive_defects.json")
                known = {d["key"] for d in defects if d["classification"] in
                         {"empty", "truncated_no_zip_end", "missing_required_arrays", "initialization_name_mismatch"}} if isinstance(defects, list) else set()
                if errors and all(e["key"] in known for e in errors):
                    return True
            raise RuntimeError(f"{source} collector needs review")
        if source == "tigge":
            r = read(ROOT / "outputs/runs/tigge_period/collection_summary.json")
            if r.get("status") == "complete": return True
            for p in (ROOT / "outputs/runs/tigge_period").glob("*_control.json"):
                if read(p).get("error"): raise RuntimeError(f"TIGGE request failed: {p.name}")
        return False

    try:
        while True:
            baseline_done = (out / "baseline_F3.json").exists()
            if not baseline_done:
                exitfiles = [logs / (name + ".exit") for name in ["gfs1x", "gefsm"]]
                if all(p.exists() for p in exitfiles):
                    assert all(p.read_text().strip() == "0" for p in exitfiles), "G315 fetch process failed"
                    weather = ROOT / "outputs/runs/work/external/gfs"
                    audit = {}
                    wanted = {d.strftime("%Y%m%d") for d in __import__("pandas").date_range("2021-01-19", "2025-05-31")}
                    for name in ["raw", "raw_gefs", "raw_gfs1x", "raw_gefsm"]:
                        paths = sorted((weather / name).glob("*/gfs_*.parquet"))
                        present = {p.stem.removeprefix("gfs_") for p in paths}
                        needed = {d for d in wanted if d >= "20220531"} if name == "raw_gefsm" else wanted
                        missing = sorted(needed - present)
                        audit[name] = {"files": len(paths), "required_days": len(needed), "missing_dates": missing,
                                       "sha256": {str(p.relative_to(weather)): sha(p) for p in paths}}
                    assert audit["raw"]["missing_dates"] == ["20210207"]
                    assert not audit["raw_gefs"]["missing_dates"]
                    assert len(audit["raw_gfs1x"]["missing_dates"]) <= 2, "review GFS extra gaps"
                    assert not audit["raw_gefsm"]["missing_dates"], "review GEFS member gaps"
                    save_json(out / "g315_input_audit.json", audit)
                    execute("g315_build_v2", ["outputs/peer_source/tools/gfs_build.py", "--v2"])
                    execute("g315_build_gm", ["outputs/peer_source/tools/gfs_build.py", "--gm"])
                    execute("baseline_F3", ["scripts/run_weather_ablation.py", "baseline"])
                    baseline_done = True
            for source in ["icon", "ens", "tigge"]:
                feature_file = ROOT / "outputs/runs/features" / f"{source}.json"
                if not feature_file.exists() and ready(source):
                    execute(f"features_{source}", ["src/weather_ablation_features.py", "--runs-root", "outputs/runs",
                            "--source", source, "--output-dir", "outputs/runs/features"])
                if baseline_done and feature_file.exists() and not (out / f"{source}_F3.json").exists():
                    execute(f"{source}_F3", ["scripts/run_weather_ablation.py", source])
                    execute("comparison_latest", ["scripts/run_weather_ablation.py", "report"])
            pending = [s for s in ["baseline", "icon", "ens", "tigge"] if not (out / f"{s}_F3.json").exists()]
            if not pending:
                execute("comparison_F3", ["scripts/run_weather_ablation.py", "report"])
                status("complete", result="outputs/runs/ablation/comparison_F3.json")
                break
            status("waiting_for_collection", pending=pending)
            time.sleep(30)
    except Exception as exc:
        status("needs_review", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__": main()
