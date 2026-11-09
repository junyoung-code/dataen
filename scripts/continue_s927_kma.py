"""Resume the fixed S927 baseline, then only explicitly validated KMA inputs.

Run inside the isolated Mac mini lane. This never downloads data, submits a
competition result, or starts duplicate neural/grid workers. Feature collection
must write <source>_ready.json only after its independent input audit succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_s927_kma import save, sha


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run(root, sources):
    state_file = root / "runs/comparison_pipeline_status.json"
    def status(phase, **details):
        save(state_file, {"status": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
                          "pid": os.getpid(), **details})

    def wait_for(path, worker_status=None):
        while not path.exists():
            if worker_status:
                worker = read(worker_status)
                if worker.get("status", "").endswith("failed") or worker.get("status") == "failed":
                    raise RuntimeError(f"Required worker failed: {worker_status.name}: {worker}")
            status("waiting", required=str(path.relative_to(root)))
            time.sleep(30)

    def command(script, *args):
        name = "_".join(str(x).replace("--", "") for x in args)
        status("running", script=script, args=args)
        with (root / "runs" / f"pipeline_{name}.log").open("a") as log:
            subprocess.run([sys.executable, str(root / "scripts" / script), "--run-root", str(root), *args],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        # Grid status must attest a completed table, not merely a partially written file.
        while read(root / "runs/grid_status.json").get("status") != "complete":
            grid = read(root / "runs/grid_status.json")
            if grid.get("status", "").endswith("failed"):
                raise RuntimeError(f"Grid worker failed: {grid}")
            status("waiting", required="grid_status.complete")
            time.sleep(30)
        for fold in ["F2", "F3"]:
            for member in ["F153", "F155"]:
                command("run_s927_kma.py", "--member", member, "--fold", fold)
        for fold in ["F2", "F3"]:
            for member in ["N502c", "N530"]:
                wait_for(root / f"runs/{member}_baseline_{fold}/complete.json",
                         root / "runs/neural_pipeline_status.json")
        command("compare_s927_kma.py", "--source", "baseline")
        command("audit_s927_kma.py", "--source", "baseline")
        for source in sources:
            gate = root / f"work/external/kma/{source}_ready.json"
            wait_for(gate)
            ready = read(gate)
            assert ready["status"] == "verified" and ready["source"] == source
            # Refuse a file-only signal: verify every audited feature and metadata hash.
            for relative, expected in ready["sha256"].items():
                path = (root / relative).resolve()
                assert path.is_relative_to(root.resolve())
                assert sha(path) == expected, relative
            for fold in ["F2", "F3"]:
                for member in ["F153", "F155"]:
                    command("run_s927_kma.py", "--member", member, "--fold", fold, "--source", source)
            command("compare_s927_kma.py", "--source", source)
            command("audit_s927_kma.py", "--source", source)
        status("complete", sources=["baseline", *sources])
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", choices=["um", "kim"], default=["um"])
    args = parser.parse_args()
    root = args.run_root.resolve()
    (root / "runs").mkdir(exist_ok=True)
    with (root / "runs/comparison_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, args.sources)


if __name__ == "__main__":
    main()
