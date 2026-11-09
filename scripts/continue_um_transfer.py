"""After collection, audit/PCA UM inputs and publish them to the user's Mac mini.

Only derived public UM forecasts are copied. Publish the ready marker last,
after remote SHA-256 verification; the remote S927 controller handles training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--remote-root", required=True, help="Directory relative to the remote home")
    args = parser.parse_args()
    r = args.run_root.resolve()
    assert r.is_relative_to(ROOT / "outputs")
    assert not Path(args.remote_root).is_absolute() and ".." not in Path(args.remote_root).parts
    state_path = r / "um_transfer_status.json"
    def status(phase, **kw):
        save(state_path, {"status": phase, "pid": os.getpid(),
                          "updated_at": datetime.now(timezone.utc).isoformat(), **kw})
    with (r / "um_transfer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                path = r / "um_core_collection/status.json"
                source = json.loads(path.read_text()) if path.exists() else {}
                if source.get("status") == "failed":
                    raise RuntimeError(f"Collector failed: {source}")
                if source.get("status") == "collected":
                    break
                status("waiting_for_collection", completed_days=source.get("completed_days"))
                time.sleep(30)
            features = r / "um_features"
            status("auditing_and_building_features")
            with (r / "um_feature_build.log").open("a") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/build_um_features.py"),
                                "--collection", str(r / "um_core_collection"),
                                "--points", str(r / "um_fixed_points.json"),
                                "--reference", str(r / "um_core_reference_verification.json"),
                                "--output-dir", str(features)], stdout=log, stderr=subprocess.STDOUT, check=True)
            status("transferring_audited_features")
            remote_dir = args.remote_root + "/work/external/kma"
            mkdir_script = "from pathlib import Path; import sys; (Path.home()/sys.argv[1]).mkdir(parents=True,exist_ok=True)"
            subprocess.run(["ssh", args.ssh_host, "python3 -c " + shlex.quote(mkdir_script) + " " + shlex.quote(remote_dir)], check=True)
            names = ["um_F2_pca.parquet", "um_F3_pca.parquet", "um_features.json", "um_response_manifest.json",
                     "um_F2_pca_fit.npz", "um_F3_pca_fit.npz"]
            subprocess.run(["scp", *[str(features / n) for n in names],
                            f"{args.ssh_host}:{remote_dir}/"], check=True)
            subprocess.run(["scp", str(features / "um_ready.json"),
                            f"{args.ssh_host}:{remote_dir}/um_ready.pending"], check=True)
            verify = """from pathlib import Path
import hashlib,json,sys
r=(Path.home()/sys.argv[1]).resolve()
p=r/'work/external/kma/um_ready.pending'
m=json.loads(p.read_text())
assert m['status']=='verified' and m['source']=='um'
for name,expected in m['sha256'].items():
    f=(r/name).resolve()
    assert r in f.parents
    assert hashlib.sha256(f.read_bytes()).hexdigest()==expected,name
p.replace(p.with_name('um_ready.json'))
print('UM feature hashes verified; ready marker published')
"""
            subprocess.run(["ssh", args.ssh_host, "python3 - " + shlex.quote(args.remote_root)],
                           input=verify, text=True, check=True)
            status("ready_published", remote_root=args.remote_root)
        except Exception as exc:
            status("failed", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
