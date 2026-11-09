"""Process-safe daily account quota, reserved before each network request."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import threading
import uuid

KST = timezone(timedelta(hours=9))
MAX_RESPONSE = 1024**2 + 1


class SharedBudget:
    def __init__(self, path, owner, request_limit=20000, byte_limit=5_000_000_000,
                 headroom_requests=512, headroom_bytes=128 * 1024**2):
        self.path, self.owner = Path(path), owner
        self.policy = dict(request_limit=request_limit, byte_limit=byte_limit,
                           headroom_requests=headroom_requests, headroom_bytes=headroom_bytes)
        if not 0 < request_limit <= 20000 or not 0 < byte_limit <= 5_000_000_000:
            raise ValueError("Invalid account limits")
        self.mutex = threading.Lock()

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.mutex, self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {"policy": self.policy, "days": {}}
            if state["policy"] != self.policy:
                raise ValueError("Shared quota policy differs")
            yield state
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(self.path)

    def day(self, state, date):
        return state["days"].setdefault(date, {"requests": self.policy["headroom_requests"],
            "bytes": self.policy["headroom_bytes"], "owners": {}, "inflight": {}})

    def seed(self, legacy):
        """One-time import after both legacy collectors have stopped."""
        with self.transaction() as state:
            if state.get("migration"):
                raise ValueError("Quota already migrated")
            if state["days"]:
                raise ValueError("Cannot seed a used ledger")
            for owner, days in legacy.items():
                for date, usage in days.items():
                    day = self.day(state, date)
                    day["requests"] += usage["requests"]
                    day["bytes"] += usage["bytes"]
                    day["owners"][owner] = dict(usage)
            state["migration"] = datetime.now(KST).isoformat()

    def reserve(self, date=None):
        date = date or datetime.now(KST).date().isoformat()
        with self.transaction() as state:
            day = self.day(state, date)
            if day["requests"] + 1 > self.policy["request_limit"] or day["bytes"] + MAX_RESPONSE > self.policy["byte_limit"]:
                return None
            token = uuid.uuid4().hex
            day["requests"] += 1
            day["bytes"] += MAX_RESPONSE
            owner = day["owners"].setdefault(self.owner, {"requests": 0, "bytes": 0})
            owner["requests"] += 1
            owner["bytes"] += MAX_RESPONSE
            day["inflight"][token] = {"owner": self.owner, "reserved_bytes": MAX_RESPONSE}
            return date, token

    def record_bytes(self, ticket, size):
        if not 0 <= size <= MAX_RESPONSE:
            raise ValueError("Invalid response size")
        date, token = ticket
        with self.transaction() as state:
            day = state["days"][date]
            reservation = day["inflight"][token]
            if reservation["owner"] != self.owner:
                raise ValueError("Wrong reservation owner")
            difference = size - reservation["reserved_bytes"]
            day["bytes"] += difference
            day["owners"][self.owner]["bytes"] += difference
            del day["inflight"][token]

    def export_owner(self, path):
        with self.transaction() as state:
            usage = {d: v["owners"][self.owner] for d, v in state["days"].items() if self.owner in v["owners"]}
            path = Path(path)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(usage, indent=2) + "\n")
            tmp.replace(path)
