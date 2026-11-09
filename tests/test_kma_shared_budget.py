"""Quota races, crash reservations, migration and midnight accounting."""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kma_shared_budget import SharedBudget, MAX_RESPONSE


def contender(path, owner):
    budget = SharedBudget(path, owner, request_limit=40, headroom_requests=0, headroom_bytes=0)
    count = 0
    for _ in range(25):
        ticket = budget.reserve("2026-11-08")
        if ticket is not None:
            budget.record_bytes(ticket, 100)
            count += 1
    return count


class SharedQuotaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.json"

    def read(self):
        return json.loads(self.path.read_text())

    def test_process_race_cannot_overspend(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(contender, str(self.path), owner) for owner in ["UM", "UM", "KIM", "KIM"]]
            self.assertEqual(sum(f.result() for f in futures), 40)
        day = self.read()["days"]["2026-11-08"]
        self.assertEqual((day["requests"], day["bytes"], day["inflight"]), (40, 4000, {}))

    def test_legacy_import_and_restart(self):
        budget = SharedBudget(self.path, "UM")
        budget.seed({"UM": {"2026-11-08": {"requests": 11000, "bytes": 100}},
                     "KIM": {"2026-11-08": {"requests": 488, "bytes": 200}}})
        restarted = SharedBudget(self.path, "KIM")
        ticket = restarted.reserve("2026-11-08")
        self.assertIsNotNone(ticket)
        restarted.record_bytes(ticket, 50)
        self.assertEqual(self.read()["days"]["2026-11-08"]["requests"], 12001)
        with self.assertRaises(ValueError):
            budget.seed({})

    def test_abandoned_reservation_and_byte_limit(self):
        budget = SharedBudget(self.path, "UM", byte_limit=MAX_RESPONSE, headroom_requests=0, headroom_bytes=0)
        ticket = budget.reserve("2026-11-08")
        self.assertIsNotNone(ticket)
        restarted = SharedBudget(self.path, "KIM", byte_limit=MAX_RESPONSE, headroom_requests=0, headroom_bytes=0)
        self.assertIsNone(restarted.reserve("2026-11-08"))
        budget.record_bytes(ticket, 0)
        self.assertIsNotNone(restarted.reserve("2026-11-08"))

    def test_rollover_and_double_settlement(self):
        budget = SharedBudget(self.path, "KIM", request_limit=1, headroom_requests=0, headroom_bytes=0)
        first = budget.reserve("2026-11-08")
        self.assertIsNone(budget.reserve("2026-11-08"))
        self.assertIsNotNone(budget.reserve("2026-11-09"))
        budget.record_bytes(first, 123)
        self.assertEqual(self.read()["days"]["2026-11-08"]["bytes"], 123)
        with self.assertRaises(KeyError):
            budget.record_bytes(first, 123)

    def test_owner_and_policy_mismatch(self):
        um = SharedBudget(self.path, "UM")
        ticket = um.reserve("2026-11-08")
        with self.assertRaises(ValueError):
            SharedBudget(self.path, "KIM").record_bytes(ticket, 10)
        with self.assertRaises(ValueError):
            SharedBudget(self.path, "UM", request_limit=19000).reserve("2026-11-08")


if __name__ == "__main__":
    unittest.main()
