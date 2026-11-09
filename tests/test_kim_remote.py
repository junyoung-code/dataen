"""Offline checks for KIM identity, coverage and shared-account limits."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import json

SPEC = importlib.util.spec_from_file_location("kim", Path(__file__).resolve().parents[1] / "scripts/collect_kim_remote.py")
kim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kim)


class KimChecks(unittest.TestCase):
    def setUp(self):
        self.params = dict(vars="tcld", group="KIMG", data="U", tmfc="2025070100", hf="24")
        self.body = ("# vars=tcld group=KIMG data=U\n# tmfc=2025070100 hf=24\n"
                     "# nx=2 ny=2\n# fname=/ARCV/g576.ft024.2025070100.nc\n0.2 0.4\n0.8 1.0\n")

    def test_complete_grid(self):
        result = kim.parse_grid(self.body, self.params)
        self.assertEqual((result["status"], result["values"]), ("verified", 4))

    def test_truncation_wrong_time_and_range_rejected(self):
        for body in (self.body.replace("0.8 1.0", "0.8"), self.body.replace("hf=24", "hf=25"),
                     self.body.replace("ft024", "ft025"), self.body.replace("0.8 1.0", "nan 1.0"),
                     self.body.replace("0.8 1.0", "0.8 10.0")):
            with self.subTest(body=body), self.assertRaises(ValueError):
                kim.parse_grid(body, self.params)

    def test_missing_is_preserved(self):
        result = kim.parse_grid(self.body.replace("0.8 1.0", "-999 1.0"), self.params)
        self.assertEqual(result["missing_values"], 1)
        self.assertEqual(kim.parse_grid("# ERROR: 파일 없음", self.params)["status"], "file_missing")

    def test_other_body_error_stops(self):
        with self.assertRaises(ValueError):
            kim.parse_grid("# ERROR: 허용되지 않은 API", self.params)

    def test_um_first_waits_locally_even_when_um_failed(self):
        for state in ("collecting", "failed"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                out = root / "outputs" / "kim"
                out.mkdir(parents=True)
                um = root / "um_status.json"
                um.write_text(json.dumps({"status": state, "completed_days": 100}))
                args = SimpleNamespace(output_dir=out, wait_for_um=um)
                with patch.object(kim, "__file__", str(root / "scripts" / "collector.py")), \
                     patch.object(kim.time, "sleep", side_effect=InterruptedError), \
                     patch.object(kim.subprocess, "Popen") as ssh:
                    with self.assertRaises(InterruptedError):
                        kim.coordinate(args)
                    ssh.assert_not_called()
                self.assertEqual(json.loads((out / "status.json").read_text())["status"], "waiting_for_um_collection")



if __name__ == "__main__":
    unittest.main()
