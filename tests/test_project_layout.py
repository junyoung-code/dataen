"""Integration checks for relocated commands and experiment source snapshots."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_peer_w
import train_smoke
import validation_protocol

ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_data_and_protocol_resolve_from_project_root(self):
        self.assertEqual(train_smoke.ROOT, ROOT)
        self.assertEqual(validation_protocol.ROOT, ROOT)
        self.assertEqual(run_peer_w.ROOT, ROOT)
        self.assertEqual(train_smoke.DATA, ROOT / 'data')
        manifest = json.loads((ROOT / 'configs/data-manifest.json').read_text())
        self.assertEqual(len(manifest), 12)

    def test_all_command_help_from_another_working_directory(self):
        # --help exits before loading data, starting a worker or creating a run.
        commands = ['train_smoke', 'evaluate_temporal', 'validation_protocol',
                    'run_p1', 'run_p2', 'run_p3', 'run_p4', 'run_p5', 'confirm_p5',
                    'run_catboost', 'run_xgboost', 'run_neural', 'run_peer_w']
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(ROOT / 'manage.py'), command, '--help'],
                                            cwd=directory, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_source_literals_point_to_existing_project_files(self):
        for path in (ROOT / 'src').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith(('docs/', 'configs/', 'requirements/', 'src/', 'tests/')):
                        self.assertTrue((ROOT / value).is_file(), f'{path.name}: {value}')

    def test_annual_runner_captures_nested_sources_before_training(self):
        class StopBeforeTraining(Exception):
            pass

        real_check_output = subprocess.check_output

        def git_fixture_only(command, *args, **kwargs):
            if command[0] == 'git':
                return 'fixture\n'
            return real_check_output(command, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            # Use real relocated sources; stop immediately after provenance is written.
            (temporary / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
            (temporary / 'tests').symlink_to(ROOT / 'tests', target_is_directory=True)
            output = temporary / 'outputs/check'
            with patch.object(run_peer_w, 'ROOT', temporary), \
                    patch.object(sys, 'argv', ['run_peer_w', '--output-dir', str(output)]), \
                    patch.object(run_peer_w.subprocess, 'check_output', side_effect=git_fixture_only), \
                    patch.object(run_peer_w.original, 'self_test', side_effect=StopBeforeTraining), \
                    patch.object(run_peer_w.original, 'check_data') as check_data, \
                    patch.object(run_peer_w, 'worker') as worker:
                with self.assertRaises(StopBeforeTraining):
                    run_peer_w.main()
                check_data.assert_not_called()
                worker.assert_not_called()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary['source_sha256']), 5)
            for name, digest in summary['source_sha256'].items():
                self.assertEqual(train_smoke.sha256(output / 'source' / name), digest)
                self.assertEqual((output / 'source' / name).read_bytes(), (ROOT / name).read_bytes())
