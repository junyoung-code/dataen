"""Run project commands or tests from the reorganized source tree."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parent
COMMANDS = ('train_smoke', 'evaluate_temporal', 'validation_protocol',
            'run_p1', 'run_p2', 'run_p3', 'run_p4', 'run_p5', 'confirm_p5',
            'run_catboost', 'run_xgboost', 'run_neural', 'run_peer_w', 'test')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     usage='%(prog)s COMMAND [arguments]')
    parser.add_argument('command', choices=COMMANDS)
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src'))
    if args.command == 'test':
        # Paths and package discovery work regardless of the caller's directory.
        sys.path.insert(0, str(ROOT))
        unittest.main(module=None, argv=[sys.argv[0], 'discover', '-s',
                      str(ROOT / 'tests'), '-t', str(ROOT), *remaining])
    else:
        script = ROOT / 'src' / (args.command + '.py')
        sys.argv = [str(script), *remaining]
        runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
