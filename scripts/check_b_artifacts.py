"""Read-only checks for published B aggregates and optional local artifact moves."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(local=False):
    contract = read(ROOT / 'AB_EVALUATION_CONTRACT.json')
    require(sha(ROOT / 'AB_EVALUATION_CONTRACT.json') ==
            'd65965390a3027d6b3267c352d7222cdfd76a9dca128e7d319bab3c77465b1ec',
            'Evaluation contract changed')
    aggregate = read(ROOT / 'docs/experiments/b/2026-11-04-b-lane-results.json')
    for hashes in [contract['fixed_source_sha256'], *aggregate['source_sha256'].values()]:
        for name, digest in hashes.items():
            require(sha(ROOT / name) == digest, f'Source changed: {name}')
    rows = aggregate['results']
    require({(r['experiment'], r['fold']) for r in rows} ==
            {(e, f) for e in ['B1', 'B2', 'B3'] for f in ['F2', 'F3']} and len(rows) == 6,
            'Expected six annual results')
    for row in rows:
        reference = contract['folds'][row['fold']]
        require(row['P4_W'] == reference['baseline_W'], 'Baseline W differs')
        require(row['evaluation_key_mask_sha256'] == reference['evaluation_key_mask_sha256'],
                'Evaluation mask differs')
        require(math.isclose(row['P4_change_percent'], (row['W'] / row['P4_W'] - 1) * 100,
                             abs_tol=1e-10), 'Change percentage differs')
    # Check the relocated documents, entry points and their relative Markdown links.
    documents = [ROOT / name for name in ['README.md', 'B_LANE_RESULT.md',
                 'CONTEXT.md', 'EXPERIMENT_PLAN.md', 'EXPERIMENTS.md', 'LANE_START.md',
                 'docs/PROJECT_STRUCTURE.md', 'docs/reports/b-process-results.md']]
    documents += list((ROOT / 'docs/experiments/b').glob('*.md'))
    for path in documents:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            require((path.parent / target.split('#')[0]).exists(),
                    f'Broken link: {path.relative_to(ROOT)} -> {target}')
    checked_files = 0
    if local:
        outputs = ROOT / 'outputs'
        mapping = read(ROOT / 'configs/b-artifact-layout.json')['legacy_aliases']
        before = read(outputs / 'reorganization/before-move-manifest.json')
        require(set(mapping) == set(before), 'Move manifest differs from layout')
        for old, new in mapping.items():
            alias, canonical = outputs / old, outputs / new
            require(alias.is_symlink() and alias.resolve() == canonical.resolve(),
                    f'Broken compatibility path: {old}')
            actual = {str(p.relative_to(canonical)) for p in canonical.rglob('*') if p.is_file()} \
                if canonical.is_dir() else {'.'}
            require(set(before[old]) <= actual, f'Missing moved files: {old}')
            for name, expected in before[old].items():
                path = canonical if name == '.' else canonical / name
                require(path.stat().st_size == expected['size'] and sha(path) == expected['sha256'],
                        f'Moved file changed: {old}/{name}')
                checked_files += 1
        for experiment, folder in [('B1', 'b1_annual'), ('B2', 'b23_tabpfn'), ('B3', 'b23_tabicl')]:
            summary = read(outputs / folder / 'summary.json')
            verification = read(outputs / folder / 'verification.json')
            require(summary['status'] == 'complete' and verification['status'] == 'passed',
                    f'Incomplete experiment: {experiment}')
            for name, digest in summary['source_sha256'].items():
                require(sha(outputs / folder / 'source' / name) == digest,
                        f'Executed source snapshot differs: {experiment}/{name}')
            for row in (r for r in rows if r['experiment'] == experiment):
                fold = row['fold']
                require(row['W'] == verification['folds'][fold]['independent_W'],
                        f'Published W differs from independent verification: {experiment}/{fold}')
                require(row['W'] == summary['folds'][fold]['metrics']['ensemble']['W'],
                        f'Published W differs from run summary: {experiment}/{fold}')
    return {'status': 'passed', 'annual_results': len(rows),
            'frozen_sources_and_document_links': 'passed',
            'local_files_unchanged': checked_files if local else 'not checked',
            'model_training_or_replay': 'not performed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true', help='Check this machine’s existing artifacts')
    args = parser.parse_args()
    print(json.dumps(check(args.local), ensure_ascii=False, indent=2))
