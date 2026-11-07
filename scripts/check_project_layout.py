"""Check project paths and, optionally, the local file-migration evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(local=False):
    plan = read(ROOT / 'configs/project-layout.json')
    for old, new in plan['source_moves'].items():
        require((ROOT / new).is_file(), f'Missing relocated file: {new}')
        require(not (ROOT / old).exists(), f'Unexpected old source path: {old}')
    documents = list(ROOT.glob('*.md')) + list((ROOT / 'docs').rglob('*.md'))
    documents.append(ROOT / 'workspaces/README.md')
    links = 0
    for path in documents:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            require((path.parent / target.split('#')[0]).exists(),
                    f'Broken link: {path.relative_to(ROOT)} -> {target}')
            links += 1
    for path in (ROOT / 'requirements').glob('*.txt'):
        for line in path.read_text().splitlines():
            if line.startswith('-r '):
                require((path.parent / line[3:].strip()).is_file(), f'Broken include: {path.name}')
    count = 0
    if local:
        audit = ROOT / 'outputs/maintenance/structure_20261105'
        before = read(audit / 'moved-artifacts-before.json')
        moves = plan['artifact_moves_with_compatibility_links'] | plan['transfer_moves']
        require(set(before) == set(moves), 'Artifact move manifest differs')
        for old, new in moves.items():
            destination = ROOT / new
            if old in plan['artifact_moves_with_compatibility_links']:
                require((ROOT / old).is_symlink() and (ROOT / old).resolve() == destination.resolve(),
                        f'Broken old output path: {old}')
            for name, expected in before[old].items():
                path = destination if name == '.' else destination / name
                require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                        f'Changed moved file: {new}/{name}')
                count += 1
        source_before = read(audit / 'source-before.json')
        for old, digest in source_before.items():
            require(sha(audit / 'before' / old) == digest, f'Changed original backup: {old}')
            # Numeric reports, manifests, dependencies and existing tests moved without edits.
            if old.endswith(('.json', '.txt')) or old.startswith('test_'):
                require(sha(ROOT / plan['source_moves'].get(old, old)) == digest,
                        f'Changed original numeric report/config/test: {old}')
        for lane, files in read(audit / 'lane-root-hashes-before.json').items():
            base = ROOT / 'outputs/ab_20261104' / lane
            require((ROOT / 'workspaces' / lane).resolve() == base.resolve(), 'Broken workspace link')
            for name, digest in files.items():
                require(sha(base / name) == digest, f'Lane file changed: {lane}/{name}')
        require(subprocess.check_output(['git', 'ls-files', '--stage'], cwd=ROOT) ==
                (audit / 'git-index-before.txt').read_bytes(), 'Preexisting Git index changed')
    return {'status': 'passed', 'relocated_project_files': len(plan['source_moves']),
            'document_links': links, 'moved_local_files_unchanged': count if local else 'not checked',
            'preexisting_sources_backed_up_and_lanes_preserved': local}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.local), indent=2))
