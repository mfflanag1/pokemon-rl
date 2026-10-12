"""Retain an R6 completion record after the full test log has passed."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_reference.contract import canonical, hash_json, parse_json
from pokemon_reference.mechanics import validate_bundle

RUN = ROOT / 'artifacts/runs/r6-mechanics-v4'
REVIEW = ROOT / 'artifacts/reviews/r6'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write_once(file, content):
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists() and file.read_bytes() != content:
        raise RuntimeError(f'Refusing changed sealed file: {file}')
    file.write_bytes(content)


def main():
    validation = validate_bundle(RUN, ROOT)
    log = (REVIEW / 'npm-test.log').read_text()
    node = int(re.search(r'^ℹ tests (\d+)$', log, re.M)[1])
    assert re.search(r'^ℹ fail 0$', log, re.M)
    assert re.search(rf'^ℹ pass {node}$', log, re.M)
    python = [int(n) for n in re.findall(r'^Ran (\d+) tests? in ', log, re.M)]
    assert len(python) == 5 and log.count('\nOK\n') == 5 and 'FAILED (' not in log
    manifest = parse_json((RUN / 'manifest.json').read_bytes())
    inventory_report = parse_json((RUN / 'report.json').read_bytes())
    source_files = []
    for directory in ['packages', 'python', 'contracts', 'scripts', 'tests', 'docs', 'tools/mechanics-deps']:
        for file in (ROOT / directory).rglob('*'):
            if file.is_file() and not {'node_modules', '__pycache__'} & set(file.parts):
                source_files.append(file)
    source_files.extend(ROOT / name for name in ['README.md', 'package.json', 'package-lock.json', 'showdown.pin.json', 'requirements-contract.txt', '.node-version', '.gitignore'])
    source_files = sorted(set(source_files))
    files = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    evidence_files = sorted([p for directory in [RUN, REVIEW] for p in directory.rglob('*') if p.is_file()])
    evidence = {str(p.relative_to(ROOT)): sha(p) for p in evidence_files}
    body = {'schema_version': 'r6-completion-1', 'artifact_kind': 'mechanics-completion',
            'showdown_commit': manifest['showdown_commit'], 'manifest_hash': manifest['content_hash'],
            'run': str(RUN.relative_to(ROOT)), 'counts': inventory_report['counts'],
            'checks': {'node_tests_passed': node, 'python_tests_passed': sum(python), 'independent_validation': validation},
            'review': {'implementation': 'completed-findings-verified-and-fixed', 'research_delegation': 'timed-out-no-result-integrated',
                       'dispositions': 'docs/R6_REVIEW.md'},
            'source_files': files, 'evidence_files': evidence,
            'limitations': ['Conservative envelope, not a minimal reachability proof.', 'No custom engine or engine parity.',
                            'Constructed fixed teams are not official generator witnesses.', 'No GPU training or Prime resources used.']}
    report = {**body, 'content_hash': hash_json(body)}
    report_bytes = (canonical(report) + '\n').encode()
    report_path = ROOT / 'artifacts/reports/r6-mechanics.json'
    write_once(report_path, report_bytes)
    archive = ROOT / 'artifacts/manifests' / ('r6-' + report['content_hash'])
    for group, selected in [('source', source_files), ('evidence', evidence_files)]:
        for file in selected:
            write_once(archive / group / file.relative_to(ROOT), file.read_bytes())
    write_once(archive / 'completion.json', report_bytes)
    for rel, expected in files.items():
        assert sha(archive / 'source' / rel) == expected
    for rel, expected in evidence.items():
        assert sha(archive / 'evidence' / rel) == expected
    print(json.dumps({'report': str(report_path), 'archive': str(archive), 'content_hash': report['content_hash'],
                      'source_files': len(files), 'evidence_files': len(evidence), 'tests': report['checks']}, sort_keys=True))


if __name__ == '__main__':
    main()
