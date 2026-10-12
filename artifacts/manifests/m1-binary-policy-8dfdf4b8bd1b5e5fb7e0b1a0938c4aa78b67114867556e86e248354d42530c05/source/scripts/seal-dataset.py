"""Verify retained R7 evidence and publish a content-bound completion snapshot."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_reference.contract import canonical
from pokemon_reference.corpus import read_json, verify_seal, sealed, file_digest
from pokemon_reference.dataset_gate import ActorDataset, validate_collection, checked_file, check

DATA = ROOT / 'artifacts/runs/r7-dataset-v1'
LEGACY = ROOT / 'artifacts/runs/r7-legacy-v1'
REVIEW = ROOT / 'artifacts/reviews/r7'
CANARY = ROOT / 'artifacts/runs/r7-canary-v1'
LOCK_HASH = '2875345ad758189cdb31a4e9473b121f2707d4d3a43417dd5611bc8de446f741'


def write_once(path, content, *, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        check(path.read_bytes() == content, f'Refusing changed sealed file: {path}')
    else:
        # Reserve final-team files with restricted permissions from creation.
        import os
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
    if private: path.chmod(0o600)


def passed_tests():
    log = (REVIEW / 'npm-test.log').read_text()
    match = re.search(r'^ℹ tests (\d+)$', log, re.M)
    check(match is not None, 'Missing Node test summary')
    node = int(match[1])
    check(re.search(r'^ℹ fail 0$', log, re.M) is not None and
          re.search(rf'^ℹ pass {node}$', log, re.M) is not None, 'Node tests failed')
    python = [int(n) for n in re.findall(r'^Ran (\d+) tests? in ', log, re.M)]
    check(len(python) == 6 and log.count('\nOK\n') == 6 and 'FAILED (' not in log, 'Python tests incomplete/failed')
    return {'node_passed': node, 'python_passed': sum(python), 'python_suite_counts': python,
            'log_sha256': file_digest(REVIEW / 'npm-test.log')}


def verify_export(document, context):
    check(document['collection_hash'] == context['index']['content_hash'] and
          document['plan_hash'] == context['plan']['content_hash'] and
          document['exposure_registry_hash'] == context['registry_hash'], 'Dataset linkage')
    expected_inputs = {'index.json': file_digest(DATA / 'index.json'), **context['index']['files'],
                       **{r['file']: r['sha256'] for r in context['index']['records']}}
    check(document['collection_files'] == expected_inputs, 'Export input set changed')
    totals, streams = {}, {}
    for kind in ['actors', 'targets']:
        seen, counts = {}, Counter()
        for row in document[kind]:
            job = context['jobs'][row['job_id']]
            key = (row['job_id'], row['viewer'])
            check(row['viewer'] in {'p1', 'p2'} and key not in seen, 'Duplicate/unknown viewer')
            check(row['split'] == job['split'] and row['group'] == job['group'], 'Export split/group')
            expected_path = f"{kind}/{job['split']}/{job['id']}-{row['viewer']}.jsonl.gz"
            check(row['file'] == expected_path, 'Unexpected export path')
            compressed = checked_file(DATA, row['file'], row['sha256']).read_bytes()
            plain = gzip.decompress(compressed)
            check(len(compressed) == row['bytes'] and len(plain) == row['json_bytes'] and
                  len(plain.splitlines()) == row['count'], 'Export byte/row counts')
            seen[key] = row['count']; counts[row['split']] += row['count']
        check(set(seen) == {(j, p) for j in context['jobs'] for p in ['p1', 'p2']}, 'Missing actor/target stream')
        streams[kind], totals[kind] = seen, dict(counts)
    check(streams['actors'] == streams['targets'], 'Actor/target alignment')
    records = context['index']['records']
    for record in records:
        check(sum(streams['actors'][(record['job_id'], p)] for p in ['p1', 'p2']) == record['decisions'], 'Episode row total')
    check(document['validation']['actor_wire_rows'] == sum(totals['actors'].values()) and
          document['validation']['episodes'] == len(records), 'Validation totals')
    check(document['validation']['all_node_replayed'] is True and document['validation']['all_python_validated'] is True,
          'Missing completed validation')
    check(all(document['validation'][k] == 0 for k in ['cross_split_overlaps', 'final_episodes', 'final_actor_rows']), 'Final/leakage count')
    check(not (DATA / 'actors/final').exists() and not (DATA / 'targets/final').exists(), 'Final exports exist')
    check(stat.S_IMODE((DATA / 'teams/final.json').stat().st_mode) == 0o600, 'Final reservation permissions')
    # Admission checks current encoder sources and consumes a real strict actor wire per purpose.
    for purpose, split in [('training', 'train'), ('development', 'development')]:
        actor = next(ActorDataset(DATA, expected_hash=document['content_hash'], purpose=purpose).actors(split))
        check(actor.policy_inputs() == actor.value_inputs(), 'Policy/value inputs differ')
    return {'stream_counts': {k: len(v) for k, v in streams.items()}, 'rows_by_split': totals,
            'all_export_file_hashes_and_row_counts_verified': True, 'final_file_mode': '0600'}


def main():
    tests = passed_tests()
    context = validate_collection(DATA, LEGACY / 'registry.json')
    document = verify_seal(read_json(DATA / 'dataset.json'))
    export_checks = verify_export(document, context)
    benchmark = verify_seal(read_json(DATA / 'benchmark.json'))
    check(benchmark['dataset_hash'] == document['content_hash'], 'Benchmark linkage')
    legacy = verify_seal(read_json(LEGACY / 'report.json'))
    for name, expected in legacy['files'].items(): checked_file(LEGACY, name, expected)
    check(legacy['policy_hash'] == document['policy_hash'], 'Legacy policy mismatch')
    check(file_digest(ROOT / 'package-lock.json') == LOCK_HASH, 'Baseline lockfile drift')
    r5 = verify_seal(read_json(ROOT / 'artifacts/reports/r5-features.json'))
    for name, expected in r5['source_hashes'].items(): checked_file(ROOT, name, expected)
    prior_files = sorted((ROOT / 'artifacts/reports').glob('r[1-6]-*.json'))
    check(len(prior_files) == 6, 'Missing prior milestone report')
    # R1's historical report predates the integer-only artifact contract and has
    # floating-point timings. Bind its exact bytes without re-encoding that data.
    previous = {str(p.relative_to(ROOT)): {'sha256': file_digest(p),
                'content_hash': json.loads(p.read_text()).get('content_hash')} for p in prior_files}
    failure = read_json(CANARY / 'failed.json')
    failure_records = (CANARY / 'failures.jsonl').read_text().splitlines()
    check(failure['status'] == 'failed' and len(failure_records) == 2, 'Canary failure evidence')
    source_files = []
    for directory in ['packages', 'python', 'contracts', 'scripts', 'tests', 'docs', 'tools/mechanics-deps']:
        source_files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and
                            not {'node_modules', '__pycache__'} & set(p.parts))
    source_files.extend(ROOT / name for name in ['README.md', 'package.json', 'package-lock.json',
                       'showdown.pin.json', 'requirements-contract.txt', '.node-version', '.gitignore'])
    source_files = sorted(set(source_files))
    canary_receipts = [ROOT / 'artifacts/runs' / directory / name for directory, names in [
        ('r7-canary-v2', ['index.json', 'plan.json']),
        ('r7-canary-export-v1', ['dataset.json', 'benchmark.json'])] for name in names]
    evidence_files = sorted(set([p for directory in [DATA, LEGACY, REVIEW] for p in directory.rglob('*') if p.is_file()] +
                               prior_files + canary_receipts + [CANARY / name for name in ['failed.json', 'failures.jsonl']]))
    sources = {str(p.relative_to(ROOT)): file_digest(p) for p in source_files}
    evidence = {str(p.relative_to(ROOT)): file_digest(p) for p in evidence_files}
    report = sealed({'version': 'r7-completion-1', 'status': 'complete', 'milestone': 'M0-reference-data-foundation',
        'dataset_hash': document['content_hash'], 'dataset_directory': str(DATA.relative_to(ROOT)),
        'showdown_commit': context['plan']['policy']['showdown_commit'], 'policy_hash': document['policy_hash'],
        'legacy_report_hash': legacy['content_hash'], 'legacy': legacy, 'benchmark': benchmark,
        'validation': document['validation'], 'checks': {'tests': tests, 'export': export_checks,
            'r5_source_hashes_unchanged': True, 'root_lockfile_unchanged': True},
        'final_reservation': {'teams': len(context['teams']['final']), 'episodes': 0, 'actor_rows': 0,
                             'access': 'reservation-only', 'file_mode': '0600'},
        'preflight_failures': {'worker_startup_events': len(failure_records), 'completed_episodes': 0,
                              'reason': 'Inherited Node --input-type flag; fixed before accepted campaign.'},
        'review': {'implementation': 'completed; findings independently checked and addressed',
                   'identity_delegation': 'integrated after opposite-model QA and corrected local checks; original routing verdict unverified',
                   'dispositions': 'docs/R7_REVIEW.md'},
        'previous_reports': previous, 'source_files': sources, 'evidence_files': evidence,
        'next': 'M1/G1 CPU learner canaries, then an explicitly restricted simulator slice.',
        'limitations': ['Exact ordered-team holdout, not set/species/mechanical-equivalence generalization.',
            'Final seal is a workflow/API restriction and integrity commitment, not filesystem confidentiality.',
            '48 paired matchups are correlated; no policy-strength claim.',
            'No custom-engine fidelity, exhaustive mechanics coverage, trained policy, GPU or Prime run.',
            'Reference collection timing excludes team reservation and Python validation/export.']})
    report_bytes = (canonical(report) + '\n').encode()
    archive = ROOT / 'artifacts/manifests' / ('r7-' + report['content_hash'])
    for group, selected, hashes in [('source', source_files, sources), ('evidence', evidence_files, evidence)]:
        for path in selected:
            rel = str(path.relative_to(ROOT)); destination = archive / group / rel
            write_once(destination, path.read_bytes(), private=rel.endswith('/teams/final.json'))
            check(file_digest(destination) == hashes[rel], 'Archived bytes differ')
    write_once(archive / 'completion.json', report_bytes)
    report_path = ROOT / 'artifacts/reports/r7-sprint.json'
    write_once(report_path, report_bytes)
    print(canonical({'report': str(report_path), 'archive': str(archive), 'content_hash': report['content_hash'],
                     'dataset_hash': document['content_hash'], 'source_files': len(sources), 'evidence_files': len(evidence),
                     'tests': tests, 'export': export_checks}))


if __name__ == '__main__':
    main()
