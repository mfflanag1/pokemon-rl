"""Archive completed R4 campaign evidence and emit a compact engineering report."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

from pokemon_reference.contract import CONTRACT
from pokemon_reference.corpus import atomic_json, file_digest, read_json, sealed, verify_seal
from pokemon_reference.episode_campaign import load_context

ROOT = CONTRACT.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('campaign'); parser.add_argument('--fixtures', required=True)
args = parser.parse_args()
directory, fixtures = Path(args.campaign).resolve(), Path(args.fixtures).resolve()
context = load_context(directory)
index = context['index']
validation = verify_seal(read_json(directory / 'validation/report.json'))
fixture_index = verify_seal(read_json(fixtures / 'index.json'))
fixture_validation = verify_seal(read_json(fixtures / 'validation.json'))
if (validation['index_hash'] != index['content_hash'] or validation['episodes'] != index['completed']
        or validation['campaign_id'] != index['campaign_id'] or not validation['all_records_validated_in_python']
        or fixture_validation['index_hash'] != fixture_index['content_hash']
        or fixture_validation['fixtures'] != len(fixture_index['records'])):
    raise ValueError('Mismatched validation evidence')
for row in fixture_index['records']:
    if file_digest(fixtures / row['file']) != row['sha256']:
        raise ValueError('Fixture file changed')
# Preserve source bytes as well as hashes: this workspace need not have a commit yet.
expected = {**context['manifest']['source_hashes'], **index['config']['source_hashes'],
            **context['manifest']['reference_contract']['payload']['contract_hashes'], **validation['consumer_hashes']}
expected.update({'packages/reference-node/' + p: h for p, h in
                 context['manifest']['reference_contract']['payload']['reference_manifest']['harness_source_hashes'].items()})
for path, digest in expected.items():
    if file_digest(ROOT / path) != digest:
        raise ValueError('Source changed since campaign/validation: ' + path)
archive = ROOT / 'artifacts/manifests' / ('r4-' + index['campaign_id'])
archive.mkdir(exist_ok=False)
for source, name in [(directory / 'manifest.json', 'manifest.json'), (directory / 'index.json', 'index.json'),
                     (directory / 'validation/report.json', 'validation.json'),
                     (directory / 'validation/validated-episodes.json', 'validated-episodes.json'),
                     (fixtures / 'index.json', 'fixtures.json'), (fixtures / 'validation.json', 'fixture-validation.json'),
                     (fixtures / 'constructed.manifest.json', 'constructed.manifest.json')]:
    shutil.copyfile(source, archive / name)
snapshot = sorted(set(expected) | {str(p.relative_to(ROOT)) for base in ['packages', 'python', 'contracts', 'scripts', 'tests']
                                for p in (ROOT / base).rglob('*') if p.is_file() and p.suffix in ('.mjs', '.py', '.json')}
                  | {'package.json', 'package-lock.json', 'showdown.pin.json', '.node-version', 'requirements-contract.txt'})
with tarfile.open(archive / 'source-snapshot.tar.gz', 'w:gz') as tar:
    for path in snapshot:
        tar.add(ROOT / path, arcname=path, recursive=False)
atomic_json(archive / 'source-files.json', {p: file_digest(ROOT / p) for p in snapshot})
node = __import__('json').loads(subprocess.check_output(['node', '--input-type=module', '-e',
    "import os from 'node:os'; console.log(JSON.stringify({node:process.version,cpu:os.cpus()[0].model,logical_cpus:os.cpus().length,available_parallelism:os.availableParallelism(),memory_bytes:os.totalmem(),platform:os.platform(),release:os.release(),arch:os.arch()}))"], cwd=ROOT))
compressed = sum(r['bytes'] for r in index['episodes'])
plain = sum(r['json_bytes'] for r in index['episodes'])
regression = ROOT / 'artifacts/runs/r4-regression.log'
log = regression.read_text()
if 'fail 0' not in log or log.count('\nOK') != 3 or 'FAILED' in log:
    raise ValueError('Full regression evidence is not passing')
shutil.copyfile(regression, archive / 'regression.log')
report = sealed({'version': 'r4-reference-report-1', 'created_at': datetime.now(timezone.utc).isoformat(),
                 'status': 'complete', 'campaign_id': index['campaign_id'], 'manifest_id': context['manifest']['manifest_id'],
                 'showdown_commit': context['manifest']['reference_contract']['payload']['reference_manifest']['commit'],
                 'episodes': index['completed'], 'replayed_episodes': sum(r['replay_verified'] is True for r in index['episodes']),
                 'independently_validated_episodes': validation['episodes'], 'failure_events': index['failure_events'],
                 'totals': validation['totals'], 'phase_groups': validation['phase_groups'],
                 'decision_phases': validation['decision_phases'], 'winners': validation['winners'], 'natural_examples': validation['examples'],
                 'fixtures': {'count': fixture_validation['fixtures'], 'index_hash': fixture_index['content_hash'],
                              'validation_hash': fixture_validation['content_hash'], 'by_evidence': {kind: sum(r['evidence'] == kind for r in fixture_index['records']) for kind in sorted({r['evidence'] for r in fixture_index['records']})}},
                 'storage': {'compressed_bytes': compressed, 'uncompressed_json_bytes': plain,
                             'bytes_per_episode_floor': compressed // index['completed']},
                 'timing': {'generation_and_exact_replay_ms': index['elapsed_ms'], 'generation_workers': index['workers'],
                            'independent_validation_ms': validation['elapsed_ms'], 'validation_workers': validation['workers'],
                            'measurement_note': 'Observed local wall times; independent validation overlapped final regression checks. Not isolated throughput benchmarks.'},
                 'hardware': node, 'python': platform.python_version(),
                 'checks': {'node_tests': 51, 'python_contract_tests': 7, 'python_corpus_tests': 6,
                            'python_episode_tests': 15, 'regression_log_sha256': file_digest(regression)},
                 'paths': {'campaign': str(directory.relative_to(ROOT)), 'fixtures': str(fixtures.relative_to(ROOT)),
                           'archive': str(archive.relative_to(ROOT)), 'regression_log': str(regression.relative_to(ROOT))},
                 'evidence_files': {p.name: file_digest(p) for p in sorted(archive.iterdir())},
                 'reporter_sha256': file_digest(Path(__file__)),
                 'limits': ['Exploratory unassigned data; not final evaluation', 'Random baseline with Tera disabled; no policy strength claim',
                            'Selected mechanics audit, not a complete state', 'Constructed probes do not establish natural reachability',
                            'Python checks artifacts, not independent mechanics simulation', 'Timings are CPU artifact generation/replay and validation, not RL throughput']})
atomic_json(ROOT / 'artifacts/reports/r4-reference.json', report)
print(__import__('json').dumps({'episodes': report['episodes'], 'totals': report['totals'], 'archive': str(archive),
                              'storage': report['storage'], 'timing': report['timing'], 'hardware': node, 'content_hash': report['content_hash']}))
