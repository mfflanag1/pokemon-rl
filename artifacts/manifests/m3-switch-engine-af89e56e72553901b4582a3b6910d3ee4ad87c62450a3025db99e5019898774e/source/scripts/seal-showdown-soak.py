"""Admission and immutable receipt for the frozen tactical client's local network soak."""
import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_learner.checkpoint import digest, file_hash, wire
from pokemon_switch.provenance import CURRENT

previous = runpy.run_path(str(ROOT / 'scripts/seal-showdown-confirmation.py'))
read, require, write_once, source_manifest = [previous[k] for k in ('read', 'require', 'write_once', 'source_manifest')]
PREDECESSOR = '76af345f4105649497bc293bd47307d49c651ca06bf1f4935d487a9cfd82d8c9'
REVIEW = ROOT / 'artifacts/reviews/showdown-soak'
POOLS = ['showdown-panel-v1', 'showdown-panel-smoke-v1', 'showdown-confirmation-v1',
         'showdown-confirmation-smoke-v1', 'showdown-tactical-network-v1', 'showdown-session-v1', 'playable-baseline-network-v2']
NODE_TESTS = 237


def admission():
    path = ROOT / 'artifacts/reports/showdown-confirmation.json'
    prior = read(path)
    require(prior['content_hash'] == PREDECESSOR == digest({k: v for k, v in prior.items() if k != 'content_hash'}), 'Confirmation receipt')
    require(prior['confirmation']['status'] == 'pass', 'Candidate confirmation')
    archive = ROOT / 'artifacts/manifests' / ('showdown-confirmation-' + PREDECESSOR)
    require((archive / 'receipt.json').read_bytes() == path.read_bytes(), 'Confirmation archive receipt')
    count = 0
    for kind, key in [('source', 'source_files'), ('evidence', 'evidence_files')]:
        for name, expected in prior[key].items():
            require(file_hash(archive / kind / name) == expected, 'Archived predecessor drift: ' + name); count += 1
            if name not in CURRENT:
                require(file_hash(ROOT / name) == expected, 'Live predecessor drift: ' + name); count += 1
    inherited = previous['admission']()
    return {'format': 'showdown-soak-admission-1', 'predecessor': PREDECESSOR, 'predecessor_hash_checks': count,
            'inherited_runtime': inherited['inherited_runtime'], 'source_files': source_manifest(),
            'prior_pool_hashes': {f'artifacts/runs/{name}/teams.json': file_hash(ROOT / f'artifacts/runs/{name}/teams.json') for name in POOLS}}


def seal(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'artifacts/runs'), 'Run path')
    now = admission()
    registration, report, audit = [read(directory / name) for name in ('preregistration.json', 'report.json', 'audit.json')]
    plan = read(ROOT / 'packages/showdown-soak/plan.json'); plan.pop('smoke'); plan['games'] = plan['sessions'] * plan['games_per_session']
    require(not (directory / 'failure.json').exists(), 'Failed run cannot be sealed')
    require(registration['admission'] == now and registration['plan'] == plan and registration['smoke'] is False, 'Frozen source/plan')
    require(read(directory / 'progress.json') == {'completed': 1000, 'total': 1000}, 'Complete progress')
    require(report['status'] == 'complete' and report['plan'] == plan and report['smoke'] is False and not report['failures']
            and len(report['records']) == 1000, 'Complete campaign')
    for key in ('training_performed', 'public_server_used', 'final_evaluation_used'):
        require(report[key] is False, 'Scope: ' + key)
    coverage = {'games': 1000, 'sessions': 20, 'rematches': 980, 'faults': 60, 'tactical_connections': 80,
                'baseline_connections': 20, 'tactical_p1': 500, 'tactical_p2': 500,
                'scenarios': {'normal': 940, 'before-choice': 20, 'lost-write': 20, 'accepted-choice': 20}}
    require(report['expected'] == audit['coverage'] == coverage and audit['status'] == 'pass'
            and audit['teams'] == 1000 and audit['final_teams_used'] == 0, 'Wire audit coverage')
    require(file_hash(directory / 'server/dist/server/chat-plugins/soak-team.js') == file_hash(ROOT / 'packages/showdown-soak/team-receipt.cjs'), 'Local receipt endpoint source')
    require(0 < audit['peak_client_rss'] <= plan['max_client_rss_mib'] * 2 ** 20, 'Client memory gate')
    for i, record in enumerate(report['records']):
        require(record['index'] == i and record['file'] == f'game-{i:04d}.json.gz', 'Complete game grid')
        require(file_hash(directory / record['file']) == record['sha256'], 'Game evidence binding')
    for key, name in [('report_sha256', 'report.json'), ('preregistration_sha256', 'preregistration.json'), ('teams_sha256', 'teams.json')]:
        require(audit[key] == file_hash(directory / name), 'Audit binding: ' + name)
    replayed = json.loads(subprocess.check_output(['node', str(ROOT / 'packages/showdown-soak/audit.mjs'), str(directory), '--check'], cwd=ROOT, text=True))
    require(replayed == audit, 'Fresh wire audit differs from saved audit')
    v = read(REVIEW / 'verification.json'); log = (REVIEW / 'npm-test.log').read_text()
    require(v['status'] == 'passed' and v['exit_code'] == 0 and v['sources_unchanged'] and v['source_files'] == now['source_files']
            and v['test_log_sha256'] == file_hash(REVIEW / 'npm-test.log'), 'Regression source binding')
    node = re.search(r'ℹ pass (\d+)', log); suites = list(map(int, re.findall(r'Ran (\d+) tests? in', log)))
    require(node and int(node[1]) == NODE_TESTS and suites == [7, 6, 15, 13, 10, 15, 13, 6, 25, 17, 16, 12, 9, 6, 13, 6]
            and log.count('\nOK\n') == len(suites), 'Regression coverage')
    for name in ('claude-review.md', 'review-dispositions.md', 'delegate-verification.md'):
        require((REVIEW / name).is_file(), 'Review missing: ' + name)
    sources = {**now['source_files'], **{str(p.relative_to(ROOT)): file_hash(p) for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]}}
    evidence = {'artifacts/reports/showdown-confirmation.json': file_hash(ROOT / 'artifacts/reports/showdown-confirmation.json'), **now['prior_pool_hashes']}
    for folder in (directory, REVIEW):
        files = folder.rglob('*') if folder == REVIEW else folder.iterdir()
        evidence.update({str(p.relative_to(ROOT)): file_hash(p) for p in files if p.is_file()})
    for name in ('server/server.log', 'server/config/config.js', 'server/dist/config/custom-formats.js', 'server/dist/server/chat-plugins/soak-team.js'):
        evidence[str((directory / name).relative_to(ROOT))] = file_hash(directory / name)
    receipt = {'format': 'showdown-soak-receipt-1', 'status': 'complete-local-network-soak', 'predecessor': PREDECESSOR,
               'run': str(directory.relative_to(ROOT)), 'audit': audit, 'tests': {'node': int(node[1]), 'python': sum(suites)},
               'source_files': sources, 'evidence_files': evidence, 'full_g2': 'open', 'full_g3': 'open', 'full_g4': 'open',
               'training_performed': False, 'public_server_tested': False, 'final_evaluation_used': False, 'prime_used': False}
    receipt['content_hash'] = digest(receipt)
    archive = ROOT / 'artifacts/manifests' / ('showdown-soak-' + receipt['content_hash'])
    for kind, files in [('source', sources), ('evidence', evidence)]:
        for name, expected in files.items():
            require(file_hash(ROOT / name) == expected, 'Live drift: ' + name)
            write_once(archive / kind / name, (ROOT / name).read_bytes())
            require(file_hash(archive / kind / name) == expected, 'Archive drift: ' + name)
    write_once(archive / 'receipt.json', wire(receipt) + b'\n')
    write_once(ROOT / 'artifacts/reports/showdown-soak.json', wire(receipt) + b'\n')
    print(json.dumps({'status': receipt['status'], 'hash': receipt['content_hash'], 'tests': int(node[1]) + sum(suites)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', nargs='?'); p.add_argument('--admission', action='store_true'); args = p.parse_args()
    if args.admission: print(json.dumps(admission(), sort_keys=True))
    elif args.directory: seal(args.directory)
    else: p.error('provide run directory or --admission')
