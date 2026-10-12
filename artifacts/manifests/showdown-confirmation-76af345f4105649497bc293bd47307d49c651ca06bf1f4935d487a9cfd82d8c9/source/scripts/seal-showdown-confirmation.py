"""Seal a fresh tactical development confirmation and its local network qualification."""
import argparse
import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_learner.checkpoint import digest, file_hash, wire
from pokemon_switch.provenance import CURRENT

previous = runpy.run_path(str(ROOT / 'scripts/seal-showdown-panel.py'))
source_manifest = runpy.run_path(str(ROOT / 'scripts/verify-fixed-loop.py'))['source_manifest']
read, require, write_once = previous['read'], previous['require'], previous['write_once']
PREDECESSOR = 'b4a2649a09870be52d0725ea0a25994c13581da183a434a7a96c4d5482b2117d'
REVIEW = ROOT / 'artifacts/reviews/showdown-confirmation'


def admission():
    path = ROOT / 'artifacts/reports/showdown-panel.json'
    prior = read(path)
    require(prior['content_hash'] == PREDECESSOR == digest({k: v for k, v in prior.items() if k != 'content_hash'}), 'Panel receipt')
    archive = ROOT / 'artifacts/manifests' / ('showdown-panel-' + PREDECESSOR)
    require((archive / 'receipt.json').read_bytes() == path.read_bytes(), 'Panel archive receipt')
    count = 0
    for kind, key in [('source', 'source_files'), ('evidence', 'evidence_files')]:
        for name, expected in prior[key].items():
            require(file_hash(archive / kind / name) == expected, 'Archived predecessor drift: ' + name); count += 1
            if name not in CURRENT:
                require(file_hash(ROOT / name) == expected, 'Live predecessor drift: ' + name); count += 1
    inherited = previous['admission']()
    return {'format': 'showdown-confirmation-admission-1', 'predecessor': PREDECESSOR, 'predecessor_hash_checks': count,
            'inherited_runtime': inherited['inherited_runtime'], 'source_files': source_manifest(),
            'prior_pool_hashes': {name: file_hash(ROOT / name) for name in ['artifacts/runs/showdown-panel-v1/teams.json', 'artifacts/runs/showdown-panel-smoke-v1/teams.json']}}


def seal(directory, network):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'artifacts/runs'), 'Run path')
    network = Path(network).resolve()
    require(network.is_relative_to(ROOT / 'artifacts/runs'), 'Network path')
    now = admission()
    registration, report, audit = [read(directory / name) for name in ('preregistration.json', 'report.json', 'audit.json')]
    plan = read(ROOT / 'packages/showdown-confirmation/plan.json')
    require(not (directory / 'failure.json').exists(), 'Failed run cannot be sealed')
    require(read(directory / 'progress.json') == {'completed': 12000, 'total': 12000}, 'Complete progress counters')
    require(read(directory / 'audit-progress.json') == {'replayed': 12000, 'total': 12000}, 'Complete replay progress')
    require(registration['provenance'] == now and registration['plan'] == plan and registration['smoke'] is False, 'Frozen production source/plan')
    require(report['status'] == 'complete' and report['namespace'] == plan['namespace'] and report['games'] == 12000
            and len(report['records']) == 12000 and len(report['matrix']) == 6 and report['pairs_per_matchup'] == 1000, 'Fixed full campaign')
    expected = [(i, p, m, *edge) for i, edge in enumerate(plan['matchups']) for p in range(1000) for m in range(2)]
    require([(r['matchup'], r['pair'], r['mirror'], r['a'], r['b']) for r in report['records']] == expected, 'Complete matchup grid')
    for r in report['records']:
        require(r['file'] == f"game-{r['matchup']}-{r['pair']:04d}-{r['mirror']}.json.gz", 'Game filename')
        require(file_hash(directory / r['file']) == r['sha256'], 'Game evidence binding')
    require(audit['status'] == 'pass' and audit['games_replayed'] == 12000 and audit['actor_inputs_and_actions_reproduced'] == report['decisions']
            and audit['pairing_seeds_policy_seats_and_intervals_verified'] and audit['final_teams_used'] == 0
            and audit['admitted_development_teams'] == 2000, 'Replay coverage')
    for key, filename in [('report_sha256', 'report.json'), ('preregistration_sha256', 'preregistration.json'), ('teams_sha256', 'teams.json')]:
        require(audit[key] == file_hash(directory / filename), 'Audit binding: ' + filename)
    v = read(REVIEW / 'verification.json'); log = (REVIEW / 'npm-test.log').read_text()
    require(v['status'] == 'passed' and v['exit_code'] == 0 and v['sources_unchanged'] and v['source_files'] == now['source_files']
            and v['test_log_sha256'] == file_hash(REVIEW / 'npm-test.log'), 'Regression source binding')
    node = re.search(r'ℹ pass (\d+)', log); suites = list(map(int, re.findall(r'Ran (\d+) tests? in', log)))
    require(node and int(node[1]) == 191 and suites == [7, 6, 15, 13, 10, 15, 13, 6, 25, 17, 16, 12, 9, 6, 13, 6]
            and log.count('\nOK\n') == len(suites), 'Regression coverage')
    for name in ('claude-review.md', 'review-dispositions.md', 'delegate-verification.md'):
        require((REVIEW / name).is_file(), 'Review missing: ' + name)
    require(not (network / 'failure.json').exists(), 'Network failure marker')
    nr, na = read(network / 'report.json'), read(network / 'audit.json')
    require(read(network / 'admission.json') == now, 'Network source/runtime binding')
    require(nr['status'] == 'complete' and nr['policy'] == 'tactical-v1' and not nr['failures'], 'Network completion')
    require(na['status'] == 'pass' and na['games'] == 6 and na['faults'] == 7
            and na['actor_inputs_and_actions_checked'] == sum(r['decisions'] for r in nr['records']), 'Network replay coverage')
    for key, filename in [('report_sha256', 'report.json'), ('trace_sha256', 'trace.json'), ('teams_sha256', 'teams.json')]:
        require(na[key] == file_hash(network / filename), 'Network audit binding: ' + filename)
    require(report['confirmation']['status'] in ('pass', 'not-passed'), 'Confirmation decision')
    sources = {**now['source_files'], **{str(p.relative_to(ROOT)): file_hash(p)
               for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]}}
    evidence = {'artifacts/reports/showdown-panel.json': file_hash(ROOT / 'artifacts/reports/showdown-panel.json')}
    for folder in (directory, REVIEW):
        evidence.update({str(p.relative_to(ROOT)): file_hash(p) for p in folder.rglob('*') if p.is_file()})
    for name in ('admission.json', 'report.json', 'progress.json', 'trace.json', 'audit.json', 'teams.json', 'server/server.log', 'server/config/config.js', 'server/dist/config/custom-formats.js'):
        evidence[str((network / name).relative_to(ROOT))] = file_hash(network / name)
    for name, value in now['prior_pool_hashes'].items(): evidence[name] = value
    receipt = {'format': 'showdown-confirmation-receipt-1', 'status': 'complete-tactical-confirmation', 'predecessor': PREDECESSOR,
               'run': str(directory.relative_to(ROOT)), 'network_run': str(network.relative_to(ROOT)),
               'confirmation': report['confirmation'], 'network': na, 'games': 12000, 'decisions_replayed': report['decisions'],
               'matrix': report['matrix'], 'tests': {'node': int(node[1]), 'python': sum(suites)},
               'source_files': sources, 'evidence_files': evidence, 'full_g2': 'open', 'full_g3': 'open',
               'training_performed': False, 'public_server_tested': False, 'final_evaluation_used': False, 'prime_used': False}
    receipt['content_hash'] = digest(receipt)
    archive = ROOT / 'artifacts/manifests' / ('showdown-confirmation-' + receipt['content_hash'])
    for kind, files in [('source', sources), ('evidence', evidence)]:
        for name, expected_hash in files.items():
            require(file_hash(ROOT / name) == expected_hash, 'Live drift: ' + name)
            write_once(archive / kind / name, (ROOT / name).read_bytes())
            require(file_hash(archive / kind / name) == expected_hash, 'Archive drift: ' + name)
    write_once(archive / 'receipt.json', wire(receipt) + b'\n')
    write_once(ROOT / 'artifacts/reports/showdown-confirmation.json', wire(receipt) + b'\n')
    print(json.dumps({'status': receipt['status'], 'hash': receipt['content_hash'], 'tests': int(node[1]) + sum(suites)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', nargs='?'); p.add_argument('--network'); p.add_argument('--admission', action='store_true'); args = p.parse_args()
    if args.admission: print(json.dumps(admission(), sort_keys=True))
    elif args.directory and args.network: seal(args.directory, args.network)
    else: p.error('provide run directory and --network or --admission')
