"""Seal a fixed development opponent panel without promoting strength or fidelity gates."""
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

previous = runpy.run_path(str(ROOT / 'scripts/seal-showdown-session.py'))
source_manifest = runpy.run_path(str(ROOT / 'scripts/verify-fixed-loop.py'))['source_manifest']
read, require, write_once = previous['read'], previous['require'], previous['write_once']
PREDECESSOR = 'a8a7e497bd1e67a8e5b13e3d5ffb616b0783bb2ae9b78ae33a57f3ba6c99adaf'
REVIEW = ROOT / 'artifacts/reviews/showdown-panel'


def admission():
    path = ROOT / 'artifacts/reports/showdown-session.json'
    prior = read(path)
    require(prior['content_hash'] == PREDECESSOR == digest({k: v for k, v in prior.items() if k != 'content_hash'}), 'Session receipt')
    archive = ROOT / 'artifacts/manifests' / ('showdown-session-' + PREDECESSOR)
    require((archive / 'receipt.json').read_bytes() == path.read_bytes(), 'Session archive receipt')
    count = 0
    for kind, key in [('source', 'source_files'), ('evidence', 'evidence_files')]:
        for name, expected in prior[key].items():
            require(file_hash(archive / kind / name) == expected, 'Archived predecessor drift: ' + name); count += 1
            if name not in CURRENT:
                require(file_hash(ROOT / name) == expected, 'Live predecessor drift: ' + name); count += 1
    inherited = previous['admission']()
    return {'format': 'showdown-panel-admission-1', 'predecessor': PREDECESSOR, 'predecessor_hash_checks': count,
            'inherited_runtime': inherited['inherited_runtime'], 'source_files': source_manifest()}


def seal(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'artifacts/runs'), 'Run path')
    now = admission()
    registration, report, audit = [read(directory / name) for name in ('preregistration.json', 'report.json', 'audit.json')]
    plan = read(ROOT / 'packages/showdown-panel/plan.json')
    require(not (directory / 'failure.json').exists(), 'Failed run cannot be sealed')
    require(read(directory / 'progress.json') == {'completed': 768, 'total': 768}, 'Complete progress counters')
    require(registration['provenance'] == now and registration['plan'] == plan and registration['smoke'] is False, 'Frozen production source/plan')
    require(report['status'] == 'complete' and report['namespace'] == plan['namespace'] and report['games'] == 768
            and len(report['records']) == 768 and len(report['matrix']) == 6 and report['pairs_per_matchup'] == 64, 'Fixed full campaign')
    expected = [(i, p, m, *edge) for i, edge in enumerate(plan['matchups']) for p in range(64) for m in range(2)]
    require([(r['matchup'], r['pair'], r['mirror'], r['a'], r['b']) for r in report['records']] == expected, 'Complete matchup grid')
    for r in report['records']:
        require(r['file'] == f"game-{r['matchup']}-{r['pair']:03d}-{r['mirror']}.json.gz", 'Game filename')
        require(file_hash(directory / r['file']) == r['sha256'], 'Game evidence binding')
    require(audit['status'] == 'pass' and audit['games_replayed'] == 768 and audit['actor_inputs_and_actions_reproduced'] == report['decisions']
            and audit['pairing_seeds_policy_seats_and_intervals_verified'] and audit['final_teams_used'] == 0, 'Replay coverage')
    for key, filename in [('report_sha256', 'report.json'), ('preregistration_sha256', 'preregistration.json'), ('teams_sha256', 'teams.json')]:
        require(audit[key] == file_hash(directory / filename), 'Audit binding: ' + filename)
    v = read(REVIEW / 'verification.json'); log = (REVIEW / 'npm-test.log').read_text()
    require(v['status'] == 'passed' and v['exit_code'] == 0 and v['sources_unchanged'] and v['source_files'] == now['source_files']
            and v['test_log_sha256'] == file_hash(REVIEW / 'npm-test.log'), 'Regression source binding')
    node = re.search(r'ℹ pass (\d+)', log); suites = list(map(int, re.findall(r'Ran (\d+) tests? in', log)))
    require(node and int(node[1]) == 149 and suites == [7, 6, 15, 13, 10, 15, 13, 6, 25, 17, 16, 12, 9, 6, 13, 6]
            and log.count('\nOK\n') == len(suites), 'Regression coverage')
    for name in ('claude-review.md', 'review-dispositions.md', 'delegate-verification.md'):
        require((REVIEW / name).is_file(), 'Review missing: ' + name)
    sources = {**now['source_files'], **{str(p.relative_to(ROOT)): file_hash(p)
               for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]}}
    evidence = {'artifacts/reports/showdown-session.json': file_hash(ROOT / 'artifacts/reports/showdown-session.json')}
    for folder in (directory, REVIEW):
        evidence.update({str(p.relative_to(ROOT)): file_hash(p) for p in folder.rglob('*') if p.is_file()})
    receipt = {'format': 'showdown-panel-receipt-1', 'status': 'complete-development-panel', 'predecessor': PREDECESSOR,
               'run': str(directory.relative_to(ROOT)), 'games': 768, 'decisions_replayed': report['decisions'],
               'matrix': report['matrix'], 'tests': {'node': int(node[1]), 'python': sum(suites)},
               'source_files': sources, 'evidence_files': evidence, 'full_g2': 'open', 'full_g3': 'open',
               'training_performed': False, 'public_server_tested': False, 'final_evaluation_used': False, 'prime_used': False}
    receipt['content_hash'] = digest(receipt)
    archive = ROOT / 'artifacts/manifests' / ('showdown-panel-' + receipt['content_hash'])
    for kind, files in [('source', sources), ('evidence', evidence)]:
        for name, expected_hash in files.items():
            require(file_hash(ROOT / name) == expected_hash, 'Live drift: ' + name)
            write_once(archive / kind / name, (ROOT / name).read_bytes())
            require(file_hash(archive / kind / name) == expected_hash, 'Archive drift: ' + name)
    write_once(archive / 'receipt.json', wire(receipt) + b'\n')
    write_once(ROOT / 'artifacts/reports/showdown-panel.json', wire(receipt) + b'\n')
    print(json.dumps({'status': receipt['status'], 'hash': receipt['content_hash'], 'tests': int(node[1]) + sum(suites)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', nargs='?'); p.add_argument('--admission', action='store_true'); args = p.parse_args()
    if args.admission: print(json.dumps(admission(), sort_keys=True))
    elif args.directory: seal(args.directory)
    else: p.error('provide a run directory or --admission')
