"""Bind local session recovery evidence to its sealed baseline and tested source."""
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

previous = runpy.run_path(str(ROOT / 'scripts/seal-playable-baseline.py'))
PREDECESSOR = '512aa4aa4b370be4ad65526c1282d9b9bba49912f73b8552397786f1094358bb'
REVIEW = ROOT / 'artifacts/reviews/showdown-session'
CASES = ['before-choice', 'lost-write', 'accepted-choice', 'next-turn', 'forced-switch', 'double-drop']
read, require, write_once = previous['read'], previous['require'], previous['write_once']


def admission():
    prior = read(ROOT / 'artifacts/reports/playable-baseline.json')
    require(prior['content_hash'] == PREDECESSOR == digest({k: v for k, v in prior.items() if k != 'content_hash'}), 'Baseline receipt')
    archive = ROOT / 'artifacts/manifests' / ('playable-baseline-' + PREDECESSOR)
    require((archive / 'receipt.json').read_bytes() == (ROOT / 'artifacts/reports/playable-baseline.json').read_bytes(), 'Baseline archive receipt')
    count = 0
    for kind, key in [('source', 'source_files'), ('evidence', 'evidence_files')]:
        for path, expected in prior[key].items():
            require(file_hash(archive / kind / path) == expected, 'Archived baseline drift: ' + path)
            count += 1
            if path not in CURRENT:
                require(file_hash(ROOT / path) == expected, 'Live baseline drift: ' + path)
                count += 1
    inherited = previous['admission']()
    return {'format': 'showdown-session-admission-1', 'predecessor': PREDECESSOR,
            'predecessor_hash_checks': count, 'inherited_runtime': inherited['inherited_runtime'],
            'source_files': previous['source_manifest']()}


def seal(directory):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(ROOT / 'artifacts/runs'), 'Run path')
    now = admission()
    require(read(directory / 'admission.json') == now, 'Run source/runtime changed')
    report, audit = read(directory / 'report.json'), read(directory / 'audit.json')
    require(report['status'] == 'complete' and report['cases'] == CASES and not report['failures']
            and report['public_server_used'] is False and report['training_performed'] is False, 'Network report')
    require([r['scenario'] for r in report['records']] == CASES, 'Scenario inventory')
    for r in report['records']:
        count = 2 if r['scenario'] == 'double-drop' else 1
        require(r['faults'] == count == len(r['restores']) == len(r['reconnects']) and r['connections'] == count + 1, 'Recovery counts')
        require(len(r['results']) == 2 and r['results'][0]['winner'] == r['results'][1]['winner']
                and r['results'][0]['turns'] == r['results'][1]['turns'], 'Terminal agreement')
    require(audit['status'] == 'pass' and audit['games'] == 6 and audit['faults'] == 7
            and audit['actor_inputs_and_actions_checked'] == sum(r['decisions'] for r in report['records'])
            and audit['report_sha256'] == file_hash(directory / 'report.json')
            and audit['trace_sha256'] == file_hash(directory / 'trace.json')
            and audit['teams_sha256'] == file_hash(directory / 'teams.json'), 'Wire audit binding')
    v = read(REVIEW / 'verification.json')
    log = (REVIEW / 'npm-test.log').read_text()
    require(v['status'] == 'passed' and v['exit_code'] == 0 and v['sources_unchanged']
            and v['source_files'] == now['source_files'] and v['test_log_sha256'] == file_hash(REVIEW / 'npm-test.log'), 'Regression source binding')
    node = re.search(r'ℹ pass (\d+)', log)
    suites = list(map(int, re.findall(r'Ran (\d+) tests? in', log)))
    require(node and int(node[1]) >= 114 and suites == [7, 6, 15, 13, 10, 15, 13, 6, 25, 17, 16, 12, 9, 6, 13, 6]
            and log.count('\nOK\n') == len(suites), 'Regression counts')
    for name in ('claude-review.md', 'review-dispositions.md', 'delegate-verification.md'):
        require((REVIEW / name).is_file(), 'Review missing: ' + name)
    sources = {**now['source_files'], **{str(p.relative_to(ROOT)): file_hash(p)
               for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]}}
    evidence = {'artifacts/reports/playable-baseline.json': file_hash(ROOT / 'artifacts/reports/playable-baseline.json')}
    for folder in (directory, REVIEW):
        for p in folder.rglob('*'):
            rel = p.relative_to(folder)
            if folder == directory and rel.parts[0] == 'server' and str(rel) not in {
                    'server/server.log', 'server/config/config.js', 'server/dist/config/custom-formats.js'}:
                continue
            if p.is_file(): evidence[str(p.relative_to(ROOT))] = file_hash(p)
    receipt = {'format': 'showdown-session-receipt-1', 'status': 'complete-local-recovery', 'predecessor': PREDECESSOR,
               'run': str(directory.relative_to(ROOT)), 'games': 6, 'injected_disconnects': 7,
               'actor_inputs_and_actions_checked': audit['actor_inputs_and_actions_checked'],
               'tests': {'node': int(node[1]), 'python': sum(suites)}, 'source_files': sources, 'evidence_files': evidence,
               'full_g2': 'open', 'full_g3': 'open', 'training_performed': False, 'public_server_tested': False,
               'final_evaluation_used': False, 'prime_used': False}
    receipt['content_hash'] = digest(receipt)
    archive = ROOT / 'artifacts/manifests' / ('showdown-session-' + receipt['content_hash'])
    for kind, files in [('source', sources), ('evidence', evidence)]:
        for path, expected in files.items():
            require(file_hash(ROOT / path) == expected, 'Live drift: ' + path)
            write_once(archive / kind / path, (ROOT / path).read_bytes())
            require(file_hash(archive / kind / path) == expected, 'Archive drift: ' + path)
    write_once(archive / 'receipt.json', wire(receipt) + b'\n')
    write_once(ROOT / 'artifacts/reports/showdown-session.json', wire(receipt) + b'\n')
    print(json.dumps({'status': receipt['status'], 'hash': receipt['content_hash'], 'tests': int(node[1]) + sum(suites)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', nargs='?'); p.add_argument('--admission', action='store_true')
    args = p.parse_args()
    if args.admission: print(json.dumps(admission(), sort_keys=True))
    elif args.directory: seal(args.directory)
    else: p.error('provide a run directory or --admission')
