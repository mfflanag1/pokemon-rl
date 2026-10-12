"""Bind the playable baseline to replay, network, regression and immutable predecessor evidence."""
import argparse
import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pokemon_learner.checkpoint import digest, file_hash, wire
from pokemon_switch.provenance import admission as inherited_admission, CURRENT

verification = runpy.run_path(str(ROOT / 'scripts/verify-fixed-loop.py'))
source_manifest = verification['source_manifest']
write_once = runpy.run_path(str(ROOT / 'scripts/seal-binary-policy.py'))['write_once']
PREDECESSOR = '1e484145df3e51b631e07d8192a01f1e35409e6ca3ec4df6b8aca0c51fe2bd34'
REVIEW = ROOT / 'artifacts/reviews/playable-baseline'


def read(path):
    return json.loads(path.read_bytes())


def require(value, message):
    if not value:
        raise ValueError(message)


def admission():
    prior = read(ROOT / 'artifacts/reports/m3-switch-reference.json')
    require(prior['content_hash'] == PREDECESSOR == digest({k: v for k, v in prior.items() if k != 'content_hash'}), 'M3 receipt')
    archive = ROOT / 'artifacts/manifests' / ('m3-switch-reference-' + PREDECESSOR)
    require(read(archive / 'receipt.json') == prior, 'M3 archive receipt')
    count = 0
    for kind, key in [('source', 'source_files'), ('evidence', 'evidence_files')]:
        for path, expected in prior[key].items():
            require(file_hash(archive / kind / path) == expected, 'Archived predecessor drift: ' + path)
            count += 1
            if path not in CURRENT:
                require(file_hash(ROOT / path) == expected, 'Live predecessor drift: ' + path)
                count += 1
    inherited = inherited_admission()
    return {'format': 'playable-baseline-admission-1', 'predecessor': PREDECESSOR,
            'predecessor_hash_checks': count, 'inherited_runtime': inherited['inherited_runtime'],
            'source_files': source_manifest()}


def seal(directory, network):
    directory, network = Path(directory).resolve(), Path(network).resolve()
    require(all(p.is_relative_to(ROOT / 'artifacts/runs') for p in (directory, network)), 'Run paths')
    now = admission()
    report, audit = read(directory / 'report.json'), read(directory / 'audit.json')
    run_admission = read(directory / 'admission.json')
    require(run_admission['provenance'] == now, 'Campaign source/runtime changed')
    require(report['status'] == 'complete' and report['games'] == report['pairs'] * 2, 'Complete campaign')
    require(audit['status'] == 'pass' and audit['games_replayed'] == report['games'] and
            audit['actor_inputs_and_actions_reproduced'] == report['decisions'] and
            audit['report_sha256'] == file_hash(directory / 'report.json'), 'Replay binding')
    require(0 <= report['mean_score'] <= 1, 'Invalid win score')
    for record in report['records']:
        require(re.fullmatch(r'game-\d{3}-[01]\.json\.gz', record['file']), 'Game filename')
        require(file_hash(directory / record['file']) == record['sha256'], 'Game file binding')
    net = read(network / 'report.json')
    require(net['status'] == 'complete' and net['games'] >= 2 and not net['failures'] and
            net['public_server_used'] is False and net['provenance'] == now, 'Network smoke binding')
    for game in net['records']:
        require(len(game) == 2 and game[0]['winner'] == game[1]['winner'] and game[0]['turns'] == game[1]['turns'], 'Network result agreement')
    v = read(REVIEW / 'verification.json')
    log = (REVIEW / 'npm-test.log').read_text()
    require(v['status'] == 'passed' and v['exit_code'] == 0 and v['sources_unchanged'] and
            v['source_files'] == now['source_files'] and v['test_log_sha256'] == file_hash(REVIEW / 'npm-test.log'), 'Regression source binding')
    node = re.search(r'ℹ pass (\d+)', log)
    suites = list(map(int, re.findall(r'Ran (\d+) tests? in', log)))
    require(node and int(node[1]) >= 92 and suites == [7, 6, 15, 13, 10, 15, 13, 6, 25, 17, 16, 12, 9, 6, 13, 6]
            and log.count('\nOK\n') == len(suites), 'Regression counts')
    for name in ('claude-review.md', 'review-dispositions.md', 'delegate-verification.md'):
        require((REVIEW / name).is_file(), 'Review missing: ' + name)
    sources = {**now['source_files'], **{str(p.relative_to(ROOT)): file_hash(p)
               for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]}}
    evidence = {'artifacts/reports/m3-switch-reference.json': file_hash(ROOT / 'artifacts/reports/m3-switch-reference.json')}
    for folder in (directory, network, REVIEW):
        for p in folder.rglob('*'):
            # The copied server runtime is reproducible from the pinned build. Keep config/log, not duplicate dependencies.
            rel = p.relative_to(folder)
            if folder == network and rel.parts[0] == 'server' and str(rel) not in {
                'server/server.log', 'server/config/config.js', 'server/dist/config/custom-formats.js'}:
                continue
            if p.is_file():
                evidence[str(p.relative_to(ROOT))] = file_hash(p)
    receipt = {'format': 'playable-baseline-receipt-1', 'status': 'complete-local-baseline',
               'predecessor': PREDECESSOR, 'run': str(directory.relative_to(ROOT)), 'network_run': str(network.relative_to(ROOT)),
               'report': report, 'network_games': net['games'], 'tests': {'node': int(node[1]), 'python': sum(suites)},
               'source_files': sources, 'evidence_files': evidence, 'full_g2': 'open', 'full_g3': 'open',
               'training_performed': False, 'public_server_tested': False, 'final_evaluation_used': False, 'prime_used': False}
    receipt['content_hash'] = digest(receipt)
    archive = ROOT / 'artifacts/manifests' / ('playable-baseline-' + receipt['content_hash'])
    for kind, files in [('source', sources), ('evidence', evidence)]:
        for path, expected in files.items():
            require(file_hash(ROOT / path) == expected, 'Live drift: ' + path)
            write_once(archive / kind / path, (ROOT / path).read_bytes())
            require(file_hash(archive / kind / path) == expected, 'Archive drift: ' + path)
    write_once(archive / 'receipt.json', wire(receipt) + b'\n')
    write_once(ROOT / 'artifacts/reports/playable-baseline.json', wire(receipt) + b'\n')
    print(json.dumps({'status': receipt['status'], 'hash': receipt['content_hash'], 'tests': int(node[1]) + sum(suites)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', action='store_true')
    parser.add_argument('directory', nargs='?')
    parser.add_argument('network', nargs='?')
    args = parser.parse_args()
    if args.admission:
        print(json.dumps(admission(), sort_keys=True))
    else:
        seal(args.directory, args.network)
