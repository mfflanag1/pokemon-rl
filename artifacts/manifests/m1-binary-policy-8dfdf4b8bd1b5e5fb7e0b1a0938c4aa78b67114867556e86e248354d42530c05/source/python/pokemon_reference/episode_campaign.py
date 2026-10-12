"""Validate every saved R4 campaign episode independently, with bounded workers."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import time
import sys

from .contract import CONTRACT, canonical, hash_json, parse_json
from .corpus import atomic_json, digest, file_digest, read_json, sealed, verify_seal
from .episodes import validate_episode, validate_manifest

ROOT = CONTRACT.parents[1]
SOURCE_PATHS = ['python/pokemon_reference/episode_campaign.py', 'python/pokemon_reference/episodes.py',
                'python/pokemon_reference/contract.py', 'python/pokemon_reference/corpus.py',
                'contracts/episode-v1/schema.json', 'contracts/battle-v1/schema.json',
                'contracts/battle-v1/actions.json', 'requirements-contract.txt']
LOADED_HASHES = {p: file_digest(ROOT / p) for p in SOURCE_PATHS}


def load_context(directory):
    directory = Path(directory)
    index = verify_seal(read_json(directory / 'index.json'))
    config = index['config']
    if (index['version'] != 'r4-campaign-1' or index['status'] != 'complete'
            or hash_json(config) != index['campaign_id'] or config['split'] != 'unassigned'
            or config['label_version'] != 'r4-campaign-label-1'
            or config['order'] != 'alternating-submission-order'):
        raise ValueError('Unsupported or inconsistent campaign')
    if type(config['count']) is not int or not 1 <= config['count'] <= 100000:
        raise ValueError('Invalid campaign count')
    if index['completed'] != config['count'] or len(index['episodes']) != config['count']:
        raise ValueError('Incomplete campaign')
    for i, record in enumerate(index['episodes']):
        if (record['index'] != i or record['file'] != f'episode-{i:06d}.json.gz'
                or record['label'] != f"{config['namespace']}/{i:06d}" or record['replay_verified'] is not True):
            raise ValueError('Episode accounting/order/replay flag')
    manifest = validate_manifest(read_json(directory / 'manifest.json'))
    if manifest['manifest_id'] != config['manifest_id'] or manifest['profile'] != 'official':
        raise ValueError('Campaign reference/profile differs')
    if manifest['source_hashes']['contracts/episode-v1/schema.json'] != LOADED_HASHES['contracts/episode-v1/schema.json']:
        raise ValueError('R4 consumer schema differs')
    reference = manifest['reference_contract']['payload']
    for path in ['contracts/battle-v1/schema.json', 'contracts/battle-v1/actions.json']:
        if reference['contract_hashes'][path] != LOADED_HASHES[path]:
            raise ValueError('R2 consumer definitions differ')
    failure_path = directory / 'failures.jsonl'
    failures = failure_path.read_bytes() if failure_path.exists() else b''
    if digest(failures) != index['failures_sha256'] or len(failures.splitlines()) != index['failure_events']:
        raise ValueError('Failure accounting mismatch')
    return {'directory': str(directory), 'index': index, 'manifest': manifest}


CONTEXT = None


def initialize_worker(directory):
    global CONTEXT
    CONTEXT = load_context(directory)


def inspect_episode(record, context=None):
    context = context or CONTEXT
    compressed = (Path(context['directory']) / record['file']).read_bytes()
    if len(compressed) != record['bytes'] or digest(compressed) != record['sha256']:
        raise ValueError('Compressed episode checksum: ' + record['file'])
    plain = gzip.decompress(compressed)
    if len(plain) != record['json_bytes'] or digest(plain) != record['json_sha256']:
        raise ValueError('Episode JSON checksum: ' + record['file'])
    bundle = parse_json(plain)
    if (canonical(bundle) + '\n').encode() != plain:
        raise ValueError('Noncanonical episode bytes')
    validate_episode(bundle, context['manifest'])
    e, p = bundle['episode'], bundle['episode']['payload']
    if (e['content_hash'] != record['episode_hash'] or p['normalized_output_hash'] != record['normalized_output_hash']
            or p['outcome']['status'] != 'terminal' or p['fixture'] is not None
            or p['controller'] != 'upstream-random-move-0.9-no-tera-r1'
            or p['submission_order'] != ('p2-first' if record['index'] % 2 else 'p1-first')):
        raise ValueError('Episode receipt/status/profile mismatch')
    expected_seed = lambda suffix: 'sodium,' + hash_json(['pokemon-reference-r1', record['label'] + suffix])
    if ([t['seed'] for t in p['spec']['teams']] != [expected_seed('/team/0'), expected_seed('/team/1')]
            or p['spec']['mechanics_seed'] != expected_seed('/mechanics')
            or p['spec']['policy_seeds'] != [expected_seed('/policy/0'), expected_seed('/policy/1')]):
        raise ValueError('Campaign seed namespace/index mismatch')
    phase_groups, decision_phases, examples, seen = Counter(), Counter(), {}, set()
    for i, frame in enumerate(p['frames']):
        if frame['phase'] == 'terminal' or frame['group'] in seen:
            continue
        seen.add(frame['group'])
        phase_groups[frame['phase']] += 1
        examples.setdefault(frame['phase'], i)
    pending_rejections = 0
    order = ('p2', 'p1') if record['index'] % 2 else ('p1', 'p2')
    for d in p['decisions']:
        frame = p['frames'][d['frame_before']]
        expected_side = next(s for s in order if s in frame['pending_sides'])
        if d['side'] != expected_side:
            raise ValueError('Campaign submission scheduling differs')
        decision_phases[frame['side_phases'][d['side']]] += 1
        other = 'p2' if d['side'] == 'p1' else 'p1'
        if not d['accepted'] and frame['commitments'][other] is not None:
            pending_rejections += 1
    summary = {'decisions': len(p['decisions']), 'transitions': len(p['transitions']), 'turns': p['outcome']['turn'],
               'winner': p['outcome']['winner'], 'phase_groups': dict(phase_groups), 'decision_phases': dict(decision_phases),
               'examples': examples, 'unavailable_rejections': sum(not d['accepted'] for d in p['decisions']),
               'rejections_with_opponent_choice': pending_rejections}
    if canonical(summary) != canonical(record['summary']):
        raise ValueError('Campaign summary disagrees with evidence')
    return {'index': record['index'], 'episode_hash': e['content_hash'], 'summary': summary}


def validate_campaign(directory, output, workers=4):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use 1–8 workers')
    started = time.monotonic()
    context = load_context(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        records = iter(context['index']['episodes'])
        if workers == 1:
            for record in records:
                results.append(inspect_episode(record, context))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(str(directory),)) as pool:
                pending = {pool.submit(inspect_episode, r) for r in itertools.islice(records, workers)}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        results.append(future.result())
                        record = next(records, None)
                        if record is not None:
                            pending.add(pool.submit(inspect_episode, record))
                    if len(results) % 10 == 0 or not pending:
                        print(json.dumps({'validated_episodes': len(results), 'requested': context['index']['completed']}), file=sys.stderr, flush=True)
        results.sort(key=lambda r: r['index'])
        if [r['index'] for r in results] != list(range(context['index']['completed'])):
            raise ValueError('Missing/duplicate independent episode validation')
        totals, phase_groups, decision_phases, winners, examples = Counter(), Counter(), Counter(), Counter(), {}
        for result in results:
            s = result['summary']
            for key in ['decisions', 'transitions', 'turns', 'unavailable_rejections', 'rejections_with_opponent_choice']:
                totals[key] += s[key]
            phase_groups.update(s['phase_groups']); decision_phases.update(s['decision_phases'])
            winners[s['winner'] or 'tie'] += 1
            for phase, frame in s['examples'].items():
                examples.setdefault(phase, {'episode_index': result['index'], 'frame': frame})
        if totals['decisions'] != totals['transitions']:
            raise ValueError('Transition count mismatch')
        atomic_json(output / 'validated-episodes.json', results)
        for path, expected in LOADED_HASHES.items():
            if file_digest(ROOT / path) != expected:
                raise ValueError('Consumer source changed during validation')
        report = sealed({'version': 'r4-independent-campaign-1', 'status': 'complete',
                         'campaign_id': context['index']['campaign_id'], 'index_hash': context['index']['content_hash'],
                         'manifest_id': context['manifest']['manifest_id'], 'episodes': len(results),
                         'all_records_validated_in_python': True, 'all_canonical_bytes_match': True,
                         'seed_namespaces_and_indices_verified': True, 'node_replay_receipts_verified': len(results),
                         'totals': dict(totals), 'phase_groups': dict(phase_groups), 'decision_phases': dict(decision_phases),
                         'winners': dict(winners), 'examples': examples, 'failure_events': context['index']['failure_events'],
                         'workers': workers, 'elapsed_ms': round((time.monotonic() - started) * 1000),
                         'python': platform.python_version(), 'platform': platform.platform(),
                         'libraries': {p: version(p) for p in ['jsonschema', 'referencing', 'rpds-py']},
                         'consumer_hashes': LOADED_HASHES,
                         'validated_episodes_file': {'file': 'validated-episodes.json',
                                                    'sha256': file_digest(output / 'validated-episodes.json')}})
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        atomic_json(output / 'failed.json', {'status': 'failed', 'validated_before_failure': len(results),
                                          'error': str(error), 'error_type': type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--out', required=True); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    report = validate_campaign(args.directory, args.out, args.workers)
    print(json.dumps({k: report[k] for k in ['episodes', 'totals', 'elapsed_ms', 'campaign_id']}))


if __name__ == '__main__':
    main()
