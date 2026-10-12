"""CPU-only R5 feature scan of byte-certified R4 episodes and diagnostic pairs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import platform
import sys
import time
from hashlib import sha256

from .contract import CONTRACT, canonical, parse_json
from .corpus import atomic_json, file_digest, read_json, sealed, verify_seal
from .episode_campaign import load_context
from .feature_spec import CATALOG, LAYOUT, LAYOUT_HASH
from .player_history import PlayerHistory
from .feature_validation import validate_tensors
from .feature_trajectories import assemble_episode, project_player_views
from .actor_actions import ActionCodec
from .feature_checks import check_live_fixtures, synthetic_probe

ROOT = CONTRACT.parents[1]
SOURCES = [str(p.relative_to(ROOT)) for pattern in ('python/pokemon_reference/*.py',
           'contracts/player-features-v1/*.json', 'scripts/player-feature*.mjs', 'tests/features/*.*')
           for p in sorted(ROOT.glob(pattern))]
LOADED_HASHES = {p: file_digest(ROOT / p) for p in SOURCES}


def certified_context(directory):
    """Reuse R4 semantic validation only after verifying its receipts and exact bytes."""
    context = load_context(directory); directory = Path(directory)
    report = verify_seal(read_json(directory / 'validation/report.json'))
    index = context['index']
    if (report['status'] != 'complete' or report['index_hash'] != index['content_hash']
            or report['campaign_id'] != index['campaign_id'] or report['manifest_id'] != context['manifest']['manifest_id']
            or report['episodes'] != len(index['episodes']) or not report['all_records_validated_in_python']
            or not report['all_canonical_bytes_match']):
        raise ValueError('Missing R4 byte/semantic certificate')
    receipt = directory / 'validation/validated-episodes.json'
    if file_digest(receipt) != report['validated_episodes_file']['sha256']:
        raise ValueError('R4 validated receipt file differs')
    rows = read_json(receipt)
    expected = [{k: r[k] for k in ('index', 'episode_hash', 'summary')} for r in index['episodes']]
    if rows != expected: raise ValueError('R4 validation receipts differ from campaign index')
    for path, digest in report['consumer_hashes'].items():
        if file_digest(ROOT / path) != digest: raise ValueError('Frozen R4 consumer changed: ' + path)
    context['certificate_hash'] = report['content_hash']
    return context


def inspect_feature_episode(args):
    directory, record = args
    compressed = (Path(directory) / record['file']).read_bytes()
    if len(compressed) != record['bytes'] or sha256(compressed).hexdigest() != record['sha256']:
        raise ValueError('R4 compressed bytes differ')
    raw = gzip.decompress(compressed)
    if len(raw) != record['json_bytes'] or sha256(raw).hexdigest() != record['json_sha256']:
        raise ValueError('R4 certified JSON bytes differ')
    p = parse_json(raw)['episode']['payload']
    histories = {s: PlayerHistory(s) for s in ('p1', 'p2')}
    phase_options = ['revival', 'switch_continuation', 'faint_replacement', 'move']
    chosen = next((d for d in p['decisions'] if not d['accepted']), None)
    if chosen is None:
        # Rotate preference so move and switch contexts both receive paired tests.
        rotation = record['index'] % len(phase_options)
        priorities = phase_options[rotation:] + phase_options[:rotation]
        chosen = next(d for phase in priorities for d in p['decisions']
                      if p['frames'][d['frame_before']]['side_phases'][d['side']] == phase)
    chain = sha256(); counts = Counter(); chosen_actor = None
    for d in p['decisions']:
        actor = histories[d['side']].append(d['view'])
        tensors = actor.model_inputs(); validate_tensors(tensors)
        chain.update(bytes.fromhex(actor.feature_id))
        counts['decisions'] += 1
        counts['event_overflow_rows'] += sum(tensors['event_overflow'])
        counts['fact_overflow_rows'] += sum(tensors['fact_overflow'])
        g = dict(zip(LAYOUT['global_columns'], tensors['global']))
        counts['decisions_with_dropped_events'] += int(g['events_dropped'] > 0)
        counts['decisions_with_dropped_appearances'] += int(g['opponent_encounters_dropped'] > 0)
        codec = ActionCodec(d['view']['request'])
        for index, legal in enumerate(codec.mask):
            if legal:
                if codec.encode(codec.decode(index)) != index: raise ValueError('Semantic action round trip')
                counts['python_action_round_trips'] += 1
        if d['sequence'] == chosen['sequence']: chosen_actor = actor
    side = chosen['side']
    # Deliberately invalid privileged carrier: selected actor-view prefixes remain
    # intact. This tests adapter noninterference, not simulator reachability.
    carrier = {k: {'forbidden': 'changed hidden truth, RNG, commitments, labels and future'} for k in p}
    carrier['decisions'] = []
    for d in p['decisions'][:chosen['sequence'] + 1]:
        copy = {k: 'changed privileged metadata/action' for k in d}
        copy['side'] = d['side']; copy['view'] = d['view'] if d['side'] == side else {'forbidden': 'opponent private request'}
        carrier['decisions'].append(copy)
    carrier['bootstrap_views'] = None
    paired = project_player_views(carrier, side)[0][-1]
    if paired != chosen_actor or synthetic_probe(paired) != synthetic_probe(chosen_actor):
        raise ValueError('Actor/critic input depends on privileged carrier')
    return {'index': record['index'], 'episode_hash': record['episode_hash'], 'feature_chain_hash': chain.hexdigest(),
            'counts': dict(counts), 'pair': {'viewer': side, 'sequence': chosen['sequence'], 'accepted': chosen['accepted'],
            'phase': p['frames'][chosen['frame_before']]['side_phases'][side], 'feature_id': paired.feature_id}}


def inspect_saved_fixtures(directory):
    directory = Path(directory); index = verify_seal(read_json(directory / 'index.json'))
    result = []
    for record in index['records']:
        compressed = (directory / record['file']).read_bytes()
        if len(compressed) != record['bytes'] or sha256(compressed).hexdigest() != record['sha256']:
            raise ValueError('R4 fixture changed')
        bundle = parse_json(gzip.decompress(compressed))
        trajectories = assemble_episode(bundle, read_json(directory / record['manifest']))
        count, bootstraps = 0, 0
        for trajectory in trajectories:
            for actor in trajectory.actor_inputs: validate_tensors(actor.model_inputs()); count += 1
            if trajectory.bootstrap is not None: validate_tensors(trajectory.bootstrap.model_inputs()); bootstraps += 1
        if count != record['decisions']: raise ValueError('Fixture decisions differ')
        result.append({'name': record['name'], 'decisions': count, 'bootstraps': bootstraps})
    return {'index_hash': index['content_hash'], 'records': result}


def run(directory, fixtures, live, output, workers):
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError('Use 1–8 workers')
    live = Path(live)
    started = time.monotonic(); context = certified_context(directory)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        live_hash = file_digest(live); live_result = check_live_fixtures(read_json(live))
        saved = inspect_saved_fixtures(fixtures)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = ((str(directory), r) for r in context['index']['episodes'])
            for result in pool.map(inspect_feature_episode, tasks, chunksize=1):
                records.append(result)
                if len(records) % 25 == 0:
                    print(json.dumps({'feature_episodes': len(records), 'requested': context['index']['completed']}), file=sys.stderr, flush=True)
        if [r['index'] for r in records] != list(range(context['index']['completed'])):
            raise ValueError('Missing feature records')
        totals = Counter(); phases = Counter(); sides = Counter()
        for r in records: totals.update(r['counts']); phases[r['pair']['phase']] += 1; sides[r['pair']['viewer']] += 1
        if totals['decisions'] != sum(r['summary']['decisions'] for r in context['index']['episodes']):
            raise ValueError('Decision accounting')
        for path, digest in LOADED_HASHES.items():
            if file_digest(ROOT / path) != digest: raise ValueError('R5 source changed during scan: ' + path)
        after = certified_context(directory)
        if after['index']['content_hash'] != context['index']['content_hash'] or file_digest(live) != live_hash:
            raise ValueError('Input provenance changed during scan')
        atomic_json(output / 'episodes.json', records)
        report = sealed({'version': 'r5-feature-campaign-1', 'status': 'complete', 'split': 'unassigned',
            'episodes': len(records), 'totals': dict(totals), 'adapter_invariant_pairs': len(records),
            'pair_phases': dict(phases), 'pair_sides': dict(sides), 'paired_rejection_boundaries': sum(not r['pair']['accepted'] for r in records),
            'live_probes': live_result, 'live_fixture_sha256': live_hash, 'saved_r4_fixtures': saved,
            'r4_campaign_id': context['index']['campaign_id'], 'r4_index_hash': context['index']['content_hash'],
            'r4_certificate_hash': context['certificate_hash'], 'r4_manifest_id': context['manifest']['manifest_id'],
            'r4_validation_mode': 'exact certified bytes; each R5 view revalidated; fixtures fully revalidated',
            'feature_version': LAYOUT['version'], 'layout_hash': LAYOUT_HASH, 'catalog_hash': CATALOG['content_hash'],
            'source_hashes': LOADED_HASHES, 'episodes_sha256': file_digest(output / 'episodes.json'),
            'workers': workers, 'python': platform.python_version(), 'platform': platform.platform(),
            'elapsed_ms': round((time.monotonic() - started) * 1000),
            'limitations': ['Constructed live probes do not assert game reachability.',
                           'Adapter pairs deliberately alter invalid privileged carriers.',
                           'Synthetic policy/value probe has no learned weights.',
                           'Bounded appearance memory is not a complete belief state.']})
        atomic_json(output / 'report.json', report); return report
    except BaseException as error:
        atomic_json(output / 'failed.json', {'status': 'failed', 'episodes_completed': len(records), 'error': str(error)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory'); p.add_argument('--fixtures', required=True); p.add_argument('--live', required=True)
    p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    result = run(args.directory, args.fixtures, args.live, args.out, args.workers)
    print(canonical({k: result[k] for k in ('episodes', 'totals', 'adapter_invariant_pairs', 'elapsed_ms')}))


if __name__ == '__main__': main()
