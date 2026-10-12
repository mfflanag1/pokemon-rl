"""Validate fresh R7 episodes, export separate actor/target streams, measure costs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import platform
import shutil
import time

from .contract import canonical, parse_json
from .corpus import read_json, sealed, file_digest, digest
from .dataset_gate import validate_collection, checked_file, check, POLICY_HASH, ROOT
from .feature_trajectories import assemble_episode
from .player_history import load_actor_input


def write_json(path, value):
    with path.open('x') as f: f.write(canonical(value) + '\n')


def write_rows(directory, file, rows):
    plain = ('\n'.join(rows) + '\n').encode() if rows else b''
    compressed = gzip.compress(plain, compresslevel=6, mtime=0)
    with (directory / file).open('xb') as stream: stream.write(compressed)
    return {'file': file, 'sha256': digest(compressed), 'bytes': len(compressed), 'json_bytes': len(plain), 'count': len(rows)}


def export_episode(args):
    directory, record, job, manifest = args
    directory = Path(directory); started = time.monotonic_ns()
    compressed = checked_file(directory, record['file'], record['sha256']).read_bytes()
    check(len(compressed) == record['bytes'], 'Episode byte count')
    plain = gzip.decompress(compressed)
    check(len(plain) == record['json_bytes'] and digest(plain) == record['json_sha256'], 'Episode JSON bytes')
    bundle = parse_json(plain)
    check((canonical(bundle) + '\n').encode() == plain, 'Noncanonical episode')
    check(bundle['episode']['content_hash'] == record['episode_hash'], 'Episode identity differs')
    p = bundle['episode']['payload']
    check(canonical(p['spec']) == canonical(job['spec']), 'Episode does not implement its predeclared job')
    check(p['episode_key'] == record['episode_key'] and p['split'] == 'unassigned' and p['fixture'] is None, 'Unexpected R4 envelope')
    check(p['outcome']['status'] == 'terminal', 'Nonterminal benchmark record')
    check(p['submission_order'] == ('p2-first' if job['mirror'] else 'p1-first'), 'Wrong submission order')
    check(record['decisions'] == len(p['decisions']) and record['turns'] == p['outcome']['turn'] and
          record['unavailable_rejections'] == sum(not d['accepted'] for d in p['decisions']), 'Record summary mismatch')
    trajectories = assemble_episode(bundle, manifest)
    validated = time.monotonic_ns()
    actors, targets = [], []
    for trajectory in trajectories:
        check(trajectory.bootstrap is None, 'Terminal episode has bootstrap')
        actor_rows = []
        for actor in trajectory.actor_inputs:
            # Independently enforce the public actor wire before export.
            verified = load_actor_input(actor._wire)
            actor_rows.append(verified._wire)
        suffix = f"{job['id']}-{trajectory.viewer}.jsonl.gz"
        meta = {'job_id': job['id'], 'split': job['split'], 'group': job['group'], 'viewer': trajectory.viewer}
        actors.append({**meta, **write_rows(directory, f"actors/{job['split']}/{suffix}", actor_rows)})
        targets.append({**meta, **write_rows(directory, f"targets/{job['split']}/{suffix}", [canonical(asdict(t)) for t in trajectory.targets])})
    return {'job_id': job['id'], 'actors': actors, 'targets': targets,
            'validation_projection_ms': (validated - started) // 1_000_000,
            'actor_wire_storage_ms': (time.monotonic_ns() - validated) // 1_000_000}


def quantiles(values):
    values = sorted(values)
    return {str(p): values[max(0, (len(values) * p + 99) // 100 - 1)] for p in [50, 95, 99, 100]}


def source_hashes():
    files = sorted((ROOT / 'python/pokemon_reference').glob('*.py'))
    files += sorted((ROOT / 'contracts/player-features-v1').glob('*.json'))
    files += [ROOT / 'contracts/dataset-v1/policy.json', ROOT / 'requirements-contract.txt']
    return {str(p.relative_to(ROOT)): file_digest(p) for p in files}


def build(collection, exposures, out, workers=4):
    check(type(workers) is int and 1 <= workers <= 8, 'Invalid workers')
    started = time.monotonic_ns(); collection, out = Path(collection), Path(out)
    context = validate_collection(collection, exposures)
    sources = source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {'index.json': file_digest(collection / 'index.json'), **context['index']['files'],
              **{r['file']: r['sha256'] for r in context['index']['records']}}
    for name, expected in inputs.items():
        destination = out / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_file(collection, name, expected), destination)
        if name == 'teams/final.json': destination.chmod(0o600)
        checked_file(out, name, expected)
    for kind in ['actors', 'targets']:
        for split in ['train', 'development']: (out / kind / split).mkdir(parents=True)
    copied = time.monotonic_ns()
    work = [(str(out), r, context['jobs'][r['job_id']], context['manifest']) for r in context['index']['records']]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(export_episode, work):
                results.append(result)
                if len(results) % 16 == 0: print(json.dumps({'validated_exported': len(results), 'requested': len(work)}), flush=True)
        check(sources == source_hashes(), 'Exporter source changed')
        records = context['index']['records']; elapsed = (time.monotonic_ns() - copied) // 1_000_000
        document = sealed({'version': 'r7-dataset-1', 'status': 'validated', 'policy_hash': POLICY_HASH,
            'plan_hash': context['plan']['content_hash'], 'collection_hash': context['index']['content_hash'],
            'exposure_registry_hash': context['registry_hash'], 'final_access': 'reservation-only',
            'collection_files': inputs, 'source_hashes': sources,
            'actors': [a for r in results for a in r['actors']], 'targets': [a for r in results for a in r['targets']],
            'validation': {'episodes': len(results), 'all_node_replayed': True, 'all_python_validated': True,
                'actor_wire_rows': sum(a['count'] for r in results for a in r['actors']),
                'ledger_entries': context['ledger_entries'], 'cross_split_overlaps': 0, 'final_episodes': 0, 'final_actor_rows': 0}})
        write_json(out / 'dataset.json', document)
        ms = context['index']['collection_wall_ms']
        report = sealed({'version': 'r7-benchmark-1', 'dataset_hash': document['content_hash'],
            'episodes': len(records), 'paired_matchups': len(records) // 2,
            'decisions': sum(r['decisions'] for r in records), 'turns': sum(r['turns'] for r in records),
            'failure_events': context['index']['failure_events'], 'unavailable_rejections': sum(r['unavailable_rejections'] for r in records),
            'collection_wall_ms_including_replay': ms, 'collection_workers': context['index']['workers'],
            'episode_rate_per_second_milli': len(records) * 1_000_000 // ms,
            'decision_rate_per_second_milli': sum(r['decisions'] for r in records) * 1_000_000 // ms,
            'generation_latency_ms_nearest_rank': quantiles([r['generation_ms'] for r in records]),
            'replay_latency_ms_nearest_rank': quantiles([r['replay_ms'] for r in records]),
            'storage': {'raw_compressed_bytes': sum(r['bytes'] for r in records), 'raw_json_bytes': sum(r['json_bytes'] for r in records),
                        'actor_compressed_bytes': sum(r['bytes'] for r in document['actors']), 'actor_json_bytes': sum(r['json_bytes'] for r in document['actors']),
                        'target_compressed_bytes': sum(r['bytes'] for r in document['targets'])},
            'validation_export_wall_ms': elapsed, 'validation_workers': workers,
            'input_audit_copy_ms': (copied - started) // 1_000_000, 'python': platform.python_version(),
            'hardware': context['index']['hardware'], 'sampling': context['index']['sampling'],
            'scope': 'Pinned random baseline with Tera disabled; both seats per pair. Includes exact reference replay, serialization and storage. No learner, accelerator, strength inference or final outcomes.'})
        write_json(out / 'benchmark.json', report)
        print(json.dumps({'dataset_hash': document['content_hash'], 'episodes': len(records), 'actor_rows': document['validation']['actor_wire_rows'], 'elapsed_ms': elapsed}), flush=True)
        return document
    except BaseException as error:
        write_json(out / 'failed-export.json', {'status': 'failed', 'message': str(error), 'completed': len(results)})
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('collection'); p.add_argument('exposures'); p.add_argument('out'); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args(); build(a.collection, a.exposures, a.out, a.workers)
