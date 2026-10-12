"""R7 reuses exact-byte R3/R4 certificates to quarantine all prior exposures."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sqlite3
import time

from .contract import canonical, hash_json
from .corpus import load_context, read_json, verify_seal, sealed, file_digest, digest
from .feature_campaign import certified_context
from .dataset_identity import team_key, partition

ROOT = Path(__file__).resolve().parents[2]
POLICY = read_json(ROOT / 'contracts/dataset-v1/policy.json')
R3 = ROOT / 'artifacts/runs/r3-gen9-random-100k-v2'
R4 = ROOT / 'artifacts/runs/r4-reference-1000-v1'


def episode_payload(entry, catalog):
    """Resolve already-certified raw team fields through the frozen R3 catalog."""
    mons = entry['team']; resolvers = catalog['resolvers']
    ids = []
    for mon in mons:
        species = resolvers['species'][mon['species']]
        ids.append({**species, 'species': species['id'], 'ability': resolvers['ability'][mon['ability']],
                    'item': resolvers['item'][mon['item']], 'moves': mon['moves'], 'tera_type': resolvers['tera_type'][mon['teraType']]})
        del ids[-1]['id']
    return {'generator_seed': {'kind': 'showdown-sodium', 'hex': entry['seed'].removeprefix('sodium,')},
            'source_team_hash': entry['team_hash'], 'pokemon': mons, 'ids': ids}


def scan(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic_ns()
    r3 = load_context(R3)
    cert = verify_seal(read_json(R3 / 'inventory-final/report.json'))
    summary = verify_seal(read_json(ROOT / 'artifacts/reports/r3-corpus.json'))
    if (cert['content_hash'] != summary['inventory_report_hash'] or cert['index_hash'] != r3['index']['content_hash']
        or cert['teams'] != r3['index']['completed'] or cert['status'] != 'complete'
        or not cert['all_records_validated_in_python'] or not cert['all_canonical_bytes_match'] or not cert['seed_order_verified']):
        raise ValueError('R3 certificate mismatch')
    for file, expected in cert['consumer_hashes'].items():
        if file_digest(ROOT / file) != expected: raise ValueError('R3 consumer drift')
    r4 = certified_context(R4)
    db = sqlite3.connect(out / 'exposures.sqlite')
    db.executescript('CREATE TABLE teams (team_key TEXT PRIMARY KEY, split TEXT NOT NULL, occurrences INTEGER NOT NULL); CREATE TABLE seeds (seed TEXT PRIMARY KEY, split TEXT, role TEXT NOT NULL);')
    counts, duplicates, legacy_pair_counts = Counter(), 0, Counter()
    def team(payload, origin):
        nonlocal duplicates
        key, split = team_key(payload, POLICY), partition(payload, POLICY)
        prev = db.execute('SELECT split FROM teams WHERE team_key=?', (key,)).fetchone()
        if prev and prev[0] != split: raise ValueError('Exact team crossed partitions')
        duplicates += int(prev is not None)
        db.execute('INSERT INTO teams VALUES(?,?,1) ON CONFLICT(team_key) DO UPDATE SET occurrences=occurrences+1', (key, split))
        seed('sodium,' + payload['generator_seed']['hex'], split, 'team')
        counts[f'{origin}/{split}'] += 1
        return split
    def seed(value, split, role):
        previous = db.execute('SELECT split,role FROM seeds WHERE seed=?', (value,)).fetchone()
        if previous and previous != (split, role): raise ValueError('Legacy seed crosses roles/partitions')
        db.execute('INSERT OR IGNORE INTO seeds VALUES(?,?,?)', (value, split, role))
    for shard in r3['index']['chunks']:
        compressed = (R3 / shard['file']).read_bytes()
        if digest(compressed) != shard['sha256'] or len(compressed) != shard['bytes']: raise ValueError('R3 shard bytes differ')
        plain = gzip.decompress(compressed)
        if digest(plain) != shard['jsonl_sha256'] or len(plain) != shard['jsonl_bytes']: raise ValueError('R3 JSON bytes differ')
        lines = plain.splitlines()
        if len(lines) != shard['count']: raise ValueError('R3 row count')
        for line in lines: team(json.loads(line)['payload'], 'r3')
        db.commit()
    episode_keys = set()
    for record in r4['index']['episodes']:
        compressed = (R4 / record['file']).read_bytes()
        if digest(compressed) != record['sha256'] or len(compressed) != record['bytes']: raise ValueError('R4 episode bytes differ')
        plain = gzip.decompress(compressed)
        if digest(plain) != record['json_sha256'] or len(plain) != record['json_bytes']: raise ValueError('R4 episode JSON differs')
        bundle = json.loads(plain); p = bundle['episode']['payload']
        if bundle['episode']['content_hash'] != record['episode_hash']: raise ValueError('R4 certificate identity differs')
        parts = [team(episode_payload(t, r3['catalog']), 'r4') for t in p['spec']['teams']]
        legacy_pair_counts['same/' + parts[0] if parts[0] == parts[1] else 'cross-partition'] += 1
        # Prior R4 episodes remain exploratory even if their prospective assignments match.
        for value, role in [(p['spec']['mechanics_seed'], 'mechanics'), *[(s, 'policy') for s in p['spec']['policy_seeds']]]:
            seed(value, 'exploratory', role)
        episode_keys.add(p['episode_key'])
    db.commit()
    registry = sealed({'version': 'r7-exposures-1', 'policy_hash': hash_json(POLICY),
        'team_keys': [r[0] for r in db.execute('SELECT team_key FROM teams ORDER BY team_key')],
        'seeds': [r[0] for r in db.execute('SELECT seed FROM seeds ORDER BY seed')], 'episode_keys': sorted(episode_keys),
        'r3_index_hash': r3['index']['content_hash'], 'r3_certificate_hash': cert['content_hash'],
        'r4_index_hash': r4['index']['content_hash'], 'r4_certificate_hash': r4['certificate_hash']})
    (out / 'registry.json').write_text(canonical(registry) + '\n')
    db.close()
    report = sealed({'version': 'r7-exposures-report-1', 'policy_hash': hash_json(POLICY), 'counts': dict(counts),
        'legacy_episode_assignments': dict(legacy_pair_counts), 'duplicate_team_occurrences': duplicates,
        'unique_teams': len(registry['team_keys']), 'unique_seeds': len(registry['seeds']), 'episodes': len(episode_keys),
        'elapsed_ms': (time.monotonic_ns() - started) // 1_000_000,
        'files': {f: file_digest(out / f) for f in ['registry.json', 'exposures.sqlite']},
        'scope': 'All prior R3/R4 content is exposed and excluded from fresh R7 collection. Assignments are prospective; no legacy final holdout is certified.'})
    (out / 'report.json').write_text(canonical(report) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('out'); args = parser.parse_args(); scan(args.out)
