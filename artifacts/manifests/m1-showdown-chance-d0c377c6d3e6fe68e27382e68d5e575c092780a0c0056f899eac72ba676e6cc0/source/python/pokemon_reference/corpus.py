"""Independent streaming validation and disk-backed inventory of R3 team corpora."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import hashlib
import itertools
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time

from .contract import canonical, hash_json, parse_json, load_artifact, CONTRACT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


_CONSUMER_PATHS = [Path(__file__).resolve(), Path(__file__).with_name('contract.py').resolve(),
                   CONTRACT / 'schema.json', CONTRACT.parents[1] / 'requirements-contract.txt']
_LOADED_HASHES = {str(p.relative_to(CONTRACT.parents[1])): digest(p.read_bytes()) for p in _CONSUMER_PATHS}


def sealed(body):
    return {**body, 'content_hash': hash_json(body)}


def read_json(path):
    return parse_json(Path(path).read_bytes())


def verify_seal(value):
    body = {k: v for k, v in value.items() if k != 'content_hash'}
    if value.get('content_hash') != hash_json(body):
        raise ValueError('Corpus metadata hash mismatch')
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
    temporary.replace(path)


def seed_at(namespace, index):
    return hash_json(['r3-team-seeds-v1', namespace, index])


def load_context(directory):
    directory = Path(directory)
    index = verify_seal(read_json(directory / 'index.json'))
    config = index['config']
    if index['corpus_version'] != 'r3-corpus-1' or index['status'] != 'complete':
        raise ValueError('Not a completed R3 corpus')
    if (hash_json(config) != index['corpus_id'] or config['seed_version'] != 'r3-team-seeds-v1'
            or config['split'] != 'unassigned'):
        raise ValueError('Invalid corpus identity')
    if (type(config['count']) is not int or not 1 <= config['count'] <= 10000000
            or type(config['shard_size']) is not int or not 1 <= config['shard_size'] <= 10000
            or type(config['namespace']) is not str or not 1 <= len(config['namespace']) <= 200):
        raise ValueError('Invalid corpus configuration')
    next_index = 0
    for number, chunk in enumerate(index['chunks']):
        if (chunk['file'] != f'teams-{number:06d}.jsonl.gz' or chunk['start'] != next_index
                or chunk['count'] != min(config['shard_size'], config['count'] - next_index)
                or chunk['count'] <= 0):
            raise ValueError('Missing, reordered or malformed shard')
        next_index += chunk['count']
    if next_index != config['count'] or index['completed'] != config['count']:
        raise ValueError('Incomplete corpus')
    manifest = load_artifact((directory / 'manifest.json').read_bytes(), kind='manifest', manifest_id=config['battle_manifest_id'])
    catalog, vocabulary = read_json(directory / 'catalog.json'), read_json(directory / 'vocabulary.json')
    if hash_json(catalog) != config['catalog_hash'] or hash_json(vocabulary) != config['vocabulary_hash']:
        raise ValueError('Catalog/vocabulary checksum mismatch')
    reference = manifest['payload']['reference_manifest']
    if (catalog['catalog_version'] != 'r3-declarative-sets-v1' or catalog['showdown_commit'] != reference['commit']
            or catalog['source_sha256'] != reference['source_hashes'][catalog['source']]
            or vocabulary['showdown_commit'] != reference['commit']
            or manifest['payload']['vocabulary_hash'] != config['vocabulary_hash']):
        raise ValueError('Reference/catalog provenance mismatch')
    if manifest['payload']['contract_hashes']['contracts/battle-v1/schema.json'] != digest((CONTRACT / 'schema.json').read_bytes()):
        raise ValueError('Consumer schema differs from corpus schema')
    failures = (directory / 'failures.jsonl').read_bytes() if (directory / 'failures.jsonl').exists() else b''
    if digest(failures) != index['failures_sha256'] or len(failures.splitlines()) != index['failure_events']:
        raise ValueError('Failure accounting mismatch')
    return {'directory': str(directory), 'index': index, 'catalog': catalog,
            'vocabulary': {k: set(v) for k, v in vocabulary['domains'].items()}}


def set_configuration(mon, ids, *, ordered_moves=False):
    """Inventory key only; R7 must define its own benchmark grouping policy."""
    return {'species': ids['species'], 'ability': ids['ability'], 'item': ids['item'],
            'moves': ids['moves'] if ordered_moves else sorted(ids['moves']),
            'tera_type': ids['tera_type'], 'level': mon['level'], 'gender': mon['gender'],
            'evs': mon['evs'], 'ivs': mon['ivs']}


_CONTEXT = None


def initialize_worker(directory):
    global _CONTEXT
    _CONTEXT = load_context(directory)


def inspect_chunk(chunk, context=None):
    context = context or _CONTEXT
    config = context['index']['config']
    compressed = (Path(context['directory']) / chunk['file']).read_bytes()
    if len(compressed) != chunk['bytes'] or digest(compressed) != chunk['sha256']:
        raise ValueError('Compressed shard integrity failure: ' + chunk['file'])
    plain = gzip.decompress(compressed)
    if len(plain) != chunk['jsonl_bytes'] or digest(plain) != chunk['jsonl_sha256']:
        raise ValueError('JSONL shard integrity failure: ' + chunk['file'])
    lines = plain.split(b'\n')
    if lines.pop() != b'' or len(lines) != chunk['count']:
        raise ValueError('Shard row count mismatch')
    frequencies, joint_sets, teams = defaultdict(Counter), {}, defaultdict(Counter)
    move_slots = 0
    catalog, vocabulary = context['catalog'], context['vocabulary']
    for offset, line in enumerate(lines):
        index = chunk['start'] + offset
        artifact = load_artifact(line, kind='team', manifest_id=config['battle_manifest_id'])
        if canonical(artifact).encode('utf-8') != line:
            raise ValueError(f'Noncanonical bytes at row {index}')
        payload = artifact['payload']
        if payload['generator_seed'] != {'kind': 'showdown-sodium', 'hex': seed_at(config['namespace'], index)}:
            raise ValueError(f'Seed order mismatch at row {index}')
        team_configs, bases = [], []
        for slot, (mon, ids) in enumerate(zip(payload['pokemon'], payload['ids'], strict=True)):
            if mon['speciesId'] not in catalog['entries']:
                raise ValueError('Unknown generator source entry: ' + mon['speciesId'])
            resolved = catalog['resolvers']['species'].get(mon['species'])
            if resolved != {'id': ids['species'], 'base_species': ids['base_species']}:
                raise ValueError('Emitted species identifier mismatch')
            for field, raw in [('ability', 'ability'), ('item', 'item'), ('tera_type', 'teraType')]:
                if catalog['resolvers'][field].get(mon[raw]) != ids[field]:
                    raise ValueError('Resolved identifier mismatch: ' + field)
            if mon['moves'] != ids['moves'] or len(set(ids['moves'])) != len(ids['moves']):
                raise ValueError('Move IDs or uniqueness mismatch')
            # The pinned generator emits move IDs directly (not display names).
            # Membership/equality validate the mapping; only oracle replay proves generation.
            move_slots += len(mon['moves'])
            for field, domain in [('species', 'species'), ('base_species', 'species'), ('ability', 'abilities'), ('item', 'items'), ('tera_type', 'types')]:
                if ids[field] and ids[field] not in vocabulary[domain]:
                    raise ValueError(f'Unknown vocabulary ID: {domain}/{ids[field]}')
            if any(move not in vocabulary['moves'] for move in ids['moves']):
                raise ValueError('Unknown move vocabulary ID')
            source = mon['speciesId']
            templates = catalog['entries'][source]['sets']
            matching = [template['index'] for template in templates if template['role'] == mon['role']
                        and ids['ability'] in template['abilities'] and ids['tera_type'] in template['tera_types']
                        and set(ids['moves']).issubset(template['moves'])]
            frequencies['template_match_cardinality'][str(len(matching))] += 1
            for template_index in matching:
                frequencies['compatible_templates'][canonical([source, template_index])] += 1
                if len(matching) == 1:
                    frequencies['unique_template_matches'][canonical([source, template_index])] += 1
            for domain, value in [('source_species', source), ('base_species', ids['base_species']),
                                  ('forms', ids['species']), ('display_forms', mon['species']),
                                  ('abilities', ids['ability']), ('items', ids['item'] or '<none>'),
                                  ('tera_types', ids['tera_type']), ('levels', str(mon['level'])),
                                  ('roles', mon['role']), ('gender', mon['gender']),
                                  ('shiny', str(mon['shiny']).lower())]:
                frequencies[domain][value] += 1
            frequencies['moves'].update(ids['moves'])
            frequencies['move_count_per_pokemon'][str(len(mon['moves']))] += 1
            if slot == 0:
                frequencies['lead_source_species'][source] += 1
            for domain, values in [('source_moves', ids['moves']), ('source_abilities', [ids['ability']]),
                                   ('source_tera_types', [ids['tera_type']]), ('source_roles', [mon['role']]),
                                   ('source_forms', [ids['species']])]:
                for value in values:
                    frequencies[domain][canonical([source, value])] += 1
            configuration = set_configuration(mon, ids)
            key = hash_json(configuration)
            if key not in joint_sets:
                joint_sets[key] = {'configuration': configuration, 'count': 0}
            joint_sets[key]['count'] += 1
            team_configs.append(set_configuration(mon, ids, ordered_moves=True))
            bases.append(ids['base_species'])
        if len(set(bases)) != 6:
            raise ValueError('Duplicate base species in official team')
        frequencies['base_species_pairs'].update(canonical(list(pair)) for pair in itertools.combinations(sorted(bases), 2))
        teams['exact_generated'][payload['source_team_hash']] += 1
        teams['ordered_config'][hash_json(team_configs)] += 1
    return {'start': chunk['start'], 'count': chunk['count'], 'move_slots': move_slots,
            'frequencies': dict(frequencies), 'joint_sets': joint_sets, 'teams': dict(teams)}


def coverage_report(catalog, frequencies):
    """Compare observed source-conditioned marginals, without claiming reachability."""
    comparisons = {}
    expected = {'source_species': set(catalog['entries'])}
    for domain in ['moves', 'abilities', 'tera_types', 'roles']:
        expected['source_' + domain] = set()
    for source, entry in catalog['entries'].items():
        for template in entry['sets']:
            for domain in ['moves', 'abilities', 'tera_types', 'roles']:
                for value in ([template['role']] if domain == 'roles' else template[domain]):
                    expected['source_' + domain].add(canonical([source, value]))
    for domain, declared in expected.items():
        observed = {key for key, count in frequencies[domain].items() if count > 0}
        decode = (lambda key: key) if domain == 'source_species' else parse_json
        comparisons[domain] = {'declared': len(declared), 'observed_declared': len(declared & observed),
                               'unobserved_declared': [decode(key) for key in sorted(declared - observed)],
                               'observed_outside_declaration': [decode(key) for key in sorted(observed - declared)]}
    templates = {canonical([source, template['index']]) for source, entry in catalog['entries'].items() for template in entry['sets']}
    compatible = {key for key, count in frequencies['compatible_templates'].items() if count > 0}
    unique = {key for key, count in frequencies['unique_template_matches'].items() if count > 0}
    comparisons['source_templates'] = {
        'declared': len(templates), 'observed_compatible': len(templates & compatible),
        'observed_unique_match': len(templates & unique),
        'unobserved_compatible': [parse_json(key) for key in sorted(templates - compatible)],
        'match_cardinality_counts': dict(frequencies['template_match_cardinality']),
        'interpretation': 'Compatibility with emitted role/moves/ability/Tera is not a trace of the selected generator template. Overlapping templates can match the same emitted set.'}
    return {'coverage_version': 'r3-declarative-coverage-v1', 'comparisons': comparisons,
            'interpretation': 'Unobserved declared alternatives are coverage gaps, not proof of unreachability. Items, display forms, EV/IVs and other programmatic choices have no exhaustive declarative denominator here. Initial sets do not close dynamic battle mechanics.'}


def write_gzip_rows(path, rows):
    with path.open('xb') as target, gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((canonical(row) + '\n').encode('utf-8'))


def inventory(directory, output, workers=4):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use between one and eight validation workers')
    started = time.monotonic()
    context = load_context(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never mix counts from separate analysis attempts.
    connection = sqlite3.connect(output / 'counts.sqlite')
    connection.execute('CREATE TABLE sets (key TEXT PRIMARY KEY, configuration TEXT NOT NULL, count INTEGER NOT NULL)')
    connection.execute('CREATE TABLE teams (domain TEXT, key TEXT, count INTEGER NOT NULL, PRIMARY KEY(domain,key))')
    frequencies = defaultdict(Counter)
    completed, completed_starts, move_slots = 0, set(), 0

    def merge(result):
        nonlocal completed, move_slots
        if result['start'] in completed_starts:
            raise ValueError('Duplicate shard result')
        completed_starts.add(result['start'])
        move_slots += result['move_slots']
        for domain, counts in result['frequencies'].items():
            frequencies[domain].update(counts)
        with connection:
            connection.executemany('INSERT INTO sets VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET count=count+excluded.count',
                                   ((k, canonical(v['configuration']), v['count']) for k, v in result['joint_sets'].items()))
            connection.executemany('INSERT INTO teams VALUES (?,?,?) ON CONFLICT(domain,key) DO UPDATE SET count=count+excluded.count',
                                   ((domain, k, n) for domain, values in result['teams'].items() for k, n in values.items()))
        completed += result['count']
        print(json.dumps({'validated_teams': completed, 'requested': context['index']['completed']}), file=sys.stderr, flush=True)

    try:
        chunks = iter(context['index']['chunks'])
        if workers == 1:
            for chunk in chunks:
                merge(inspect_chunk(chunk, context))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(str(directory),)) as pool:
                pending = set()
                # At most one result per worker can be resident/pending in the coordinator.
                for chunk in itertools.islice(chunks, workers):
                    pending.add(pool.submit(inspect_chunk, chunk))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        merge(future.result())
                        chunk = next(chunks, None)
                        if chunk is not None:
                            pending.add(pool.submit(inspect_chunk, chunk))
        if completed != context['index']['completed'] or len(completed_starts) != len(context['index']['chunks']):
            raise ValueError('Incomplete independent validation')
        pokemon_count = completed * 6
        for domain in ['source_species', 'base_species', 'forms', 'display_forms', 'abilities', 'items',
                       'tera_types', 'levels', 'roles', 'gender', 'shiny', 'source_forms', 'source_abilities',
                       'source_tera_types', 'source_roles', 'move_count_per_pokemon', 'template_match_cardinality']:
            if sum(frequencies[domain].values()) != pokemon_count:
                raise ValueError('Inventory denominator mismatch: ' + domain)
        if (sum(frequencies['moves'].values()) != move_slots
                or sum(frequencies['source_moves'].values()) != move_slots
                or sum(int(k) * n for k, n in frequencies['move_count_per_pokemon'].items()) != move_slots):
            raise ValueError('Move-slot denominator mismatch')
        if sum(frequencies['lead_source_species'].values()) != completed:
            raise ValueError('Lead denominator mismatch')
        if sum(frequencies['base_species_pairs'].values()) != completed * 15:
            raise ValueError('Pair denominator mismatch')
        set_count, set_total = connection.execute('SELECT COUNT(*), SUM(count) FROM sets').fetchone()
        if set_total != pokemon_count:
            raise ValueError('Joint-set denominator mismatch')
        duplicate_teams = {}
        for domain in ['exact_generated', 'ordered_config']:
            unique, total = connection.execute('SELECT COUNT(*), SUM(count) FROM teams WHERE domain=?', (domain,)).fetchone()
            if total != completed:
                raise ValueError('Team denominator mismatch')
            duplicate_teams[domain] = {'unique': unique, 'duplicate_records': total - unique}
        atomic_json(output / 'frequencies.json', {k: dict(v) for k, v in frequencies.items()})
        coverage = coverage_report(context['catalog'], frequencies)
        atomic_json(output / 'coverage.json', coverage)
        write_gzip_rows(output / 'joint-sets.jsonl.gz',
                        ({'set_key': k, 'configuration': parse_json(c), 'count': n}
                         for k, c, n in connection.execute('SELECT key,configuration,count FROM sets ORDER BY key')))
        write_gzip_rows(output / 'team-counts.jsonl.gz',
                        ({'domain': domain, 'team_key': k, 'count': n}
                         for domain, k, n in connection.execute('SELECT domain,key,count FROM teams ORDER BY domain,key')))
        connection.close()
        for path in _CONSUMER_PATHS:
            if digest(path.read_bytes()) != _LOADED_HASHES[str(path.relative_to(CONTRACT.parents[1]))]:
                raise ValueError('Consumer source changed during inventory; do not publish mixed provenance')
        files = ['frequencies.json', 'coverage.json', 'joint-sets.jsonl.gz', 'team-counts.jsonl.gz', 'counts.sqlite']
        report = sealed({'report_version': 'r3-inventory-1', 'status': 'complete',
                         'corpus_id': context['index']['corpus_id'], 'index_hash': context['index']['content_hash'],
                         'teams': completed, 'pokemon': pokemon_count, 'move_slots': move_slots, 'shards': len(completed_starts),
                         'failure_events': context['index']['failure_events'], 'all_records_validated_in_python': True,
                         'all_canonical_bytes_match': True, 'seed_order_verified': True, 'unique_joint_sets': set_count,
                         'team_duplicates': duplicate_teams, 'distinct': {k: sum(n > 0 for n in v.values()) for k, v in frequencies.items()},
                         'coverage': {domain: {k: v for k, v in data.items() if isinstance(v, int)} for domain, data in coverage['comparisons'].items()},
                         'python': platform.python_version(), 'platform': platform.platform(), 'workers': workers,
                         'elapsed_ms': round((time.monotonic() - started) * 1000),
                         'consumer_hashes': _LOADED_HASHES,
                         'python_libraries': {package: version(package) for package in ['jsonschema', 'referencing', 'rpds-py']},
                         'files': {name: {'sha256': file_digest(output / name), 'bytes': (output / name).stat().st_size} for name in files}})
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as error:
        connection.close()
        atomic_json(output / 'failed.json', {'status': 'failed', 'validated_teams_before_failure': completed,
                                          'error_type': type(error).__name__, 'message': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus')
    parser.add_argument('--out', required=True, help='Fresh inventory directory')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    report = inventory(args.corpus, args.out, args.workers)
    print(json.dumps({k: report[k] for k in ['teams', 'pokemon', 'unique_joint_sets', 'elapsed_ms', 'corpus_id']}))


if __name__ == '__main__':
    main()
