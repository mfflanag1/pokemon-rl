"""R7 provenance, split, pairing and actor-admission boundary."""
import gzip
from pathlib import Path

from .contract import canonical, hash_json, parse_json, validate_artifact
from .corpus import read_json, verify_seal, file_digest
from .dataset_identity import team_key, partition
from .dataset_legacy import episode_payload, R3
from .corpus import load_context as corpus_context
from .episodes import validate_manifest
from .player_history import load_actor_input

ROOT = Path(__file__).resolve().parents[2]
POLICY = read_json(ROOT / 'contracts/dataset-v1/policy.json')
POLICY_HASH = hash_json(POLICY)


def check(ok, message):
    if not ok: raise ValueError(message)


def confined(root, name):
    check(type(name) is str and not Path(name).is_absolute(), 'Expected relative dataset path')
    result = (root / name).resolve()
    check(result.is_relative_to(root.resolve()), 'Dataset path escapes root')
    return result


def checked_file(root, name, expected):
    file = confined(root, name)
    check(file_digest(file) == expected, f'Dataset file changed: {name}')
    return file


class LeakageLedger:
    """A repeated identity may only retain the same split, group and seed role."""
    def __init__(self): self.entries = {}

    def add(self, kind, key, split, group, role=None):
        check(split in {'train', 'development', 'final'}, 'Unknown partition')
        value = (split, group, role); ident = (kind, key)
        check(ident not in self.entries or self.entries[ident] == value, f'Leakage: shared {kind} crosses split/group/role')
        self.entries[ident] = value


def reference_seed(label): return 'sodium,' + hash_json(['pokemon-reference-r1', label])


def expected_jobs(teams, namespace):
    result = []
    for split in ['train', 'development']:
        rows = teams[split]
        for i in range(0, len(rows), 2):
            pair, number = rows[i:i+2], i // 2
            group = hash_json(['r7-matchup-1', POLICY_HASH, sorted(t['team_key'] for t in pair)])
            label = f'{namespace}/{split}/pair/{number}'
            policies = [reference_seed(f'{label}/policy/{i}') for i in range(2)]
            for mirror in [False, True]:
                ordered = list(reversed(pair)) if mirror else pair
                result.append({'id': f'{split}-{number:04d}-{int(mirror)}', 'split': split, 'group': group, 'mirror': mirror,
                    'team_keys': [t['team_key'] for t in ordered],
                    'spec': {'format_id': POLICY['profile'], 'teams': [t['entry'] for t in ordered],
                        'mechanics_seed': reference_seed(label + '/mechanics'), 'policy_seeds': list(reversed(policies)) if mirror else policies,
                        'policy_id': 'upstream-random-move-0.9-no-tera-r1', 'max_submissions': 20000}})
    return result


def validate_collection(directory, exposures):
    directory = Path(directory)
    index = verify_seal(read_json(directory / 'index.json'))
    check(index['version'] == 'r7-collection-1' and index['status'] == 'complete' and index['failure_events'] == 0, 'Incomplete collection')
    expected_files = {'plan.json', 'reference-manifest.json', 'jobs.json', 'teams/train.json', 'teams/development.json', 'teams/final.json'}
    check(set(index['files']) == expected_files, 'Collection metadata file set')
    for name, expected in index['files'].items(): checked_file(directory, name, expected)
    plan = verify_seal(read_json(directory / 'plan.json'))
    check(plan['version'] == 'r7-collection-plan-1' and index['plan_hash'] == plan['content_hash'], 'Plan linkage')
    check(plan['policy_hash'] == POLICY_HASH and canonical(plan['policy']) == canonical(POLICY), 'Split policy drift')
    check(plan['final_access'] == 'reservation-only', 'Final holdout is sealed')
    for name, expected in plan['source_hashes'].items(): checked_file(ROOT, name, expected)
    registry = verify_seal(read_json(exposures))
    check(registry['version'] == 'r7-exposures-1', 'Unknown exposure registry version')
    check(registry['policy_hash'] == POLICY_HASH and registry['content_hash'] == plan['exposure_registry_hash']
          and file_digest(Path(exposures)) == plan['exposure_registry_sha256'], 'Exposure registry linkage')
    manifest = validate_manifest(read_json(directory / 'reference-manifest.json'))
    check(manifest['profile'] == 'official' and manifest['manifest_id'] == plan['episode_manifest_id'], 'Wrong reference profile')
    for name, expected in manifest['source_hashes'].items(): checked_file(ROOT, name, expected)
    contract = manifest['reference_contract']['payload']; reference = contract['reference_manifest']
    check(reference['commit'] == POLICY['showdown_commit'] and reference['format']['id'] == POLICY['profile']
          and reference['source_dirty'] is False, 'Wrong pinned reference')
    for name, expected in contract['contract_hashes'].items(): checked_file(ROOT, name, expected)
    for name, expected in reference['harness_source_hashes'].items(): checked_file(ROOT / 'packages/reference-node', name, expected)
    for name, expected in reference['source_hashes'].items(): checked_file(ROOT / 'vendor/pokemon-showdown', name, expected)
    check(hash_json(read_json(ROOT / 'contracts/battle-v1/vocabulary.json')) == contract['vocabulary_hash'], 'Vocabulary drift')
    check(canonical(read_json(ROOT / 'vendor/pokemon-showdown/.reference-build.json')) == canonical(reference['build']), 'Reference build drift')
    catalog_context = corpus_context(R3)
    check(catalog_context['index']['content_hash'] == registry['r3_index_hash'], 'Exposure catalog mismatch')
    catalog = catalog_context['catalog']
    legacy_keys, legacy_seeds = set(registry['team_keys']), set(registry['seeds'])
    teams, keys, ledger = {}, set(), LeakageLedger()
    for split in ['train', 'development', 'final']:
        reservation = verify_seal(read_json(directory / f'teams/{split}.json'))
        check(reservation['version'] == 'r7-team-reservation-1' and reservation['split'] == split and reservation['plan_hash'] == plan['content_hash'], 'Reservation linkage')
        rows = reservation['rows']; teams[split] = rows
        check(len(rows) == plan['team_counts'][split] and len(rows) >= 2 and len(rows) % 2 == 0, 'Reservation count')
        stats = index['sampling'][split]
        check(stats['accepted'] == len(rows) and stats['attempts'] == len(rows) + sum(stats['rejected'].values()), 'Rejection accounting')
        for row in rows:
            artifact = validate_artifact(row['artifact'], kind='team', manifest_id=manifest['reference_contract']['manifest_id'])
            check(canonical(episode_payload(row['entry'], catalog)) == canonical(artifact['payload']), 'Raw/resolved team mismatch')
            key = team_key(artifact['payload'], POLICY)
            check(row['team_key'] == key and row['split'] == split and partition(artifact['payload'], POLICY) == split, 'Wrong team assignment')
            check(key not in keys and key not in legacy_keys, 'Duplicate/exposed team admitted')
            check(row['entry']['seed'] not in legacy_seeds, 'Exposed seed admitted')
            keys.add(key); ledger.add('team', key, split, key)
            ledger.add('seed', row['entry']['seed'], split, key, 'team')
    jobs_doc = verify_seal(read_json(directory / 'jobs.json'))
    check(jobs_doc['version'] == 'r7-job-plan-1' and jobs_doc['plan_hash'] == plan['content_hash'], 'Job plan linkage')
    check(canonical(jobs_doc['jobs']) == canonical(expected_jobs(teams, plan['namespace'])), 'Pairing/seed/mirror plan mismatch')
    jobs = {j['id']: j for j in jobs_doc['jobs']}
    check(len(jobs) == len(jobs_doc['jobs']) == plan['expected_episodes'] == index['expected'], 'Job count')
    check(len(index['records']) == len(jobs) and {r['job_id'] for r in index['records']} == jobs.keys(), 'Missing/duplicate episodes')
    for job in jobs.values():
        check(job['split'] in {'train', 'development'}, 'Final episode forbidden')
        ledger.add('mirror-group', job['group'], job['split'], job['group'])
        for value, role in [(job['spec']['mechanics_seed'], 'mechanics'), *[(s, 'policy') for s in job['spec']['policy_seeds']]]:
            check(value not in legacy_seeds, 'Exposed battle/policy seed admitted')
            ledger.add('seed', value, job['split'], job['group'], role)
    for record in index['records']:
        job = jobs[record['job_id']]
        check(all(record[k] == job[k] for k in ['split', 'group', 'mirror']), 'Record split/group mismatch')
        check(record['file'] == f"raw/{job['id']}.json.gz" and record['replay_verified'] is True, 'Episode path/replay mismatch')
        check(record['episode_key'] not in registry['episode_keys'], 'Legacy episode admitted')
        ledger.add('episode', record['episode_key'], record['split'], job['id'])
        checked_file(directory, record['file'], record['sha256'])
    return {'index': index, 'plan': plan, 'manifest': manifest, 'jobs': jobs, 'teams': teams,
            'registry_hash': registry['content_hash'], 'ledger_entries': len(ledger.entries)}


class ActorDataset:
    """Admission requires a separately trusted dataset hash. Final access is absent.

    Only validated ActorInput objects leave this interface; full episode data and
    training targets are stored separately and never accepted as model inputs.
    """
    def __init__(self, directory, *, expected_hash, purpose):
        check(purpose in {'training', 'development'}, 'Final evaluation requires a separate release; holdout is sealed')
        self.directory, self.purpose = Path(directory), purpose
        self.document = verify_seal(read_json(self.directory / 'dataset.json'))
        check(self.document['version'] == 'r7-dataset-1' and self.document['content_hash'] == expected_hash, 'Untrusted dataset identity')
        check(self.document['policy_hash'] == POLICY_HASH and self.document['status'] == 'validated', 'Dataset admission status')
        check(self.document['final_access'] == 'reservation-only', 'Final holdout is sealed')
        for name, expected in self.document['source_hashes'].items(): checked_file(ROOT, name, expected)
        # Bind the collection and reservation files without returning their content.
        for name, expected in self.document['collection_files'].items(): checked_file(self.directory, name, expected)
        check(all(r['split'] in {'train', 'development'} for r in self.document['actors']), 'Final actor export forbidden')

    def actors(self, split):
        expected = 'train' if self.purpose == 'training' else 'development'
        check(split == expected, 'Purpose/split mismatch; final holdout is sealed')
        for row in self.document['actors']:
            if row['split'] != split: continue
            check(row['file'].startswith(f'actors/{split}/'), 'Not an actor-only file')
            file = checked_file(self.directory, row['file'], row['sha256'])
            lines = gzip.decompress(file.read_bytes()).splitlines()
            check(len(lines) == row['count'], 'Actor row count changed')
            for line in lines: yield load_actor_input(line)
