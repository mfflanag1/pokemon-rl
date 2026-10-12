"""R6 artifact verification and a fail-closed custom-engine boundary.

This module consumes Node's source inventory. It never parses TypeScript, derives
actor features from diagnostics, or treats reference replay as engine support.
"""
from pathlib import Path
import argparse
import hashlib
import json

from .contract import hash_json, parse_json

DOMAINS = {'moves', 'abilities', 'items', 'forms', 'conditions', 'core'}
FAMILIES = {'move-resolution', 'ability-events', 'item-events', 'species-state', 'conditions',
            'callback-dispatch', 'move-calling', 'identity-copy', 'tera', 'battle-core'}
KINDS = {'move-execution', 'active-move-construction', 'transform-copy', 'form-change',
         'ability-change', 'item-transfer', 'item-consumption', 'condition-install',
         'event-dispatch', 'callback-invocation', 'computed-call', 'dex-resolution', 'identity-state-write',
         'source-owned-call', 'source-owned-construction'}
BOUNDS = {
    'move-execution': ('moves', 'move-calling'), 'active-move-construction': ('moves', 'move-calling'),
    'transform-copy': ('forms+moves+abilities', 'identity-copy'), 'form-change': ('forms', 'species-state'),
    'ability-change': ('abilities', 'identity-copy'), 'item-transfer': ('items', 'item-events'),
    'item-consumption': ('items', 'item-events'),
    'condition-install': ('conditions+moves+abilities+items+core', 'conditions'),
    **{k: ('pinned-source-callbacks', 'callback-dispatch') for k in ['event-dispatch', 'callback-invocation', 'computed-call']},
    'dex-resolution': ('moves+abilities+items+forms+conditions', 'callback-dispatch'),
    'identity-state-write': ('moves+abilities+items+forms', 'identity-copy'),
    **{k: ('pinned-runtime', 'battle-core') for k in ['source-owned-call', 'source-owned-construction']},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sealed(document, version):
    require(document.get('schema_version') == version, 'Unknown mechanics artifact version')
    require(document.get('content_hash') == hash_json({k: v for k, v in document.items() if k != 'content_hash'}),
            'Mechanics content hash mismatch')


def confined(root, relative):
    require(type(relative) is str and not Path(relative).is_absolute(), 'Expected relative source path')
    full = (root / relative).resolve()
    require(full.is_relative_to(root.resolve()), 'Source path escapes root')
    return full


def validate_manifest(manifest, root=None):
    sealed(manifest, 'r6-mechanics-1')
    require(manifest['artifact_kind'] == 'mechanics-manifest', 'Wrong artifact kind')
    require(manifest['profile'] == 'gen9randombattle', 'Unknown mechanics profile')
    require(manifest['scope']['kind'] == 'conservative-pinned-base-dex-envelope', 'Unsupported analysis scope')
    require(manifest['scope']['minimal_reachability_proven'] is False, 'Invalid reachability claim')
    require(manifest['scope']['custom_engine_implemented'] is False, 'R6 does not implement an engine')
    require(manifest['unclassified_pathways'] == [], 'Unclassified pathways block the profile')
    sources = manifest['sources']
    contents = {}
    if root is not None:
        root = Path(root)
        pin = parse_json((root / 'showdown.pin.json').read_bytes())
        require(manifest['showdown_commit'] == pin['commit'], 'Showdown pin mismatch')
        for file, expected in sources.items():
            contents[file] = confined(root / 'vendor/pokemon-showdown', file).read_bytes()
            require(digest(contents[file]) == expected, f'Source hash mismatch: {file}')
        for spec in manifest['inputs'].values():
            require(digest(confined(root, spec['file']).read_bytes()) == spec['sha256'], 'Input hash mismatch')
        require(digest((root / 'tools/mechanics-deps/package-lock.json').read_bytes()) == manifest['parser']['lock_sha256'], 'Parser lock mismatch')

    def source_ref(ref):
        require(ref['file'] in sources and ref['file_sha256'] == sources[ref['file']], 'Source reference mismatch')
        require(type(ref['start_byte']) is int and type(ref['end_byte']) is int and 0 <= ref['start_byte'] < ref['end_byte'], 'Bad source span')
        require(1 <= ref['start_line'] <= ref['end_line'], 'Bad source lines')
        if contents:
            data = contents[ref['file']]
            require(ref['end_byte'] <= len(data), 'Source span outside file')
            require(digest(data[ref['start_byte']:ref['end_byte']]) == ref['fragment_sha256'], 'Source fragment mismatch')
            require(data[:ref['start_byte']].count(b'\n') + 1 == ref['start_line'], 'Source start line mismatch')
            require(data[:ref['end_byte']].count(b'\n') + 1 == ref['end_line'], 'Source end line mismatch')

    entries = {}
    for entry in manifest['entries']:
        key = entry['key']
        require(key not in entries, 'Duplicate mechanics entry')
        require(entry['domain'] in DOMAINS and key == f"{entry['domain']}:{entry['id']}", 'Invalid mechanics identity')
        require(entry['handler_id'] == key, 'Handler identity mismatch')
        require(entry['support'] == 'unsupported', 'R6 cannot certify custom-engine support')
        require(entry['ticket'].startswith('ENG-') and entry['fixture_families'] and
                set(entry['fixture_families']) <= FAMILIES, 'Missing ticket or fixture family')
        require(type(entry['observed_count']) is int and entry['observed_count'] >= 0, 'Invalid observation count')
        require(entry['reachability'] in {'witnessed-initial', 'conservative-envelope'}, 'Unsupported reachability classification')
        require((entry['observed_count'] > 0) == (entry['reachability'] == 'witnessed-initial'), 'Witness classification mismatch')
        require(entry['initial_class'] in {'observed-initial', 'declared-candidate', 'not-established-initial', 'profile-runtime'}, 'Unknown initial evidence class')
        require((entry['observed_count'] > 0) == (entry['initial_class'] == 'observed-initial'), 'Initial evidence classification mismatch')
        source_ref(entry['source'])
        callback_paths = []
        for callback in entry['callback_paths']:
            callback_paths.append(callback['path'])
            source_ref(callback['source'])
        require(len(callback_paths) == len(set(callback_paths)), 'Duplicate callback path')
        entries[key] = entry
    require(entries and {e['domain'] for e in entries.values()} == DOMAINS, 'Incomplete domain inventory')
    paths, by_owner = {}, {}
    for pathway in manifest['pathways']:
        require(pathway['id'] not in paths, 'Duplicate pathway')
        require(pathway['kind'] in KINDS and pathway['family'] in FAMILIES, 'Unknown pathway classification')
        require((pathway['bound'], pathway['family']) == BOUNDS[pathway['kind']], 'Incorrect pathway bound')
        require(pathway['classification'] in {'literal-with-conservative-envelope', 'dynamic-conservative-envelope'} and pathway['bound'], 'Missing pathway bound')
        require(pathway['owner'] in entries or pathway['owner'] == 'generator', 'Unknown pathway owner')
        ref = pathway['source']
        require(pathway['id'] == hash_json([pathway['owner'], ref['file'], ref['start_byte'], ref['end_byte'], pathway['kind']]), 'Pathway identity mismatch')
        source_ref(ref)
        paths[pathway['id']] = pathway
        by_owner.setdefault(pathway['owner'], set()).add(pathway['id'])
    for key, entry in entries.items():
        for pid in entry['pathways']:
            require(pid in paths and paths[pid]['owner'] == key, 'Pathway linkage mismatch')
        require(set(entry['pathways']) == by_owner.get(key, set()), 'Orphan pathway')
        if 'derived_from' in entry:
            require(entry['derived_from'] in entries, 'Missing base-form source')
    for audit in manifest['indirect_audit']:
        require(audit['entry'] in entries, 'Unknown indirect audit entry')
    source_ref(manifest['generator_audit']['source'])
    for ref in manifest['generator_audit']['methods'].values():
        source_ref(ref)
    for ids in manifest['generator_audit']['initial_forms'].values():
        require(ids and all(f'forms:{ident}' in entries for ident in ids), 'Uncatalogued generator form')
    if root is not None:
        frequencies = parse_json(confined(root, manifest['inputs']['frequencies']['file']).read_bytes())
        for domain in DOMAINS - {'core', 'conditions'}:
            expected = {('' if k == '<none>' else k): v for k, v in frequencies[domain].items()}
            actual = {e['id']: e['observed_count'] for e in entries.values() if e['domain'] == domain and e['observed_count']}
            require(actual == expected, f'Initial corpus coverage mismatch: {domain}')
    return entries


def validate_bundle(directory, root):
    directory, root = Path(directory), Path(root)
    docs = {name: parse_json((directory / f'{name}.json').read_bytes()) for name in ['manifest', 'fixtures', 'catalog', 'report']}
    manifest, fixtures, catalog, report = (docs[n] for n in ['manifest', 'fixtures', 'catalog', 'report'])
    entries = validate_manifest(manifest, root)
    for name, version in [('fixtures', 'r6-fixtures-1'), ('catalog', 'r6-catalog-1'), ('report', 'r6-report-1')]:
        sealed(docs[name], version)
    require(catalog['manifest_hash'] == report['manifest_hash'] == manifest['content_hash'], 'Manifest linkage mismatch')
    require(catalog['fixtures_hash'] == fixtures['content_hash'], 'Fixture linkage mismatch')
    cases = {}
    for case in fixtures['cases']:
        require(case['id'] not in cases, 'Duplicate fixture ID')
        require(case['profile'] == 'r6-constructed-fixed-teams-v1' and case['official_generator_witness'] is False, 'Fixture reachability overclaim')
        require(case['assertions_passed'] is True and case['replay_verified'] is True, 'Unverified reference fixture')
        require(case['result_hash'] == hash_json({k: case[k] for k in ['checkpoints', 'protocol']}), 'Fixture result mismatch')
        require(set(case['entries']) <= entries.keys() and set(case['fixture_families']) <= FAMILIES, 'Unknown fixture linkage')
        cases[case['id']] = case
    require({f['id'] for f in catalog['families']} == FAMILIES, 'Incomplete family catalog')
    require(len(catalog['families']) == len(FAMILIES), 'Duplicate fixture family')
    for family in catalog['families']:
        require(family['differential_status'] == 'planned', 'Family parity is not established')
        require(set(family['reference_cases']) == {key for key, c in cases.items() if family['id'] in c['fixture_families']}, 'Wrong family reference link')
    require(len(catalog['entries']) == len(entries) and {e['entry'] for e in catalog['entries']} == entries.keys(), 'Catalog omission or duplicate')
    for row in catalog['entries']:
        entry = entries[row['entry']]
        require(row['ticket'] == entry['ticket'] and row['fixture_families'] == entry['fixture_families'], 'Catalog ownership mismatch')
        require(row['differential_status'] == 'planned', 'Reference evidence is not differential parity')
        require(set(row['reference_cases']) == {key for key, c in cases.items() if row['entry'] in c['entries']}, 'Wrong reference evidence link')
    for name, expected in report['files'].items():
        require(digest(confined(directory, name).read_bytes()) == expected, 'Report file hash mismatch')
    for file, expected in report['producer'].items():
        require(digest(confined(root, file).read_bytes()) == expected, 'Producer source drift')
    counts = report['counts']
    require(counts['entries'] == len(entries) and counts['pathways'] == len(manifest['pathways']), 'Report counts mismatch')
    require(counts['reference_cases'] == counts['replay_verified'] == len(cases), 'Report fixture count mismatch')
    require(counts['custom_engine_supported'] == counts['unclassified_pathways'] == 0, 'Invalid promotion claim')
    require(counts['source_files'] == len(manifest['sources']) and counts['fixture_families'] == len(FAMILIES), 'Report source/family count mismatch')
    require(counts['by_domain'] == {d: {'envelope': sum(e['domain'] == d for e in entries.values()),
            'observed_initial': sum(e['domain'] == d and e['observed_count'] > 0 for e in entries.values())}
            for d in DOMAINS}, 'Report domain counts mismatch')
    return {'manifest_hash': manifest['content_hash'], 'entries_verified': len(entries),
            'pathways_verified': len(manifest['pathways']), 'reference_cases_verified': len(cases),
            'custom_engine_gate': 'blocked-unsupported', 'verification': 'artifact/source integrity; reference assertions executed in Node'}


class UnsupportedMechanic(RuntimeError):
    pass


class SupportGate:
    """R6 has no custom engine. No JSON support flag can grant implementation."""
    def __init__(self, manifest):
        self.entries = validate_manifest(manifest)

    def require(self, keys):
        keys = list(keys)
        if not keys:
            raise UnsupportedMechanic('Empty mechanics profile cannot authorize simulation')
        unknown = [key for key in keys if key not in self.entries]
        if unknown:
            raise UnsupportedMechanic(f'Unknown mechanics: {unknown}')
        first = self.entries[keys[0]]
        raise UnsupportedMechanic(f"Unsupported {first['key']}; implement {first['ticket']} and its fixture obligations before custom-engine execution")

    def require_profile(self):
        self.require(self.entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.directory, args.root), sort_keys=True))


if __name__ == '__main__':
    main()
