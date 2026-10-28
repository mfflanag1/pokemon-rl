"""Exact descriptive coverage and feature-hash collision counts.

Pair coverage is relative to the same training family. Hash collisions measure
only the supplied hashes, which already include masks; they establish neither
statistical independence nor absence of broader ambiguity.
"""

from collections import Counter, defaultdict
import re


_FAMILIES = ('normal_move', 'tera_move', 'voluntary_switch', 'forced_switch')
_KEYS = frozenset(('game', 'phase', 'target', 'own_species', 'foe_species',
                   'destination_species', 'feature_hash'))
_HASH = re.compile(r'[0-9a-f]{64}')


def _validate(records, name):
    if type(records) is not list or not records:
        raise ValueError(f'{name} must be a nonempty list')
    for index, row in enumerate(records):
        if type(row) is not dict or row.keys() != _KEYS:
            raise ValueError(f'{name}[{index}] must have exactly the record keys')
        if type(row['game']) is not int or row['game'] < 0:
            raise ValueError('game must be a nonnegative Python int')
        if type(row['phase']) is not str or row['phase'] not in ('move', 'replace', 'revival'):
            raise ValueError('invalid phase')
        if type(row['target']) is not int or not 0 <= row['target'] <= 13:
            raise ValueError('target must be a Python int from 0 to 13')
        for field in ('own_species', 'foe_species'):
            if type(row[field]) is not str or not row[field]:
                raise ValueError(f'{field} must be a nonempty string')
        destination = row['destination_species']
        if type(destination) is not str or bool(destination) != (row['target'] >= 8):
            raise ValueError('destination must be empty exactly for move targets')
        if row['phase'] != 'move' and row['target'] < 8:
            raise ValueError('switch phases require switch targets')
        if type(row['feature_hash']) is not str or _HASH.fullmatch(row['feature_hash']) is None:
            raise ValueError('feature_hash must be 64 lowercase hexadecimal characters')


def _family(row):
    if row['phase'] != 'move':
        return 'forced_switch'
    if row['target'] < 4:
        return 'normal_move'
    if row['target'] < 8:
        return 'tera_move'
    return 'voluntary_switch'


def _pairs(rows, destination=False):
    field = 'destination_species' if destination else 'own_species'
    return {(row[field], row['foe_species']) for row in rows if row[field]}


def _hash_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['feature_hash']].append(row['target'])
    return groups


def _counts(rows, field, exclude_empty=False):
    return dict(sorted(Counter(str(row[field]) for row in rows
                               if not exclude_empty or row[field]).items()))


def _split(rows):
    families = {}
    for name in _FAMILIES:
        members = [row for row in rows if _family(row) == name]
        families[name] = {
            'rows': len(members),
            'games': len({row['game'] for row in members}),
            'target_counts': _counts(members, 'target'),
            'own_species_counts': _counts(members, 'own_species'),
            'foe_species_counts': _counts(members, 'foe_species'),
            'destination_species_counts': _counts(members, 'destination_species', True),
            'unique_own_foe_pairs': len(_pairs(members)),
            'unique_destination_foe_pairs': len(_pairs(members, True)),
        }
    groups = _hash_groups(rows)
    conflicts = [labels for labels in groups.values() if len(set(labels)) > 1]
    return {
        'rows': len(rows), 'games': len({row['game'] for row in rows}),
        'families': families,
        'unique_feature_hashes': len(groups),
        'repeated_hash_groups': sum(len(labels) > 1 for labels in groups.values()),
        'conflicting_label_groups': len(conflicts),
        'rows_in_conflicting_groups': sum(map(len, conflicts)),
    }


def summarize(train, probe):
    """Validate both record lists and return JSON-native, unfiltered exact counts.

    Invalid records raise ValueError. All families, including empty families, are
    present. Probe family dictionaries additionally contain pair coverage counts.
    """
    _validate(train, 'train')
    _validate(probe, 'probe')
    result = {'train': _split(train), 'probe': _split(probe)}
    for name in _FAMILIES:
        training = [row for row in train if _family(row) == name]
        probing = [row for row in probe if _family(row) == name]
        for destination, label in ((False, 'own_foe'), (True, 'destination_foe')):
            seen = _pairs(training, destination)
            field = 'destination_species' if destination else 'own_species'
            eligible = [row for row in probing if row[field]]
            count = sum((row[field], row['foe_species']) in seen for row in eligible)
            result['probe']['families'][name][f'seen_{label}_rows'] = count
            result['probe']['families'][name][f'unseen_{label}_rows'] = len(eligible) - count
    train_hashes, probe_hashes = _hash_groups(train), _hash_groups(probe)
    shared = train_hashes.keys() & probe_hashes.keys()
    result['overlap'] = {
        'shared_feature_hashes': len(shared),
        'shared_feature_rows_in_probe': sum(len(probe_hashes[key]) for key in shared),
        'conflicting_shared_hashes': sum(
            len(set(train_hashes[key]) | set(probe_hashes[key])) > 1 for key in shared),
    }
    return result
