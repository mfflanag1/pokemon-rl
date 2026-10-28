"""Typed player features plus bounded observable history, packed without hashing.

The first 1278 floats are exactly the frozen F32 feature vector. Entity category
IDs use the frozen R5 catalog (0 unknown, 1 explicitly absent), and are exactly
representable in float32. Event sides use a separate relative enum: 0 means no
seat encoded (unknown or inapplicable), 1 own, 2 opponent. Public move-use counts
count protocol executions, including called moves; they are not PP deductions
or estimates of remaining PP.
"""
from pathlib import Path
import json
import math

import numpy as np

from pokemon_full.features import hp_fraction, STATS, BOOSTS
from pokemon_reference.feature_spec import CATALOG, INDEX
from pokemon_switch_audit.semantic import to_id
from pokemon_terminal_matchup.features import encode as base_encode

ROOT = Path(__file__).resolve().parents[2]
HISTORY_CONTRACT = json.loads((ROOT / 'packages/structured-policy/contract.json').read_text())
HISTORY_SCHEMA = HISTORY_CONTRACT['schema_version']
WINDOW = HISTORY_CONTRACT['window']
BASE_FEATURES = 1278
EVENT_VOCABS = {key: tuple(HISTORY_CONTRACT[key]) for key in
                ('event_kind', 'event_side', 'status', 'event_aux')}
CATEGORICAL_SIZES = {key: len(CATALOG['domains'][domain]) for key, domain in
                     (('species', 'species'), ('ability', 'abilities'), ('item', 'items'),
                      ('type', 'types'), ('move', 'moves'))}
CATEGORICAL_SIZES.update({key: len(value) for key, value in EVENT_VOCABS.items()})
MON_NUMERIC_FIELDS = ('present', 'own', 'active', 'hp_fraction', 'hp_known', 'level',
                      'stats_known', 'atk', 'def', 'spa', 'spd', 'spe',
                      'boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe',
                      'boost_accuracy', 'boost_evasion', 'reviving', 'type_known',
                      'tera_known', 'ability_known', 'item_known')
ACTION_NUMERIC_FIELDS = ('present', 'move', 'switch', 'tera', 'pp', 'maxpp', 'base_power', 'disabled')
EVENT_ID_FIELDS = tuple(HISTORY_CONTRACT['event_id_fields'])
EVENT_ID_SIZES = tuple(CATEGORICAL_SIZES[key] for key in
                       ('event_kind', 'event_side', 'event_side', 'move', 'species',
                        'ability', 'item', 'type', 'status', 'event_aux'))
FIELD_SHAPES = {
    'base': (BASE_FEATURES,), 'species': (12,), 'ability': (12,), 'item': (12,),
    'types': (12, 5), 'moves': (12, 4), 'status': (12,), 'mon_numeric': (12, 24),
    'action_move': (14,), 'action_type': (14,), 'action_numeric': (14, 8),
    'event_ids': (WINDOW, 10), 'event_numeric': (WINDOW, 4), 'event_mask': (WINDOW,),
    'public_move_counts': (2, CATEGORICAL_SIZES['move']),
}
FIELD_SLICES = {}
_offset = 0
for _key, _shape in FIELD_SHAPES.items():
    _size = math.prod(_shape)
    FIELD_SLICES[_key] = slice(_offset, _offset + _size)
    _offset += _size
FEATURES = _offset
_STATUS = {name: i for i, name in enumerate(EVENT_VOCABS['status'])}
_INTEGER_FIELDS = {'species', 'ability', 'item', 'types', 'moves', 'status',
                   'action_move', 'action_type', 'event_ids'}


def category(domain, name):
    """Resolve frozen aliases; unknown names remain unknown rather than hashing."""
    if name is None:
        return 0
    if not isinstance(name, str):
        raise ValueError('Category name must be text or null')
    if name == '':
        return 1
    canonical = CATALOG['names'][domain].get(name.lower())
    return INDEX[domain].get(canonical, 0)


def empty_history(side):
    if side not in ('p1', 'p2'):
        raise ValueError('Invalid history side')
    return {'schema_version': HISTORY_SCHEMA, 'side': side, 'total_events': 0,
            'events': [], 'public_move_counts': [[], []]}


def _integer(value, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('Expected integer history metadata')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError('History integer out of range')
    return int(value)


def _history(history, player, result):
    fields = {'schema_version', 'side', 'total_events', 'events', 'public_move_counts'}
    if not isinstance(history, dict) or set(history) != fields or history['schema_version'] != HISTORY_SCHEMA:
        raise ValueError('Expected versioned observable history')
    if history['side'] != player['side']:
        raise ValueError('Foreign player history rejected')
    events = history['events']
    if not isinstance(events, list) or len(events) > WINDOW:
        raise ValueError('Invalid history window')
    _integer(history['total_events'], len(events))
    previous_turn = -1
    for i, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {'ids', 'numeric'}:
            raise ValueError('Unexpected history event fields')
        ids, numeric = event['ids'], event['numeric']
        if not isinstance(ids, list) or len(ids) != 10 or not isinstance(numeric, list) or len(numeric) != 4:
            raise ValueError('Invalid history event shape')
        result['event_ids'][i] = [_integer(v, 0, size - 1) for v, size in zip(ids, EVENT_ID_SIZES)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numeric):
            raise ValueError('Invalid history numeric field')
        turn, fraction, known, magnitude = numeric
        if not float(turn).is_integer() or not previous_turn <= turn <= player['turn'] or turn < 0:
            raise ValueError('History is unordered or includes future events')
        if not 0 <= fraction <= 1 or known not in (0, 1) or abs(magnitude) > 12:
            raise ValueError('Invalid history numeric range')
        previous_turn = turn
        result['event_numeric'][i] = [(player['turn'] - turn) / 100., fraction, known, magnitude / 6.]
        result['event_mask'][i] = 1
    counts = history['public_move_counts']
    if not isinstance(counts, list) or len(counts) != 2:
        raise ValueError('Expected own and opponent public move counts')
    for side, rows in enumerate(counts):
        if not isinstance(rows, list) or len(rows) > CATEGORICAL_SIZES['move']:
            raise ValueError('Invalid public move counters')
        previous_id = -1
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError('Invalid public move counter row')
            move_id = _integer(row[0], previous_id + 1, CATEGORICAL_SIZES['move'] - 1)
            count = _integer(row[1], 1, 2**53 - 1)
            previous_id = move_id
            result['public_move_counts'][side, move_id] = math.log1p(count)


def encode(player, catalog, history):
    """Return fixed-shape NumPy tensors and the unchanged 14-action bool mask."""
    try:
        base, mask = base_encode(player, catalog, 'matchup')
    except KeyError as error:
        # The frozen prefix requires complete public metadata even when a
        # dynamic type override is present; never invent a substitute prefix.
        raise ValueError(f'Missing required player field or public metadata: {error}') from error
    if player['schema_version'] != 'full-showdown-player-v2':
        raise ValueError('Structured actor requires the v2 player projection')
    result = {key: np.zeros(shape, dtype=np.int32 if key in _INTEGER_FIELDS else np.float32)
              for key, shape in FIELD_SHAPES.items()}
    result['base'] = base
    for i, mon in enumerate(player['own'] + player['opponent']):
        if mon is None:
            continue
        own = i < 6
        result['species'][i] = category('species', mon['species'])
        result['ability'][i] = category('abilities', mon.get('ability'))
        result['item'][i] = category('items', mon.get('item'))
        types = mon.get('types')
        type_known = types is not None  # Explicit observation, before species fallback.
        if types is None:
            types = catalog['species'][to_id(mon['species'])]['types']
        if len(types) > 3:
            raise ValueError('Too many dynamic public types')
        result['types'][i, :3] = 1
        result['types'][i, :len(types)] = [category('types', name) for name in types]
        result['types'][i, 3] = category('types', mon.get('tera'))
        result['types'][i, 4] = category('types', mon.get('tera_type')) if own else 0
        result['moves'][i] = 1 if own else 0
        result['moves'][i, :min(4, len(mon['moves']))] = [category('moves', name) for name in mon['moves'][:4]]
        condition = mon.get('condition')
        status = (condition or '').split(' ')[1:]
        result['status'][i] = _STATUS.get(status[0], 0) if status else (1 if condition else 0)
        stats, boosts = mon.get('stats', {}), mon.get('boosts', {})
        result['mon_numeric'][i] = [
            1, own, bool(mon['active']), *hp_fraction(condition), mon['level'] / 100., bool(stats),
            *[stats.get(stat, 0) / 600. for stat in STATS],
            *[boosts.get(stat, 0) / 6. for stat in BOOSTS],
            bool(mon.get('reviving')), type_known, mon.get('tera') is not None,
            mon.get('ability') is not None, mon.get('item') is not None,
        ]
    for i in range(14):
        if i < 8:
            if i % 4 >= len(player['active_moves']):
                continue
            move = player['active_moves'][i % 4]
            result['action_move'][i] = category('moves', move['id'])
            result['action_type'][i] = category('types', move['type'])
            result['action_numeric'][i] = [1, 1, 0, i >= 4, (move['pp'] or 0) / 64.,
                                            (move['maxpp'] or 0) / 64., move['base_power'] / 250., move['disabled']]
        else:
            result['action_numeric'][i, :4] = [1, 0, 1, 0]
    _history(history, player, result)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite structured features')
    return result, mask


def pack(tensors):
    if set(tensors) != set(FIELD_SHAPES):
        raise ValueError('Unexpected structured tensor fields')
    if any(np.shape(tensors[key]) != shape for key, shape in FIELD_SHAPES.items()):
        raise ValueError('Unexpected structured tensor shape')
    return np.concatenate([np.asarray(tensors[key], dtype=np.float32).reshape(-1) for key in FIELD_SHAPES])


def unpack_packed(x):
    """Static slices work with NumPy or JAX arrays and any leading batch shape."""
    if x.shape[-1] != FEATURES:
        raise ValueError('Unexpected packed feature size')
    return {key: x[..., FIELD_SLICES[key]].reshape((*x.shape[:-1], *shape))
            for key, shape in FIELD_SHAPES.items()}


def encode_packed(player, catalog, history):
    tensors, mask = encode(player, catalog, history)
    return pack(tensors), mask
