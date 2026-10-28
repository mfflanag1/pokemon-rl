"""A deliberately compact, lossy public-history summary, never full simulator state.

Categorical tokens use fixed SHA256 buckets (collisions are expected and disclosed).
The feature contract is independent of every sealed checkpoint/feature format.
"""
from hashlib import sha256
import numpy as np

FEATURES = 928
FIELDS = {'schema_version', 'side', 'request_id', 'turn', 'phase', 'own', 'opponent',
          'active_moves', 'weather', 'terrain', 'side_conditions', 'mask'}
STATS = ['atk', 'def', 'spa', 'spd', 'spe']
BOOSTS = STATS + ['accuracy', 'evasion']


def token(vector, value):
    digest = sha256(str(value).encode()).digest()
    vector[int.from_bytes(digest[:4], 'big') % len(vector)] += 1 if digest[4] & 1 else -1


def hp_fraction(condition):
    if not condition:
        return 0., 0.
    if 'fnt' in condition or condition.split(' ')[0] == '0':
        return 0., 1.
    values = condition.split(' ')[0].split('/')
    if len(values) != 2 or float(values[1]) <= 0:
        raise ValueError('Unknown condition encoding')
    return min(1., max(0., float(values[0]) / float(values[1]))), 1.


def mon_features(mon):
    x = np.zeros(64, np.float32)
    if mon is None:
        return x
    x[0:5] = [1, bool(mon['active']), *hp_fraction(mon.get('condition')), mon['level'] / 100]
    stats = mon.get('stats', {})
    x[5] = bool(stats)
    x[6:11] = [stats.get(s, 0) / 600 for s in STATS]
    x[11:18] = [mon.get('boosts', {}).get(s, 0) / 6 for s in BOOSTS]
    for key in ['species', 'ability', 'item', 'tera', 'tera_type']:
        token(x[18:], f'{key}:{mon.get(key, "<unknown>")}')
    for move in mon['moves']:
        token(x[18:], f'move:{move}')
    for type_ in mon.get('types') or []:
        token(x[18:], f'type:{type_}')
    token(x[18:], f'status:{(mon.get("condition") or "").split(" ")[1:]}')
    return x


def encode(player):
    if not isinstance(player, dict) or set(player) != FIELDS or player['schema_version'] != 'full-showdown-player-v1':
        raise ValueError('Actor accepts only the versioned player projection')
    if player['side'] not in ['p1', 'p2'] or len(player['own']) != 6 or len(player['opponent']) != 6:
        raise ValueError('Player must have six own and six public opponent slots')
    if len(player['active_moves']) > 4:
        raise ValueError('Unsupported active move count')
    opponent_fields = {'ident', 'species', 'level', 'condition', 'active', 'moves', 'ability', 'item', 'tera', 'boosts', 'types'}
    if any(p is not None and set(p) != opponent_fields for p in player['opponent']):
        raise ValueError('Opponent slots accept public fields only')
    mask = np.asarray(player['mask'])
    if mask.shape != (14,) or mask.dtype != np.bool_:
        raise ValueError('Expected 14-action boolean mask')
    x = np.zeros(FEATURES, np.float32)
    x[:768] = np.concatenate([mon_features(p) for p in player['own'] + player['opponent']])
    for i, move in enumerate(player['active_moves']):
        row = x[768 + 24*i:768 + 24*(i+1)]
        row[:5] = [1, move['disabled'], (move['pp'] or 0)/64, (move['maxpp'] or 0)/64, move['base_power']/250]
        for key in ['id', 'type', 'category']:
            token(row[5:], f'{key}:{move[key]}')
    context = x[864:]
    context[0] = min(player['turn'], 500) / 100
    phases = ['wait', 'move', 'replace', 'revival']
    if player['phase'] not in phases:
        raise ValueError('Unknown phase')
    context[1 + phases.index(player['phase'])] = 1
    for key in ['weather', 'terrain']:
        token(context[5:], f'{key}:{player[key]}')
    for side, conditions in player['side_conditions'].items():
        for condition, amount in conditions.items():
            if amount:
                token(context[5:], f'{side}:{condition}:{amount}')
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite public features')
    return x, mask
