"""Public action-aligned nominal type proxies, not exact damage calculations.

Ignores abilities, items, dynamic move typing/power and Stellar modifiers. Own
absent members cannot support legal actions. Unknown opponent types produce zero
outgoing pressure; unrevealed moves and opponent base stats are never inferred.
STAB retains ordinary public species/types alongside known/prospective Tera.
F24 accepts versioned dynamic public types, including known typeless and added types.
Historical v1 probes retain their original observation limits.
Foe-type-known remains a context flag even for a switch with no known attacks.
"""
import math
import numpy as np
from pokemon_full.features import encode as legacy_encode, hp_fraction
from pokemon_switch_audit.semantic import TYPES, to_id

CHANNELS = 25
EXTRA = 350
FEATURES = 1278


def legacy_player(player):
    """Explicit v2-to-v1 encoder adapter; no coordinator/private-opponent fields."""
    if not isinstance(player, dict):
        raise ValueError('Expected player projection')
    version = player.get('schema_version')
    if version not in ('full-showdown-player-v1', 'full-showdown-player-v2'):
        raise ValueError('Unknown player version')
    if version == 'full-showdown-player-v2':
        own_fields = {'ident','species','level','condition','active','moves','ability',
                      'item','tera','boosts','tera_type','stats','reviving','types'}
        if any(set(mon) != own_fields for mon in player.get('own', [])):
            raise ValueError('V2 own field inventory')
    return {**player, 'schema_version':'full-showdown-player-v1'}


def base_encode(player):
    return legacy_encode(legacy_player(player))


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid public numeric metadata')
    return float(value)


def _types(value, dynamic=False):
    allowed = (*TYPES, '???') if dynamic else TYPES
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= (3 if dynamic else 2):
        raise ValueError('Invalid public types')
    if any(t not in allowed for t in value):
        raise ValueError('Invalid public types')
    ordinary = [t for t in value if t != '???']
    if len(set(ordinary)) != len(ordinary):
        raise ValueError('Invalid duplicate public type')
    # Removing two distinct ordinary types can produce two neutral placeholders.
    return list(value)


def _move(move):
    typ, category = move['type'], move['category']
    power = _number(move['base_power'], 0, 250)
    if category not in ('Physical', 'Special', 'Status') or (typ not in TYPES and not (typ == '???' and category == 'Status' and power == 0)):
        raise ValueError('Invalid public move metadata')
    return typ, category, power


def validate_catalog(catalog):
    """Full pinned catalog admission, once per pool; no hidden team information."""
    if catalog['schema_version'] != 'f17-public-metadata-v1' or tuple(catalog['types']) != TYPES:
        raise ValueError('Metadata schema/types changed')
    for name, record in catalog['species'].items():
        # The pinned all-generation Dex includes this unreachable Gen1 glitch.
        # Actual actor lookups still reject Bird through ordinary()/ _types().
        if name == 'missingno' and record['types'] == ['Bird', 'Normal']:
            continue
        _types(record['types'])
    for record in catalog['moves'].values():
        _move(record)
    for attack in TYPES:
        for defense in TYPES:
            if _number(catalog['effectiveness'][attack][defense], 0, 2) not in (0, .5, 1, 2):
                raise ValueError('Invalid nominal type chart')


def _extras(player, catalog, mask):
    if catalog['schema_version'] != 'f17-public-metadata-v1' or tuple(catalog['types']) != TYPES:
        raise ValueError('Metadata schema/types changed')

    def ordinary(mon):
        # Require real metadata even when an explicit public type override exists.
        species = _types(catalog['species'][to_id(mon['species'])]['types'])
        return _types(mon['types'], dynamic=True) if mon.get('types') is not None else species

    def tera(mon):
        value = mon.get('tera')
        if value and value not in TYPES:
            raise ValueError('Invalid public Tera type')
        return value

    def defense(mon, prospective=None):
        original = ordinary(mon)
        transformed = prospective or tera(mon)
        return [transformed] if transformed and transformed != 'Stellar' else original

    def effect(typ, defending):
        if typ not in TYPES or not defending:
            return 0.
        values = [1. if t == '???' else _number(catalog['effectiveness'][typ][t], 0, 2) for t in defending]
        if any(v not in (0, .5, 1, 2) for v in values):
            raise ValueError('Invalid nominal type chart')
        return math.prod(values)

    def stab(typ, mon, prospective=None):
        original = ordinary(mon)
        transformed = prospective or tera(mon)
        if transformed and transformed != 'Stellar' and typ == transformed:
            return 2. if typ in original else 1.5
        return 1.5 if typ in original else 1.

    def lookup(move_id):
        return catalog['moves'][to_id(move_id)]

    # Validate all observed records independently of legality and arm selection.
    for mon in player['own'] + player['opponent']:
        if mon is not None:
            defense(mon)
            for move_id in mon['moves']:
                _move(lookup(move_id))
    for move in player['active_moves']:
        _move(move)
    active = [m for m in player['own'] if m and m['active']]
    foes = [m for m in player['opponent'] if m and m['active']]
    if len(active) > 1 or len(foes) > 1:
        raise ValueError('Expected singles public projection')
    foe = foes[0] if foes else None
    ft = defense(foe) if foe else []
    fh, fk = hp_fraction(foe['condition']) if foe else (0., 0.)
    revealed = [lookup(m) for m in foe['moves']] if foe else []

    def outgoing(move, mon, prospective):
        typ, category, power = _move(move)
        s = stab(typ, mon, prospective) if typ in TYPES else 0.
        e = effect(typ, ft)
        p = (power / 250) * (s / 2) * (e / 4) if category != 'Status' else 0.
        return [float(category == c) for c in ('Physical', 'Special', 'Status')] + [power / 250, s / 2, float(bool(ft)), e / 4, p]

    out = np.zeros((14, CHANNELS), dtype=np.float32)
    for i in range(14):
        if not mask[i]:
            continue
        mon = active[0] if i < 8 and active else (player['own'][i - 8] if i >= 8 else None)
        if mon is None:
            raise ValueError('Legal action requires own member')
        prospective = mon.get('tera_type') if 4 <= i < 8 else None
        if 4 <= i < 8 and prospective not in TYPES:
            raise ValueError('Legal Tera action requires known own Tera type')
        dt = defense(mon, prospective)
        row = out[i]
        row[:4] = [1, i < 8, i >= 8, 4 <= i < 8]
        if i < 8:
            if i % 4 >= len(player['active_moves']):
                raise ValueError('Legal move slot missing metadata')
            row[4:12] = outgoing(player['active_moves'][i % 4], mon, prospective)
        elif mon['moves']:
            candidates = [outgoing(lookup(m), mon, None) for m in mon['moves']]
            row[4:12] = max(candidates, key=lambda c: c[-1])
        row[9] = bool(ft)
        row[12:14] = hp_fraction(mon['condition'])
        row[14] = any(s in (mon['condition'] or '').split()[1:] for s in ('brn', 'par', 'slp', 'psn', 'tox', 'frz'))
        row[15:18] = [fh, fk, min(len(revealed), 4) / 4]
        damaging = [m for m in revealed if m['category'] != 'Status' and m['base_power'] > 0]
        row[18] = max((m['base_power'] / 250 * stab(m['type'], foe) / 2 * effect(m['type'], dt) / 4 for m in damaging), default=0.)
        row[19] = bool(damaging)
        for j, stat in enumerate(('atk', 'spa', 'def', 'spd', 'spe')):
            if stat not in mon.get('stats', {}):
                raise ValueError('Legal action requires known own stats')
            value = _number(mon['stats'][stat], 0, 1200)
            stage = _number(mon.get('boosts', {}).get(stat, 0), -6, 6)
            if not stage.is_integer():
                raise ValueError('Invalid boost stage')
            row[20 + j] = value * ((2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)) / 1200
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite nominal features')
    return out.reshape(EXTRA)


def extras(player, catalog):
    """Return flattened 14 by 25 public action rows, validating actor schema."""
    _, mask = base_encode(player)
    return _extras(player, catalog, mask)


def encode(player, catalog, arm):
    """Append identical validated extras (or their zero control) to sealed 928."""
    if arm not in ('zero', 'matchup'):
        raise ValueError('Unknown matchup arm')
    base, mask = base_encode(player)
    extra = _extras(player, catalog, mask)
    return np.concatenate((base, extra if arm == 'matchup' else np.zeros_like(extra))), mask
