"""Semantic bounds beyond the actor wire's fixed-shape JSON schema."""
from .feature_spec import CATALOG, LAYOUT


def validate_tensors(t):
    def bounded(value, maximum, label):
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError('Actor tensor bound: ' + label)

    for prefix, width in (('event', 192), ('fact', 128)):
        for row, length, overflow in zip(t[prefix + '_bytes'], t[prefix + '_lengths'], t[prefix + '_overflow']):
            bounded(length, width, prefix + ' length'); bounded(overflow, 1, prefix + ' overflow')
            if overflow and length != width: raise ValueError('Invalid truncated byte row')
            if any(not 1 <= x <= 256 for x in row[:length]) or any(row[length:]):
                raise ValueError('Invalid actor byte data/padding')
    domains = {'request_species': 'species', 'apparent_species': 'species', 'entry_species': 'species',
               'base_ability': 'abilities', 'ability': 'abilities', 'last_seen_ability': 'abilities',
               'item': 'items', 'last_seen_item': 'items', 'tera_type': 'types', 'terastallized': 'types',
               'revealed_tera': 'types', 'last_used_move': 'moves', 'move': 'moves',
               **{f'move_{i}': 'moves' for i in range(4)}}
    flags = {'present', 'active', 'hp_known', 'max_hp_known', 'fainted', 'reviving', 'hp_is_percent',
             'item_evidence_known', 'ability_evidence_known', 'last_used_move_known', 'transformed',
             'illusion_broken', 'appearance_changed', 'item_removed', 'ability_suppression_seen',
             'disabled', 'disabled_known', 'pp_known', 'max_pp_known'}
    groups = [(t['own'], 'own_columns'), (t['opponent_appearances'], 'appearance_columns'),
              ([t['own_public_active']], 'appearance_columns'), (t['advertised_moves'], 'advertised_move_columns')]
    for rows, columns in groups:
        for row in rows:
            d = dict(zip(LAYOUT[columns], row))
            for key, value in d.items():
                if key in domains: bounded(value, len(CATALOG['domains'][domains[key]]) - 1, key)
                elif key in flags: bounded(value, 1, key)
                elif key == 'true_roster_slot_known': bounded(value, 0, key)
                elif key == 'gender': bounded(value, 3, key)
                elif key == 'status': bounded(value, len(LAYOUT['statuses']) - 1, key)
            if d.get('hp_known') and d.get('max_hp_known') and d['hp'] > d['max_hp']:
                raise ValueError('Own HP exceeds maximum')
    g = dict(zip(LAYOUT['global_columns'], t['global']))
    for key in ('move_phase', 'switch_phase', 'wait_phase', 'pass_control', 'trapped', 'trapped_known', 'maybe_trapped', 'maybe_trapped_known', 'can_tera'):
        bounded(g[key], 1, key)
    if g['move_phase'] + g['switch_phase'] + g['wait_phase'] != 1:
        raise ValueError('Actor request phase must be one-hot')
    if (g['wait_phase'] or g['pass_control']) and any(t['action_mask']):
        raise ValueError('Control request has policy actions')
    for i, row in enumerate(t['actions']):
        kind, slot, subject, tera, legal = row
        bounded(kind, 3, 'action kind'); bounded(tera, 1, 'action tera'); bounded(legal, 1, 'action legal')
        expected_slot = i % 4 if i < 8 else i - 8
        if slot != expected_slot or bool(legal) != t['action_mask'][i]: raise ValueError('Action binding/mask mismatch')
        if kind == 0:
            if subject or tera or legal: raise ValueError('Unused action binding is populated')
        else:
            if kind != (1 if i < 4 else 2 if i < 8 else 3) or tera != int(4 <= i < 8):
                raise ValueError('Action kind/index mismatch')
            bounded(subject, len(CATALOG['domains']['moves' if i < 8 else 'species']) - 1, 'action subject')
