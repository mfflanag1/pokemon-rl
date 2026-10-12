"""R5 deterministic diagnostics, not a learned policy or simulator accuracy proof."""
from .contract import canonical, hash_json, parse_json
from .feature_spec import LAYOUT, category
from .player_history import PlayerHistory, load_actor_input
from .actor_actions import ActionCodec


def synthetic_probe(actor):
    """Exercise the same tensor-only boundary for a deterministic policy/value stub."""
    t = actor.model_inputs()
    digest = hash_json(t)
    scores = [int(hash_json([digest, i])[:8], 16) if legal else None
              for i, legal in enumerate(t['action_mask'])]
    return {'policy_scores': scores, 'value_probe': int(digest[:8], 16)}


def check_live_fixtures(data):
    if data['content_hash'] != hash_json({k: v for k, v in data.items() if k != 'content_hash'}):
        raise ValueError('Live fixture checksum')
    outputs = {}; cases = {c['id']: c for c in data['cases']}
    for c in data['cases']:
        history = PlayerHistory(c['viewer'])
        outputs[c['id']] = tuple(history.append(v) for v in c['views'])
        for actor in outputs[c['id']]:
            if load_actor_input(actor._wire) != actor: raise ValueError('Actor round trip')
    for p in data['pairs']:
        a, b = outputs[p['a']], outputs[p['b']]
        if (a == b) != p['equal']: raise ValueError('Live input pair failed: ' + p['kind'])
        if (synthetic_probe(a[-1]) == synthetic_probe(b[-1])) != p['equal']:
            raise ValueError('Policy/value boundary probe failed')
    for s in data['semantics']:
        c = cases[s['case']]; before, after = outputs[c['id']]
        old, new = before.model_inputs(), after.model_inputs()
        if s['kind'] == 'opponent_transform':
            row = dict(zip(LAYOUT['appearance_columns'], new['opponent_appearances'][0]))
            previous = dict(zip(LAYOUT['appearance_columns'], old['opponent_appearances'][0]))
            assert row['transformed'] == 1
            assert row['apparent_species'] == category('species', c['evidence']['target_species'])
            for key in ('shown_hp', 'shown_denominator', 'apparent_level', 'last_seen_item', 'last_seen_ability', 'last_used_move'):
                assert row[key] == previous[key], key
        elif s['kind'] == 'own_transform':
            row = dict(zip(LAYOUT['own_columns'], new['own'][0]))
            assert [row['base_' + k] for k in ('atk', 'def', 'spa', 'spd', 'spe')] == [c['evidence']['before']['base_stats'][k] for k in ('atk', 'def', 'spa', 'spd', 'spe')]
            assert row['hp'] == c['evidence']['before']['hp']
            assert row['level'] == c['evidence']['before']['level']
        elif s['kind'] == 'absorb_heal':
            row = dict(zip(LAYOUT['appearance_columns'], new['opponent_appearances'][0]))
            assert row['last_seen_ability'] == category('abilities', c['evidence']['ability'])
            own_before = dict(zip(LAYOUT['appearance_columns'], old['own_public_active']))
            own_after = dict(zip(LAYOUT['appearance_columns'], new['own_public_active']))
            assert own_after['last_seen_ability'] == own_before['last_seen_ability']
        else:
            assert s['kind'] == 'illusion_replace'
            h = PlayerHistory(c['viewer']); disguise = h.append(c['evidence']['disguised'])
            row = dict(zip(LAYOUT['appearance_columns'], disguise.model_inputs()['opponent_appearances'][0]))
            assert row['apparent_species'] == category('species', c['evidence']['apparent'])
            row = dict(zip(LAYOUT['appearance_columns'], new['opponent_appearances'][0]))
            assert row['apparent_species'] == category('species', c['evidence']['real'])
            assert row['entry_species'] == category('species', c['evidence']['apparent'])
            assert row['illusion_broken'] == 1 and row['true_roster_slot_known'] == 0
    action_count = 0
    for case in data['requests']:
        codec = ActionCodec(case['request'])
        assert list(codec.mask) == case['mask']
        for record in case['actions']:
            if record['index'] is None:
                assert codec.control == 'pass' and not any(codec.mask)
            else:
                assert codec.encode(record['action']) == record['index']
                assert codec.decode(record['index']) == record['action']
            action_count += 1
    return {'cases': len(data['cases']), 'invariant_pairs': sum(p['equal'] for p in data['pairs']),
            'positive_controls': sum(not p['equal'] for p in data['pairs']),
            'semantic_probes': len(data['semantics']), 'request_cases': len(data['requests']),
            'node_python_action_round_trips': action_count, 'fixture_hash': data['content_hash']}


if __name__ == '__main__':
    import sys
    print(canonical(check_live_fixtures(parse_json(sys.stdin.read()))))
