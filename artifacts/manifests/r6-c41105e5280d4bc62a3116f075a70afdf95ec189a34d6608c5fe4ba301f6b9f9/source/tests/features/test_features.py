from copy import deepcopy
from pathlib import Path
import unittest
from pokemon_reference.contract import (CONTRACT, canonical, hash_json, parse_json,
                                        admissible_actions, action_mask, request_phase)
from pokemon_reference.player_history import PlayerHistory, load_actor_input
from pokemon_reference.actor_actions import ActionCodec
from pokemon_reference.feature_spec import LAYOUT, category
from pokemon_reference.feature_trajectories import assemble_episode, project_player_views

GOLDEN = parse_json((CONTRACT.parents[1] / 'tests/fixtures/r4-golden.json').read_bytes())


def request_for(side='p1'):
    r = deepcopy(GOLDEN['bundle']['episode']['payload']['decisions'][0]['view']['request'])
    r['side']['id'] = side; r['side']['name'] = 'Player ' + side[1]
    for mon in r['side']['pokemon']: mon['ident'] = side + mon['ident'][2:]
    return r


def view_for(request, protocol, start=0):
    body = {'viewer': request['side']['id'], 'request_id': sum(l.startswith('|request|') for l in protocol),
            'turn': next((int(l[6:]) for l in reversed(protocol) if l.startswith('|turn|')), 0),
            'phase': request_phase(request), 'request': deepcopy(request), 'protocol_start': start,
            'protocol_end': len(protocol), 'protocol_delta': protocol[start:], 'protocol_prefix_hash': hash_json(protocol),
            'admissible_actions': admissible_actions(request), 'action_mask': action_mask(request)}
    return {**body, 'view_id': hash_json(body)}


def initial(side='p1', request=None):
    r = request or request_for(side); own = r['side']['pokemon'][0]; other = 'p2' if side == 'p1' else 'p1'
    lines = ['|start', f"|switch|{side}a: {own['ident'].split(': ', 1)[1]}|{own['details']}|{own['condition']}",
             f'|switch|{other}a: Garchomp|Garchomp, L82, M|100/100', '|turn|1', '|request|' + canonical(r)]
    return view_for(r, lines)


def deliver(h, lines, request=None):
    old = h.snapshot()['protocol']; r = request or h.request
    protocol = old + lines + ['|request|' + canonical(r)]
    v = view_for(r, protocol, len(old))
    return h.append(v)


def columns(actor, group, row=0):
    names = LAYOUT['appearance_columns' if group in ('opponent_appearances', 'own_public_active') else 'own_columns']
    values = actor.model_inputs()[group]
    if group != 'own_public_active': values = values[row]
    return dict(zip(names, values))


class FeaturesTests(unittest.TestCase):
    def test_golden_round_trip_separate_model_inputs_and_targets(self):
        before = deepcopy(GOLDEN)
        trajectories = assemble_episode(GOLDEN['bundle'], GOLDEN['manifest'])
        for trajectory in trajectories:
            self.assertEqual(len(trajectory.actor_inputs), len(trajectory.targets))
            self.assertTrue(trajectory.targets[-1].truncated)
            self.assertTrue(trajectory.targets[-1].next_bootstrap)
            self.assertIsNotNone(trajectory.bootstrap)
            for actor in (*trajectory.actor_inputs, trajectory.bootstrap):
                self.assertEqual(load_actor_input(actor._wire), actor)
                self.assertEqual(actor.policy_inputs(), actor.value_inputs())
                self.assertNotIn('reward', actor._wire)
                self.assertNotIn('episode_key', actor._wire)
                self.assertNotIn('decision_id', actor._wire)
        self.assertEqual(GOLDEN, before)
        with self.assertRaises(Exception): load_actor_input(canonical(GOLDEN['bundle']['episode']))

    def test_exact_own_data_unknown_opponent_and_percent_precision(self):
        for side in ('p1', 'p2'):
            h = PlayerHistory(side); actor = h.append(initial(side))
            own, other = columns(actor, 'own'), columns(actor, 'opponent_appearances')
            mon = h.request['side']['pokemon'][0]
            self.assertEqual(own['hp'], int(mon['condition'].split('/')[0]))
            self.assertEqual(own['base_atk'], mon['stats']['atk'])
            self.assertGreater(own['item'], 1)
            self.assertEqual(other['shown_hp'], 100)
            self.assertEqual(other['shown_denominator'], 100)
            self.assertEqual(other['hp_is_percent'], 1)
            self.assertEqual(other['last_seen_item'], 0)
            self.assertEqual(other['last_seen_ability'], 0)
            self.assertEqual(other['true_roster_slot_known'], 0)
            self.assertEqual(other['revealed_tera'], 0)
            opponent = 'p2' if side == 'p1' else 'p1'
            for hp in ['1/100', '99/100', '100/100', '0 fnt']:
                actor = deliver(h, [f'|-damage|{opponent}a: Garchomp|{hp}'])
                row = columns(actor, 'opponent_appearances')
                self.assertEqual(row['shown_hp'], 0 if hp == '0 fnt' else int(hp.split('/')[0]))
            before = h.snapshot()
            with self.assertRaisesRegex(ValueError, 'precision'):
                deliver(h, [f'|-damage|{opponent}a: Garchomp|197/400'])
            self.assertEqual(h.snapshot(), before)

    def test_transactional_corruption_retry_stale_and_foreign_history(self):
        h = PlayerHistory('p1'); v = initial(); a = h.append(v)
        self.assertIs(h.append(deepcopy(v)), a)
        deliver(h, ['|turn|2'])
        before = h.snapshot()
        with self.assertRaisesRegex(ValueError, 'stale'): h.append(v)
        for mutation in ('hash', 'foreign', 'mask', 'extra'):
            old = h.snapshot()['protocol']; r = h.request
            lines = old + ['|request|' + canonical(r)]
            if mutation == 'foreign': lines.insert(len(old), '|request|' + canonical(request_for('p2')))
            bad = view_for(r, lines, len(old))
            if mutation == 'hash': bad['protocol_prefix_hash'] = '0' * 64
            if mutation == 'mask': bad['action_mask'][8] = True
            if mutation == 'extra': bad['opponent_private_state'] = {'ability': 'shadowtag'}
            bad['view_id'] = hash_json({k:v for k,v in bad.items() if k != 'view_id'})
            with self.assertRaises(Exception): h.append(bad)
            self.assertEqual(h.snapshot(), before)
        copied = h.request; copied['side']['pokemon'][0]['item'] = 'airballoon'
        self.assertNotEqual(copied, h.request)

    def test_illusion_does_not_merge_repeated_nicknames_or_rewrite_past(self):
        h = PlayerHistory('p1'); before = h.append(initial()); frozen = before._wire
        reveal = deliver(h, ['|move|p2a: Garchomp|Knock Off|p1a: Sandy Shocks',
                             '|-item|p2a: Garchomp|Choice Scarf', '|-damage|p2a: Garchomp|57/100',
                             '|replace|p2a: Zoroark|Zoroark-Hisui, L80, M', '|-end|p2a: Zoroark|Illusion'])
        row = columns(reveal, 'opponent_appearances')
        self.assertEqual(row['encounter'], 1)
        self.assertEqual(row['apparent_species'], category('species', 'Zoroark-Hisui'))
        self.assertEqual(row['entry_species'], category('species', 'Garchomp'))
        self.assertEqual(row['shown_hp'], 57)
        self.assertEqual(row['last_used_move'], category('moves', 'knockoff'))
        self.assertEqual(row['last_seen_item'], category('items', 'Choice Scarf'))
        self.assertEqual(row['illusion_broken'], 1)
        switched = deliver(h, ['|switch|p2a: Garchomp|Garchomp, L82, M|100/100'])
        self.assertEqual(columns(switched, 'opponent_appearances')['last_seen_item'], 0)
        self.assertEqual(columns(switched, 'opponent_appearances')['encounter'], 2)
        self.assertEqual(before._wire, frozen)
        self.assertEqual(columns(before, 'opponent_appearances')['apparent_species'], category('species', 'Garchomp'))

    def test_transform_uses_public_target_appearance_and_keeps_hp_level_unknown_set(self):
        h = PlayerHistory('p1'); h.append(initial())
        own = h.request['side']['pokemon'][0]; ident = 'p1a: ' + own['ident'].split(': ',1)[1]
        a = deliver(h, ['|-damage|p2a: Garchomp|57/100', f'|-transform|p2a: Garchomp|{ident}|[from] ability: Imposter'])
        row = columns(a, 'opponent_appearances')
        self.assertEqual(row['apparent_species'], category('species', own['details'].split(',')[0]))
        self.assertEqual(row['apparent_level'], 82)
        self.assertEqual(row['shown_hp'], 57)
        self.assertEqual(row['last_seen_ability'], 0)
        self.assertEqual(row['last_seen_item'], 0)
        self.assertEqual(row['last_used_move'], 0)
        self.assertEqual(row['transformed'], 1)
        changed = h.request; changed['side']['pokemon'][0]['ability'] = 'intimidate'
        a = deliver(h, [], changed)
        row = columns(a, 'own')
        self.assertEqual(row['base_ability'], category('abilities', own['baseAbility']))
        self.assertEqual(row['ability'], category('abilities', 'intimidate'))
        self.assertEqual(row['base_atk'], own['stats']['atk'])

    def test_evidence_source_and_removal_are_temporal_not_inferred_sets(self):
        h = PlayerHistory('p1'); h.append(initial())
        a = deliver(h, ['|-item|p2a: Garchomp|Choice Scarf|[from] ability: Frisk|[of] p1a: Sandy Shocks',
                        '|-ability|p1a: Sandy Shocks|Intimidate|[from] ability: Trace|[of] p2a: Garchomp'])
        self.assertEqual(columns(a, 'opponent_appearances')['last_seen_item'], category('items','choicescarf'))
        self.assertEqual(columns(a, 'opponent_appearances')['last_seen_ability'], 0)
        self.assertEqual(columns(a, 'own_public_active')['last_seen_ability'], category('abilities','intimidate'))
        a = deliver(h, ['|-enditem|p2a: Garchomp|Choice Scarf|[silent]', '|-endability|p1a: Sandy Shocks'])
        self.assertEqual(columns(a, 'opponent_appearances')['item_removed'], 1)
        self.assertEqual(columns(a, 'own_public_active')['ability_suppression_seen'], 1)
        a = deliver(h, ['|-item|p2a: Garchomp|Leftovers'])
        self.assertEqual(columns(a, 'opponent_appearances')['item_removed'], 0)

    def test_fainted_own_maxhp_retained_only_if_previously_observed(self):
        h = PlayerHistory('p1'); original = h.append(initial()); r = h.request
        maximum = columns(original,'own')['max_hp'];r['side']['pokemon'][0]['condition'] = '0 fnt'
        a = deliver(h, [], r)
        self.assertEqual(columns(a,'own')['max_hp'],maximum)
        self.assertEqual(columns(a,'own')['max_hp_known'],1)
        fresh = PlayerHistory('p1').append(initial(request=r))
        self.assertEqual(columns(fresh,'own')['max_hp_known'],0)

    def test_bounded_event_and_appearance_memory_exposes_overflow(self):
        h=PlayerHistory('p1');h.append(initial())
        lines=[f'|drag|p2a: Garchomp|Garchomp, L82, M|{100-i}/100' for i in range(8)]
        lines += ['|-futureeffect|' + 'x'*250] * 70
        a=deliver(h,lines); t=a.model_inputs();g=dict(zip(LAYOUT['global_columns'],t['global']))
        self.assertEqual(g['opponent_encounters'],9);self.assertEqual(g['opponent_encounters_dropped'],3)
        self.assertGreater(g['events_dropped'],0);self.assertEqual(t['event_overflow'],[1]*64)
        self.assertEqual(len(h.snapshot()['opponent_appearances']),9)

    def test_request_codec_recharge_struggle_wait_pass_revival_and_order(self):
        r=request_for();codec=ActionCodec(r)
        for a in admissible_actions(r): self.assertEqual(codec.decode(codec.encode(a)),a)
        with self.assertRaises(ValueError): codec.decode(True)
        with self.assertRaises(ValueError): codec.decode(8)
        r['side']['pokemon'][1],r['side']['pokemon'][2]=r['side']['pokemon'][2],r['side']['pokemon'][1]
        self.assertEqual(ActionCodec(r).bindings[9][2],category('species',r['side']['pokemon'][1]['details'].split(',')[0]))
        for name in ('recharge','struggle'):
            r=request_for();r['active']=[{'moves':[{'id':name,'move':name.title()}]}]
            a=PlayerHistory('p1').append(initial(request=r));t=a.model_inputs()
            self.assertEqual(t['action_mask'][:8],[True,False,False,False,False,False,False,False])
            self.assertEqual(t['advertised_moves'][0][-2:],[0,0])
            self.assertEqual(ActionCodec(r).decode(0),{'kind':'move','slot':0,'terastallize':False})
        r=request_for();r.pop('active');r['wait']=True
        codec=ActionCodec(r);self.assertEqual(codec.control,'wait');self.assertFalse(any(codec.mask))
        r.pop('wait');r['forceSwitch']=[False]
        codec=ActionCodec(r);self.assertEqual(codec.control,'pass');self.assertFalse(any(codec.mask))
        with self.assertRaises(ValueError):codec.encode({'kind':'pass'})
        r['forceSwitch']=[True];r['side']['pokemon'][0]['reviving']=True;r['side']['pokemon'][0]['condition']='0 fnt'
        self.assertTrue(ActionCodec(r).mask[8])

    def test_privileged_metadata_and_other_view_cannot_enter_projection(self):
        p=deepcopy(GOLDEN['bundle']['episode']['payload']);before=project_player_views(p,'p1')
        p['spec']={'secret':'different teams, seed and future'};p['frames']=[{'rng':'other','commitments':['different']}]
        p['outcome']={'winner':'p2'};p['transitions']=[];p['input_tape']=['hidden action']
        for d in p['decisions']:
            if d['side']=='p2':d['view']={'secret':'unreadable opponent request'}
            d['action']={'secret':'different action'};d['decision_id']='future-dependent ID'
        p['bootstrap_views']['p2']={'secret':'opponent future'}
        self.assertEqual(project_player_views(p,'p1'),before)
        # Full artifact validation still rejects this deliberately invalid diagnostic carrier.
        b=deepcopy(GOLDEN['bundle']);b['episode']['payload']=p
        with self.assertRaises(Exception):assemble_episode(b,GOLDEN['manifest'])

    def test_rehashed_tensor_corruption_is_rejected(self):
        actor = PlayerHistory('p1').append(initial())
        for group, mutate in [
            ('event_bytes', lambda t: t['event_bytes'][0].__setitem__(0, 257)),
            ('padding', lambda t: t['fact_bytes'][-1].__setitem__(-1, 3)),
            ('phase', lambda t: t['global'].__setitem__(3, 1)),
            ('unknown_identity', lambda t: t['opponent_appearances'][0].__setitem__(22, 1)),
            ('action', lambda t: t['actions'][0].__setitem__(1, 3)),
            ('vocabulary', lambda t: t['own'][0].__setitem__(1, 999999)),
        ]:
            with self.subTest(group=group):
                body = actor.to_dict(); mutate(body['tensors'])
                body['feature_id'] = hash_json({k:v for k,v in body.items() if k != 'feature_id'})
                with self.assertRaises(ValueError): load_actor_input(canonical(body))

    def test_public_source_tags_are_scoped_and_ambiguous_as_one_stays_ambiguous(self):
        h = PlayerHistory('p1'); h.append(initial())
        a = deliver(h, ['|-damage|p1a: Sandy Shocks|100/300|[from] ability: Rough Skin|[of] p2a: Garchomp'])
        self.assertEqual(columns(a, 'opponent_appearances')['last_seen_ability'], category('abilities', 'roughskin'))
        a = deliver(h, ['|-heal|p2a: Garchomp|60/100|[from] item: Leftovers'])
        self.assertEqual(columns(a, 'opponent_appearances')['last_seen_item'], category('items', 'leftovers'))
        a = deliver(h, ['|-ability|p2a: Garchomp|As One'])
        self.assertEqual(columns(a, 'opponent_appearances')['last_seen_ability'], category('abilities', 'asone'))
        self.assertNotEqual(category('abilities','asone'), category('abilities','asoneglastrier'))
        self.assertNotEqual(category('abilities','asone'), category('abilities','asonespectrier'))

    def test_unmatched_public_subjects_are_diagnostic_not_roster_guesses(self):
        h = PlayerHistory('p1'); before = h.append(initial())
        after = deliver(h, ['|-curestatus|p2: Garchomp|slp'])
        g = dict(zip(LAYOUT['global_columns'], after.model_inputs()['global']))
        old = dict(zip(LAYOUT['global_columns'], before.model_inputs()['global']))
        self.assertEqual(g['unstructured_events'], old['unstructured_events'] + 1)
        self.assertEqual(columns(before, 'opponent_appearances'), columns(after, 'opponent_appearances'))


if __name__ == '__main__': unittest.main()
