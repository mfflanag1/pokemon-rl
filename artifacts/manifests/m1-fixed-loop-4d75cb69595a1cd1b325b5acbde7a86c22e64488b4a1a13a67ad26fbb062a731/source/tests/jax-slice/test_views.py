import copy
import json
import unittest
from pokemon_jax.views import reduce_view


def wire(side='p1', events=(), terminal=False):
    pp = [56, 48]
    ownhp = 105
    for line in events:
        if line.startswith('|move|' + side + 'a:'):
            pp['Quick Attack' in line] -= 1
        if line.startswith('|-damage|' + side + 'a:'):
            ownhp = int(line.split('|')[3].split('/')[0].split()[0])
    request = {'active': [{'moves': [{'id': m, 'pp': p, 'maxpp': mx, 'disabled': False}
        for m, p, mx in zip(('tackle', 'quickattack'), pp, (56,48))]}],
        'side': {'id': side, 'pokemon': [{'active': True, 'condition': f'{ownhp}/105'}]}}
    protocol = ['|start', '|switch|p1a: Rattata|Rattata, L50, M|105/105',
        '|switch|p2a: Rattata|Rattata, L50, M|105/105', '|turn|1', *events,
        '|request|' + json.dumps(request)]
    return {'side': side, 'request_id': 1, 'turn': 1, 'request': request, 'protocol': protocol}


class Views(unittest.TestCase):
    def test_initial_and_no_mutation(self):
        v = wire(); before = copy.deepcopy(v)
        self.assertEqual(reduce_view(v), {'header':[1,0,0,105,105,56,48,0,0,1,1], 'history':[[-1]*6 for _ in range(8)]})
        self.assertEqual(v, before)

    def test_exact_hp_and_reveal(self):
        v = wire(events=['|move|p2a: Rattata|Quick Attack|p1a: Rattata', '|-damage|p1a: Rattata|71/105'])
        self.assertEqual(reduce_view(v)['header'], [1,0,0,71,105,56,48,0,1,1,1])
        self.assertEqual(reduce_view(v)['history'][0], [1,1,1,0,34,71])

    def test_crit(self):
        v = wire(events=['|move|p1a: Rattata|Tackle|p2a: Rattata','|-crit|p2a: Rattata','|-damage|p2a: Rattata|47/105'])
        self.assertEqual(reduce_view(v)['history'][0], [1,0,0,1,58,47])

    def test_reject_foreign_carrier(self):
        v=wire(); v['rng']=123
        with self.assertRaises(ValueError): reduce_view(v)

    def test_reject_opponent_request(self):
        v=wire(); v['request']['side']['id']='p2'
        with self.assertRaises(ValueError): reduce_view(v)

    def test_reject_unreachable_and_inconsistent(self):
        for line in ['|-status|p1a: Rattata|brn', '|-damage|p1a: Rattata|100/105', '|turn|3', '|win|P1']:
            v=wire(events=[line])
            with self.subTest(line=line), self.assertRaises(ValueError): reduce_view(v)
        v=wire();v['request']['active'][0]['moves'][0]['pp']=55
        with self.assertRaises(ValueError):reduce_view(v)

    def test_unsupported_request_keys_even_false(self):
        for key in ('teamPreview','forceSwitch','wait'):
            v=wire();v['request'][key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):reduce_view(v)
        v=wire();v['request']['active'][0]['canTerastallize']=False
        with self.assertRaises(ValueError):reduce_view(v)

    def test_reported_profile_facts(self):
        for key,value in [('details','Pikachu, L50, M'),('ability','guts'),('item','leftovers'),('teraType','Ghost'),('moves',['protect','tackle'])]:
            v=wire();v['request']['side']['pokemon'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):reduce_view(v)

    def test_embedded_request_needs_own_side(self):
        for payload in ({}, {'side':{'id':'p2'}}, None):
            v=wire();v['protocol'][-1]='|request|'+json.dumps(payload)
            with self.subTest(payload=payload),self.assertRaises(ValueError):reduce_view(v)

    def test_terminal_stale_request_and_both_seats(self):
        events=['|move|p1a: Rattata|Tackle|p2a: Rattata','|-crit|p2a: Rattata','|-damage|p2a: Rattata|47/105',
                '|move|p2a: Rattata|Tackle|p1a: Rattata','|-damage|p1a: Rattata|72/105','|turn|2',
                '|move|p1a: Rattata|Quick Attack|p2a: Rattata','|-crit|p2a: Rattata','|-damage|p2a: Rattata|0 fnt',
                '|faint|p2a: Rattata','|win|P1']
        for side in ('p1','p2'):
            v=wire(side);stale=copy.deepcopy(v['request']);v['protocol'].extend(events);v['turn']=2
            view=reduce_view(v)
            self.assertEqual(v['request'],stale)
            self.assertEqual(view['header'][1:3],[1,1 if side=='p1' else 2])
            self.assertEqual(view['header'][9:],[0,0]);self.assertEqual(view['history'][2][4:], [47,0])
            self.assertEqual(view['history'][3],[-1]*6)
