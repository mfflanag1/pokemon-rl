import copy
import json
from pathlib import Path
import unittest
from pokemon_switch.contract import mask,command
from pokemon_switch.views import actor
ROWS=json.loads((Path(__file__).with_name('player-views.json')).read_text())


class ContractTests(unittest.TestCase):
    def test_live_round_trips_and_stable_identity(self):
        for r in ROWS:
            req=r['view']['request'];self.assertEqual(mask(req),r['mask'])
            for i,cmd in enumerate(r['commands']):
                if cmd is not None:self.assertEqual(command(req,i),cmd)
                else:
                    with self.assertRaises(ValueError):command(req,i)
        switched=next(r for r in ROWS if r['mask']==[True,True,True,False])
        self.assertEqual(command(switched['view']['request'],2),'switch 2')

    def test_invalid_indices_and_request_mutations(self):
        req=ROWS[0]['view']['request']
        for index in (True,False,-1,4,1.0,'1',None):
            with self.assertRaises(ValueError):command(req,index)
        for change in (lambda r:r.update(oracle={'opponent':'secret'}),
                       lambda r:r.update(forceSwitch=[1]),
                       lambda r:r['active'][0].update(canTerastallize='Grass'),
                       lambda r:r['active'][0]['moves'][0].update(pp=True),
                       lambda r:r['active'][0]['moves'][0].update(disabled='hidden'),
                       lambda r:r['active'][0]['moves'][0].update(id='struggle'),
                       lambda r:r['side']['pokemon'][1].update(ident='p2: A')):
            bad=copy.deepcopy(req);change(bad)
            with self.assertRaises(ValueError):mask(bad)

    def test_wait_pp_and_disabled(self):
        self.assertEqual(mask({'wait':True}),[False]*4)
        for kwargs in ({'pp':0},{'disabled':True}):
            req=copy.deepcopy(ROWS[0]['view']['request']);req['active'][0]['moves'][0].update(kwargs)
            self.assertEqual(mask(req),[False,True,False,True])
        req['active'][0]['moves'][1]['pp']=0
        with self.assertRaises(ValueError):mask(req)

    def test_actor_protocol_boundary_and_same_policy_value_input(self):
        for row in ROWS:
            value=actor(row['view']);self.assertEqual(value.policy_inputs(),value.value_inputs())
            exported=value.policy_inputs();exported['turn']=999
            self.assertNotEqual(value.policy_inputs()['turn'],999)
        v=ROWS[0]['view']
        for change in (lambda x:x.update(audit={}),lambda x:x.update(request_id=99),lambda x:x.update(side='p2'),
                       lambda x:x['protocol'].append('|win|P1'),
                       lambda x:x['request']['side']['pokemon'][0]['stats'].update(spe=999)):
            bad=copy.deepcopy(v);change(bad)
            with self.assertRaises(ValueError):actor(bad)

    def test_own_private_variant_is_a_positive_actor_control(self):
        v=copy.deepcopy(ROWS[0]['view']);before=actor(v)
        v['request']['side']['pokemon'][0]['stats']['spe']+=32
        for i,line in enumerate(v['protocol']):
            if line.startswith('|request|'):v['protocol'][i]='|request|'+json.dumps(v['request'],separators=(',',':'))
        self.assertNotEqual(before.feature_id,actor(v).feature_id)

    def test_opponent_exact_hp_is_rejected(self):
        v=copy.deepcopy(ROWS[0]['view'])
        v['protocol']=[line.replace('Squirtle, L50, M|100/100','Squirtle, L50, M|119/119') for line in v['protocol']]
        with self.assertRaises(ValueError):actor(v)
        v=copy.deepcopy(ROWS[0]['view']);v['protocol'].append('|-damage|p2a: A|58/119')
        with self.assertRaises(ValueError):actor(v)
