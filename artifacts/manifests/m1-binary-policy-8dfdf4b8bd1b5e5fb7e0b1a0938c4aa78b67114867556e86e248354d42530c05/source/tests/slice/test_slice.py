from dataclasses import replace
import copy
import unittest
from pokemon_slice.engine import PROFILE,State,initial,step,damage
from pokemon_slice.verify import validate_plan
from pokemon_reference.corpus import read_json
from pokemon_reference.contract import hash_json
from pathlib import Path


def event(actor,action,crit=False,roll=85):
    return {'actor':actor,'move':('tackle','quickattack')[action],'crit':crit,'roll':roll,'damage':damage(crit,roll)}


class SliceTest(unittest.TestCase):
    def test_reference_plan_pin_and_sources(self):
        plan=read_json(Path(__file__).resolve().parents[2]/'artifacts/runs/m1-slice-v1/plan.json')
        validate_plan(plan)
        bad=copy.deepcopy(plan);bad['reference']['commit']='0'*40
        bad['reference']['manifest_id']=hash_json({k:v for k,v in bad['reference'].items() if k!='manifest_id'})
        with self.assertRaisesRegex(ValueError,'Showdown pin'):validate_plan(bad)

    def test_damage_rounding_and_all_outcomes(self):
        self.assertEqual(damage(False,85),33);self.assertEqual(damage(False,100),39)
        self.assertEqual(damage(True,85),49);self.assertEqual(damage(True,100),58)
        outcomes=[damage(c,r) for c in [False,True] for r in range(85,101)]
        self.assertEqual((min(outcomes),max(outcomes)),(33,58))
        for crit,roll in [(1,85),(False,84),(False,101),(False,85.)]:
            with self.assertRaises(ValueError):damage(crit,roll)

    def test_priority_both_seats_and_tie_orders(self):
        for first in [0,1]:
            actions=[0,0];actions[first]=1
            s=step(initial(PROFILE),actions,[event(first,1),event(1-first,0)],None)
            self.assertEqual(s.hp,(72,72));self.assertEqual(s.turn,2)
            self.assertEqual(s.pp[first],(56,47));self.assertEqual(s.pp[1-first],(55,48))
            s=step(initial(PROFILE),[0,0],[event(first,0),event(1-first,0)],first)
            self.assertEqual(s.hp,(72,72))

    def test_ko_cancels_second_move_no_pp_and_terminal_has_no_requests(self):
        for first in [0,1]:
            hp=[105,105];hp[1-first]=33
            s=replace(initial(PROFILE),hp=tuple(hp),turn=3)
            actions=[0,0];actions[first]=1
            after=step(s,actions,[event(first,1)],None)
            self.assertEqual(after.winner,first);self.assertEqual(after.turn,3)
            self.assertEqual(after.pp[1-first],(56,48));self.assertEqual(after.snapshot()['requests'],[])
            with self.assertRaisesRegex(ValueError,'Terminal'):step(after,actions,[],None)

    def test_duration_pp_bound(self):
        s=initial(PROFILE)
        for turn in range(4):
            events=[event(0,0)] if turn==3 else [event(0,0),event(1,0)]
            s=step(s,[0,0],events,0)
        self.assertEqual(s.winner,0);self.assertEqual(s.pp,((52,48),(53,48)))

    def test_unsupported_config_action_stale_or_extra_event_rejects(self):
        p=copy.deepcopy(PROFILE);p['item']='Leftovers'
        with self.assertRaisesRegex(ValueError,'Unsupported profile'):initial(p)
        s=initial(PROFILE)
        for actions,events,tie in [(['move 1',0],[],0),([0,0],[],0),([0,0],[event(1,0),event(0,0)],0),
                ([1,0],[event(0,1),event(1,0)],1),([0,0],[event(0,0),event(1,0),event(0,0)],0)]:
            with self.assertRaises(ValueError):step(s,actions,events,tie)
        self.assertEqual(s,initial(PROFILE))
        with self.assertRaisesRegex(ValueError,'PP exhausted'):step(replace(s,pp=((0,48),(56,48))),[0,0],[],0)
