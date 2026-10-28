import copy
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_continuation import scalar as s
from pokemon_switch_engine import scalar as old

C={'crit':[False,False],'roll':[15,15],'tie':0}


class ScalarTests(unittest.TestCase):
    def test_long_switch_history_and_five_windows(self):
        state=s.initial([0,23]);boundaries=[]
        for turn in range(1,61):
            before=copy.deepcopy(state);actions=[3-state['active'][p] for p in (0,1)]
            state,events=s.step(state,actions,C)
            self.assertEqual(before['turn'],turn);self.assertEqual(state['turn'],turn+1)
            self.assertEqual(len(events),2)
            if state['turn'] in (13,25,37,49,61):boundaries.append(copy.deepcopy(state))
        self.assertEqual(len(boundaries),5);self.assertEqual(len(state['history']),120)
        self.assertEqual(len(s.view(state,0)['history']),128)
        self.assertEqual(s.view(state,0)['context'][4],120)
        with self.assertRaises(ValueError):s.step(state,[3,3],C)

    def test_original_prefix_and_input_immutability(self):
        a=s.initial([0,23]);b=old.initial([0,23])
        for _ in range(12):
            actions=[3-a['active'][p] for p in (0,1)];before=copy.deepcopy((a,actions,C))
            a,e=s.step(a,actions,C);b,f=old.step(b,actions,C)
            self.assertEqual(a,b);self.assertEqual(e,f);self.assertEqual((before[1],before[2]),(actions,C))
            self.assertEqual(before[0]['group'],a['group']-1)
            for side in (0,1):
                x=s.view(a,side);y=old.view(b,side);x['history']=x['history'][:32]
                self.assertEqual(x,y)

    def test_delayed_attacks_terminate_after_old_cutoff(self):
        state=s.initial([0,23])
        for _ in range(12):state,_=s.step(state,[3-state['active'][p] for p in (0,1)],C)
        while state['winner']==-1 and state['turn']<=60:
            actions=[next((i for i,ok in enumerate(m) if ok),-1) for m in s.masks(state)]
            state,_=s.step(state,actions,C)
        self.assertNotEqual(state['winner'],-1);self.assertGreater(state['turn'],12)

    def test_hidden_reserve_variant_and_speed(self):
        for observer in (0,1):
            groups={}
            for opponent in range(24):
                state=s.initial([0,opponent] if observer==0 else [opponent,0])
                key=s.ROSTER[opponent][0][0];v=s.view(state,observer)
                if key in groups:self.assertEqual(groups[key],v)
                groups[key]=v

    def test_bad_action_and_chance_rejected(self):
        state=s.initial([0,0])
        for actions in ([2,0],[-1,0],[4,0]):
            with self.assertRaises(ValueError):s.step(state,actions,C)
        with self.assertRaises(ValueError):s.step(state,[0,0],{**C,'roll':[16,0]})
        self.assertEqual(state,s.initial([0,0]))


if __name__=='__main__':unittest.main()
