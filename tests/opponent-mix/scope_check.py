import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_full.policy import initialize
from pokemon_matchup.network import extend
from pokemon_full_train.storage import canonical
from pokemon_opponent_mix.common import *
from pokemon_opponent_mix.campaign import training_specs,training_regimes
from pokemon_terminal_matchup.common import diagnostic_specs as old_diagnostic

def games(reward,terminal=True,opponent='tactical'):
    return [{'episode':i,'swap':swap,'learner_side':'p2' if swap else 'p1',
        'opponent':opponent,'provenance':{'teams':[{'team_key':str(i),'partition':'development'}]},
        'decisions':1,'trace':[],'value_mean':.5,'terminal_value_mse':.25,
        'outcome':{'kind':'terminal' if terminal else 'administrative_cutoff',
        'rewards':{'p1':1-reward if swap else reward,'p2':reward if swap else 1-reward}}}
        for i in range(128) for swap in [False,True]]
class Scope(unittest.TestCase):
    def test_budget_and_mutations(self):
        p=read(PLAN);validate_plan(p)
        self.assertEqual(p['expected_counts']['cohorts'],3*2*128)
        self.assertEqual(p['expected_counts']['evaluation_games'],3*2*3*3*256)
        self.assertEqual(p['expected_counts']['optimizer_steps'],3*2*128*8)
        for k,v in [('seeds',[7103]),('cohorts',256),('ppo',{}),('prime_used',True),('evaluation_seed',1)]:
            q=copy.deepcopy(p);q[k]=v
            with self.assertRaises(ValueError):validate_plan(q)
    def test_both_arms_identical_f25_hosts_and_reset(self):
        for seed in [7103,8209,9311]:
            p,o,a=initial(None,seed,'self');q,qo,b=initial(None,seed,'mixed')
            same_state(p,o,q,qo);self.assertEqual(a,b);self.assertEqual(int(o[2]),0)
            self.assertTrue(all(np.all(np.asarray(v)==0) for group in o[:2] for v in group.values()))
            report=read(ROOT/'artifacts/runs/f25-terminal-matchup-v1'/str(seed)/'result.json')
            self.assertEqual(parameter_hash(p),report['arms']['matchup']['endpoints']['512']['parameter_hash'])
    def test_checkpoint_counters_and_pointer(self):
        p,o=extend(initialize(2))
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);save(d,p,o,'h',2,'self',0,0);q,qo=load(d,'h',2,'self',0,0);same_state(p,o,q,qo)
            for args in [('x',2,'self',0,0),('h',2,'mixed',0,0),('h',2,'self',0,8),('h',3,'self',0,0)]:
                with self.assertRaises(ValueError):load(d,*args)
            ptr=read(d/'latest.json');ptr['file']='../escape.npz';(d/'latest.json').write_bytes(canonical(ptr))
            with self.assertRaises(ValueError):load(d,'h',2,'self',0,0)
    def test_adaptation_does_not_imply_transfer(self):
        p=read(PLAN);panels={a:{e:{o:games(int(a=='mixed' and e=='128' and o=='tactical'),opponent=o)
            for o in p['opponents']} for e in ['0','128']} for a in p['arms']}
        r=analysis(panels,p)
        self.assertEqual(sum(len(c) for c in r['comparisons'].values()),9)
        self.assertFalse(r['mixture_benefit']['passed']);self.assertTrue(r['tactical_adaptation_benefit']['passed'])
        panels['mixed']['128']['base_power']=games(1,opponent='base_power')
        self.assertTrue(analysis(panels,p)['mixture_benefit']['passed'])
        panels['mixed']['128']={o:games(1,False,o) for o in p['opponents']}
        self.assertFalse(analysis(panels,p)['mixture_benefit']['passed'])
    def test_regression_and_pair_corruption_rejected(self):
        p=read(PLAN);panels={a:{e:{o:games(int(o=='random'),opponent=o) for o in p['opponents']}
            for e in ['0','128']} for a in p['arms']}
        panels['mixed']['128']['base_power']=games(1,opponent='base_power')
        panels['mixed']['128']['random']=games(0,opponent='random')
        self.assertFalse(analysis(panels,p)['mixture_benefit']['passed'])
        panels['mixed']['128']['random'][0]['swap']=True
        with self.assertRaises(ValueError):analysis(panels,p)
    def test_new_schedules_and_stream_namespaces(self):
        p=read(PLAN);prior=read(ROOT/'contracts/terminal-matchup-v1/plan.json')
        self.assertNotEqual(evaluation_specs(p,7103,'tactical',0,4)[0]['seed'],evaluation_specs(prior,7103,'tactical',0,4)[0]['seed'])
        self.assertNotEqual(diagnostic_specs(p,7103,0,4)[0]['seed'],evaluation_specs(p,7103,'tactical',0,4)[0]['seed'])
        a=training_specs(p,7103,0,None);b=training_specs(p,7103,0,'tactical')
        self.assertNotEqual(a[0]['seed'],b[0]['seed'])
        self.assertTrue(all(x['split']=='train' for x in a+b))
    def test_selfplay_terminal_value_excludes_cutoff(self):
        g=games(1)[0];g['trace']=[['p1',0,0,.8,'x'],['p2',0,0,.3,'x']]
        self.assertAlmostEqual(selfplay_value([g])['seat_views']['game_mean_mse'],.065)
        g['outcome']['kind']='administrative_cutoff'
        self.assertIsNone(selfplay_value([g])['seat_views']['game_mean_mse'])
if __name__=='__main__':unittest.main()
