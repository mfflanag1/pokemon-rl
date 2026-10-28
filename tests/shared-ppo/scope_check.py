import copy,json,tempfile,unittest
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pokemon_shared.learning import initial as fresh
from pokemon_full_train.storage import canonical,digest
from pokemon_shared_ppo.common import *

def games(reward,terminal=True,opponent='tactical'):
    return [{'episode':i,'swap':swap,'learner_side':'p2' if swap else 'p1',
        'opponent':opponent,'provenance':{'teams':[{'team_key':str(i),'partition':'development'}]},
        'decisions':1,'trace':[],'value_mean':.5,'terminal_value_mse':.25,
        'outcome':{'kind':'terminal' if terminal else 'administrative_cutoff',
            'rewards':{'p1':1-reward if swap else reward,'p2':reward if swap else 1-reward}}}
        for i in range(128) for swap in [False,True]]

class Scope(unittest.TestCase):
    def test_registered_budget_and_all_seeds(self):
        p=read(PLAN);validate_plan(p)
        self.assertEqual(p['expected_counts']['evaluation_games'],3*7*3*256)
        for k,v in [('seeds',[7103]),('cohorts',64),('ppo',{}),('prime_used',True),('evaluation_seed',1)]:
            q=copy.deepcopy(p);q[k]=v
            with self.assertRaises(ValueError):validate_plan(q)
    def test_initial_weights_preserved_adam_reset(self):
        previous=read(ROOT/'contracts/shared-battle-v1/plan.json')
        for seed in [7103,8209,9311]:
            for arm in ['flat','shared']:
                p,o,i=initial(previous,seed,arm);q,qo,_=load_policy(previous,seed,arm)
                for k in p:np.testing.assert_array_equal(p[k],q[k])
                self.assertEqual(int(o[2]),0);self.assertEqual(int(qo[2]),8192)
                for state in o[:2]:self.assertTrue(all(np.all(np.asarray(v)==0) for v in state.values()))
    def test_architecture_checkpoint_roundtrip_and_identity(self):
        for arm in ['flat','shared']:
            p,o=fresh(2,arm)
            with tempfile.TemporaryDirectory() as directory:
                d=Path(directory);save(d,p,o,'hash',2,arm,0,0);q,qo=load(d,'hash',2,arm,0,0)
                same_state(p,o,q,qo)
                for args in [('other',2,arm,0,0),('hash',3,arm,0,0),('hash',2,arm,1,0),('hash',2,arm,0,8)]:
                    with self.assertRaises(ValueError):load(d,*args)
                pointer=read(d/'latest.json');pointer['file']='../escape.npz'
                (d/'latest.json').write_bytes(canonical(pointer))
                with self.assertRaises(ValueError):load(d,'hash',2,arm,0,0)
    def test_checkpoint_array_corruption_rejected(self):
        p,o=fresh(2,'shared')
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);save(d,p,o,'h',2,'shared',0,0);ptr=read(d/'latest.json');path=d/ptr['file']
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            arrays['v__value_b']=np.float32(-1)
            with open(path,'wb') as f:np.savez_compressed(f,**arrays)
            ptr['sha256']=digest(path);(d/'latest.json').write_bytes(canonical(ptr))
            with self.assertRaises(ValueError):load(d,'h',2,'shared',0,0)
    def test_complete_contrasts_cutoff_aware_gate(self):
        p=read(PLAN)
        panels={a:{e:{o:games(1 if a=='shared' and e=='128' else 0,opponent=o)
            for o in p['opponents']} for e in ['0','128']} for a in p['arms']}
        reference={o:games(0,opponent=o) for o in p['opponents']}
        r=analysis(panels,reference,p)
        self.assertEqual(sum(len(v) for v in r['comparisons'].values()),15)
        self.assertTrue(r['learning']['shared']['passed']);self.assertTrue(r['architecture_benefit']['passed'])
        panels['shared']['128']={o:games(1,False,o) for o in p['opponents']}
        r=analysis(panels,reference,p)
        self.assertFalse(r['learning']['shared']['passed']);self.assertFalse(r['architecture_benefit']['passed'])
    def test_selfplay_both_seats_terminal_value(self):
        g=games(1)[0];g['trace']=[['p1',0,0,.8,'x'],['p2',0,0,.3,'x']]
        out=selfplay_value([g]);self.assertEqual(out['physical_games'],1)
        self.assertEqual(out['seat_views']['terminal_games'],2)
        self.assertAlmostEqual(out['seat_views']['game_mean_mse'],(.04+.09)/2)
        g['outcome']['kind']='administrative_cutoff';out=selfplay_value([g])
        self.assertEqual(out['seat_views']['terminal_games'],0);self.assertIsNone(out['seat_views']['game_mean_mse'])
    def test_heldout_schedule_separate(self):
        p=read(PLAN);a=evaluation_specs(p,7103,'random',0,4);b=diagnostic_specs(p,7103,0,4)
        self.assertNotEqual(a[0]['seed'],b[0]['seed'])
        self.assertEqual(b[0]['opponent'],'self_play');self.assertTrue(all(x['split']=='development' for x in b))
        self.assertEqual([x['learner_side'] for x in b],['p1','p2']*4)

if __name__=='__main__':unittest.main()
