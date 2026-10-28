import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_full.policy import initialize
from pokemon_matchup.network import extend
from pokemon_full_train.learning import prepare_batch
from pokemon_throughput.collector import merge_rows
from pokemon_full_train.storage import canonical
from pokemon_opponent_mix.collector import Pool,AuditedEpisode
from pokemon_opponent_mix.common import read,ROOT,PLAN,check_collection,save,load,same_state
from pokemon_opponent_mix.campaign import collect_update,training_specs,training_regimes
from pokemon_matchup.learning import optimize_batch
from pokemon_terminal_matchup.collector import Pool as PreviousPool
CAT=read(ROOT/read(PLAN)['catalog'])

def outputs():
    return [(np.full(1278,i,np.float32),np.array([True,True]+[False]*12),0,-.7,.2+i*.5) for i in range(2)]
def frame(outcome=None):
    return {'outcome':outcome,'players':{s:{'phase':'move','turn':1,'request_id':3} for s in ['p1','p2']},'tactical_actions':{'p1':1,'p2':1}}

class Collector(unittest.TestCase):
    def test_only_learner_rows_and_true_sample_likelihood(self):
        for seat in ['p1','p2']:
            ep=AuditedEpisode(seed=1,episode=0,max_turns=4,split='train',opponent='tactical',learner_side=seat)
            ep.frame=frame();ep.advance(outputs());ep.frame=frame();ep.advance(outputs())
            self.assertEqual(len(ep.rows),2);self.assertEqual(len(ep.actor_trace),2)
            self.assertEqual([r['action'] for r in ep.rows],[0,0]);self.assertEqual([r['old_logp'] for r in ep.rows],[-.7,-.7])
            self.assertEqual(ep.rows[0]['next_index'],1);self.assertIsNone(ep.last['p2' if seat=='p1' else 'p1'])
            self.assertTrue(all(r['trajectory']==['p1','p2'].index(seat) for r in ep.rows))
            self.assertEqual([r[2] for r in ep.trace if r[0]!=seat],[1,1])
    def test_p2_cutoff_bootstrap_and_terminal_reward(self):
        for terminal in [False,True]:
            ep=AuditedEpisode(seed=1,episode=1,max_turns=4,split='train',opponent='tactical',learner_side='p2')
            ep.frame=frame();ep.advance(outputs());ep.advance(outputs())
            ep.frame=frame({'kind':'terminal' if terminal else 'administrative_cutoff','rewards':{'p1':0.,'p2':1.} if terminal else None})
            ep.finish([.1,.8]);b=prepare_batch(ep.rows)
            self.assertAlmostEqual(ep.rows[-1]['bootstrap'],0. if terminal else .8)
            self.assertAlmostEqual(ep.rows[-1]['reward'],1. if terminal else 0.)
            expected_last=(1. if terminal else .8)-.7
            self.assertAlmostEqual(float(b['advantages'][-1]),expected_last,places=6)
            self.assertAlmostEqual(float(b['advantages'][0]),.95*expected_last,places=6)
    def test_invalid_advice_fails_without_opponent_row(self):
        for action in [-1,14,None]:
            ep=AuditedEpisode(seed=1,episode=0,max_turns=4,split='train',opponent='tactical',learner_side='p2')
            ep.frame=frame();ep.frame['tactical_actions']['p1']=action
            with self.assertRaises(ValueError):ep.advance(outputs())
            self.assertFalse(ep.rows)
    def test_schedule_exact_row_mixture_and_seats(self):
        p=read(PLAN)
        self.assertEqual(training_regimes('mixed'),[(None,512),('tactical',512)])
        self.assertEqual(training_regimes('self'),[(None,1024)])
        jobs=training_specs(p,7103,8,'tactical')
        self.assertEqual([s['learner_side'] for s in jobs],['p1','p2']*4)
        self.assertTrue(all(s['split']=='train' and not s['swap'] for s in jobs))
        with self.assertRaises(ValueError):training_regimes('best')
    def test_live_control_matches_sealed_collector_and_development(self):
        p,_=extend(initialize(19));plan={**read(PLAN),'max_turns':4,'training_seed_namespace':'f26-tests'}
        jobs=training_specs(plan,19,0,None)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            with PreviousPool(d/'old',2,'matchup',CAT) as pool:a,_=pool.collect(p,jobs)
            with Pool(d/'new',2,'matchup',CAT) as pool:b,_=pool.collect(p,jobs)
            self.assertEqual(canonical([g for _,g in a]),canonical([g for _,g in b]))
            for (aa,_),(bb,_) in zip(a,b):
                for x,y in zip(aa,bb):
                    for k in x:np.testing.assert_array_equal(x[k],y[k])
            dev=[{**s,'split':'development','opponent':'tactical'} for s in jobs[:2]]
            with Pool(d/'dev',2,'matchup',CAT) as pool:c,_=pool.collect(p,dev)
            self.assertTrue(all(not rows for rows,_ in c))
    def test_live_tactical_both_seats_gae_and_corrupt_row(self):
        p,_=extend(initialize(20));plan={**read(PLAN),'max_turns':4,'training_seed_namespace':'f26-tests'}
        with tempfile.TemporaryDirectory() as directory:
            with Pool(Path(directory),2,'matchup',CAT) as pool:
                c,_=pool.collect(p,training_specs(plan,20,0,'tactical'))
        check_collection(c,0,8,'train','tactical')
        merged=merge_rows(c);batch=prepare_batch(merged)
        expected=np.zeros(len(merged),np.float32)
        # Independent chain traversal, including one-seat trajectories and cutoffs.
        for start,r in enumerate(merged):
            weight=1.;i=start
            while i>=0:
                q=merged[i];j=q['next_index']
                nv=0. if q['terminal'] else merged[j]['value'] if j>=0 else q['bootstrap']
                expected[start]+=np.float32(weight*(q['reward']+nv-q['value']))
                weight*=.95;i=j
        np.testing.assert_allclose(expected,batch['advantages'],atol=2e-6)
        bad=copy.deepcopy(c);bad[0][0][0]['action']=(bad[0][0][0]['action']+1)%14
        with self.assertRaises(ValueError):check_collection(bad,0,8,'train','tactical')
    def test_complete_tactical_collect_update_recovery(self):
        plan={**read(PLAN),'max_turns':20,'training_seed_namespace':'f26-checkpoint-test'}
        p,o=extend(initialize(21))
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            with Pool(d/'workers',8,'matchup',CAT) as pool:b,ledger,_=collect_update(pool,p,plan,21,1,0,'mixed')
            self.assertEqual(len(b['actions']),1024);self.assertEqual([s['selection']['selected_rows'] for s in ledger['strata']],[512,512])
            self.assertEqual({g['opponent'] for g in ledger['games']},{None,'tactical'})
            save(d/'before',p,o,'h',21,'mixed',0,0);q,qo=load(d/'before','h',21,'mixed',0,0)
            p,o,m=optimize_batch(p,o,b,71,'matchup');q,qo,n=optimize_batch(q,qo,b,71,'matchup')
            same_state(p,o,q,qo);self.assertEqual(m,n)
            save(d/'after',p,o,'h',21,'mixed',1,ledger['next_episode'])
            q,qo=load(d/'after','h',21,'mixed',1,ledger['next_episode']);same_state(p,o,q,qo)

if __name__=='__main__':unittest.main()
