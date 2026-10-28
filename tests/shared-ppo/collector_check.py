import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import canonical
from pokemon_full_train.learning import prepare_batch,derived_seed
from pokemon_transfer_v2.collector import Pool as OldPool
from pokemon_shared.learning import initial
from pokemon_shared_ppo.collector import Pool,AuditedEpisode
from pokemon_shared_ppo.common import (read,PLAN,check_collection,diagnostic_specs,
    save,load,same_state,selfplay_value)
from pokemon_shared_ppo.campaign import collect_update
from pokemon_shared_ppo.learning import optimize_batch
from pokemon_throughput.collector import merge_rows

class Collector(unittest.TestCase):
    def test_flat_training_rows_and_games_match_sealed(self):
        p,_=initial(17,'flat');jobs=[dict(seed=1025262299,episode=i,max_turns=4,split='train') for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            with OldPool(d/'old',2) as pool:a,_=pool.collect(p,jobs)
            with Pool(d/'new',2,'flat') as pool:b,_=pool.collect(p,jobs)
        check_collection(b,0,4,'train')
        for (_,g),(_,h) in zip(a,b):
            q=dict(h);q.pop('actor_trace');self.assertEqual(canonical(g),canonical(q))
        aa=prepare_batch(merge_rows(a));bb=prepare_batch(merge_rows(b))
        for k in aa:np.testing.assert_array_equal(aa[k],bb[k])
    def test_selfplay_development_has_no_learner_rows_and_is_reproducible(self):
        p,_=initial(18,'shared');plan={**read(PLAN),'max_turns':4,'selfplay_evaluation_seed':1025262298}
        jobs=diagnostic_specs(plan,18,0,2)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            with Pool(d/'a',2,'shared') as pool:a,_=pool.collect(p,jobs)
            with Pool(d/'b',2,'shared') as pool:b,_=pool.collect(p,jobs)
        check_collection(a,0,2,'development');check_collection(b,0,2,'development')
        self.assertEqual(canonical([g for _,g in a]),canonical([g for _,g in b]))
        self.assertTrue(all(not rows for rows,_ in a))
        self.assertTrue(all(g['decisions']==len(g['actor_trace']) for _,g in a))
        self.assertEqual(selfplay_value([g for _,g in a])['seat_views']['cutoff_games'],8)
    def test_roles_and_bad_likelihood_fail_closed(self):
        with self.assertRaises(ValueError):AuditedEpisode(seed=1,episode=0,max_turns=2,split='train',opponent='self_play')
        p,_=initial(19,'flat')
        with tempfile.TemporaryDirectory() as directory:
            with Pool(Path(directory),1,'flat') as pool:
                a=dict(seed=2,episode=0,max_turns=2,split='train')
                b=dict(seed=2,episode=0,max_turns=2,split='development',opponent='random')
                with self.assertRaises(ValueError):pool.collect(p,[a,b])
                records,_=pool.collect(p,[a])
        check_collection(records,0,1,'train');bad=copy.deepcopy(records)
        bad[0][1]['actor_trace'][0]['log_probability']=float('nan')
        with self.assertRaises(ValueError):check_collection(bad,0,1,'train')
    def test_actual_collect_update_checkpoint_recovery_both_architectures(self):
        plan={**read(PLAN),'max_turns':20,'training_seed_namespace':'f22-engineering-preflight'}
        for arm in ['flat','shared']:
            p,o=initial(20,arm)
            with tempfile.TemporaryDirectory() as directory:
                d=Path(directory)
                with Pool(d/'workers',8,arm) as pool:batch,ledger,_=collect_update(pool,p,plan,20,0,0)
                self.assertEqual(len(batch['actions']),1024)
                self.assertGreaterEqual(ledger['selection']['collected_rows'],1024)
                save(d/'checkpoint',p,o,'h',20,arm,0,0);q,qo=load(d/'checkpoint','h',20,arm,0,0)
                p,o,metrics=optimize_batch(p,o,batch,71,arm)
                q,qo,other=optimize_batch(q,qo,batch,71,arm)
                same_state(p,o,q,qo);self.assertEqual(metrics,other);self.assertEqual(int(o[2]),8)
                save(d/'after',p,o,'h',20,arm,1,ledger['next_episode'])
                q,qo=load(d/'after','h',20,arm,1,ledger['next_episode']);same_state(p,o,q,qo)

if __name__=='__main__':unittest.main()
