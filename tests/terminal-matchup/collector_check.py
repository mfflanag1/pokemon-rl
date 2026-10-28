import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_full.policy import initialize
from unittest.mock import patch
from pokemon_throughput.collector import batched_actor as sealed_actor
from pokemon_terminal_matchup.features import legacy_player, encode
from pokemon_terminal_matchup.collector import Pool,AuditedEpisode
from pokemon_matchup.network import extend
from pokemon_terminal_matchup.actor import batched_actor
from pokemon_terminal_matchup.common import read,ROOT,PLAN,check_collection,diagnostic_specs,save,load,same_state
from pokemon_terminal_matchup.campaign import collect_update
from pokemon_matchup.learning import optimize_batch
from pokemon_full_train.learning import prepare_batch
from pokemon_throughput.collector import merge_rows
from pokemon_full_train.storage import canonical

CAT=read(ROOT/read(PLAN)['catalog'])

class Collector(unittest.TestCase):
    def test_initial_corrected_view_arms_match_sealed_base_actor(self):
        p=initialize(17);q,_=extend(p);jobs=[dict(seed=1025262599,episode=i,max_turns=4,split='train') for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            def reference(parameters, players, seeds, counters, capacity, arm, catalog):
                rows=sealed_actor(p,[legacy_player(player) for player in players],seeds,counters,capacity)
                return [(encode(player,catalog,arm)[0],mask,action,logp,value)
                        for player,(_,mask,action,logp,value) in zip(players,rows)]
            with patch('pokemon_terminal_matchup.collector.batched_actor',reference):
                with Pool(d/'reference',2,'zero',CAT) as pool:old,_=pool.collect(q,jobs)
            for arm in ['zero','matchup']:
                with Pool(d/arm,2,arm,CAT) as pool:new,_=pool.collect(q,jobs)
                check_collection(new,0,4,'train')
                for (ar,ag),(br,bg) in zip(old,new):
                    self.assertEqual(ag['outcome'],bg['outcome'])
                    self.assertEqual([t[:4] for t in ag['trace']],[t[:4] for t in bg['trace']])
                    self.assertEqual([a['log_probability'] for a in ag['actor_trace']],[a['log_probability'] for a in bg['actor_trace']])
                    for a,b in zip(ar,br):np.testing.assert_array_equal(a['features'][:928],b['features'][:928])
                b=prepare_batch(merge_rows(new));self.assertEqual(b['features'].shape[1],1278)
                self.assertEqual(np.any(b['features'][:,928:]!=0),arm=='matchup')

    def test_development_selfplay_no_rows_and_deterministic(self):
        p,_=extend(initialize(18));plan={**read(PLAN),'max_turns':4,'selfplay_evaluation_seed':1025262598}
        jobs=diagnostic_specs(plan,18,0,2)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            with Pool(d/'a',2,'matchup',CAT) as pool:a,_=pool.collect(p,jobs)
            with Pool(d/'b',2,'matchup',CAT) as pool:b,_=pool.collect(p,jobs)
        check_collection(a,0,2,'development');self.assertTrue(all(not rows for rows,_ in a))
        self.assertEqual(canonical([g for _,g in a]),canonical([g for _,g in b]))
        self.assertTrue(all(g['decisions']==len(g['actor_trace']) for _,g in a))

    def test_roles_and_player_boundary(self):
        with self.assertRaises(ValueError):AuditedEpisode(seed=1,episode=0,max_turns=2,split='train',opponent='self_play')
        p,_=extend(initialize(19))
        with tempfile.TemporaryDirectory() as directory:
            with Pool(Path(directory),1,'zero',CAT) as pool:
                a=dict(seed=3,episode=0,max_turns=2,split='train');b={**a,'split':'development','opponent':'random'}
                with self.assertRaises(ValueError):pool.collect(p,[a,b])
                result,_=pool.collect(p,[a])
        check_collection(result,0,1,'train');bad=copy.deepcopy(result)
        bad[0][1]['actor_trace'][0]['log_probability']=float('nan')
        with self.assertRaises(ValueError):check_collection(bad,0,1,'train')
        with self.assertRaises(ValueError):batched_actor(p,[{'players':{},'provenance':{}}],[1],[0],2,'matchup',CAT)

    def test_collect_update_and_next_update_checkpoint(self):
        plan={**read(PLAN),'max_turns':20,'training_seed_namespace':'f25-engineering-preflight'}
        for arm in ['zero','matchup']:
            p,o=extend(initialize(20))
            with tempfile.TemporaryDirectory() as directory:
                d=Path(directory)
                with Pool(d/'workers',8,arm,CAT) as pool:b,ledger,_=collect_update(pool,p,plan,20,0,0)
                self.assertEqual(len(b['actions']),1024);save(d/'checkpoint',p,o,'h',20,arm,0,0)
                q,qo=load(d/'checkpoint','h',20,arm,0,0)
                p,o,m=optimize_batch(p,o,b,71,arm);q,qo,n=optimize_batch(q,qo,b,71,arm)
                same_state(p,o,q,qo);self.assertEqual(m,n)
                save(d/'after',p,o,'h',20,arm,1,ledger['next_episode'])
                q,qo=load(d/'after','h',20,arm,1,ledger['next_episode']);same_state(p,o,q,qo)

if __name__=='__main__':unittest.main()
