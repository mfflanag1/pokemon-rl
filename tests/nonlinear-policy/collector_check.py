import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pokemon_full.policy import initialize
from pokemon_matchup.network import extend as old_extend
from pokemon_nonlinear_policy.network import extend
from pokemon_opponent_mix.collector import Pool as OldPool
from pokemon_nonlinear_policy.collector import Pool
from pokemon_opponent_mix.campaign import collect_update as previous_collect
from pokemon_nonlinear_policy.campaign import collect_update,training_regimes,training_specs
from pokemon_nonlinear_policy.common import read,ROOT,PLAN,budget,parameter_hash
from pokemon_fit.data import fingerprint
from pokemon_full_train.storage import canonical

class Collector(unittest.TestCase):
    def test_exact_stream_budgets_and_cap_failure(self):
        class Empty:
            def __init__(self):self.calls=0
            def collect(self,p,jobs):self.calls+=1;return [],{}
        for arm,quota,cap in [('control',512,64),('nonlinear',512,64)]:
            self.assertEqual(training_regimes(arm),[(None,quota),('tactical',quota)])
            pool=Empty()
            with patch('pokemon_nonlinear_policy.campaign.check_collection'):
                with self.assertRaisesRegex(ValueError,'cap exhausted'):collect_update(pool,None,read(PLAN),1,0,0,arm)
            self.assertEqual(pool.calls,cap//8)
    def test_small_frozen_collector_matches_previous_on_identical_streams(self):
        # Compare complete game tapes, then account for intentionally fresh row-selection seeds.
        old,_=old_extend(initialize(45));p,o=extend(old,45);before=parameter_hash(p)
        plan={**read(PLAN),'max_turns':20,'training_seed_namespace':'f29-tests','maximum_games_per_update':64}
        with tempfile.TemporaryDirectory() as tmp:
            with Pool(Path(tmp)/'new',8,'control',read(ROOT/plan['catalog'])) as pool:
                b,a,_=collect_update(pool,p,plan,45,0,0,'control')
            with OldPool(Path(tmp)/'old',8,'matchup',read(ROOT/plan['catalog'])) as pool:
                c,d,_=previous_collect(pool,old,plan,45,0,0,'mixed')
            with Pool(Path(tmp)/'nonlinear',8,'nonlinear',read(ROOT/plan['catalog'])) as pool:
                _,e,_=collect_update(pool,p,plan,45,0,0,'nonlinear')
            self.assertEqual(canonical(a['games']),canonical(e['games']))
        self.assertEqual(canonical(a['games']),canonical(d['games']))
        self.assertEqual(parameter_hash(p),before)
        self.assertEqual(a['selection']['selected_rows'],1024)
        self.assertEqual([s['selection']['selected_rows'] for s in a['strata']],[512,512])
        self.assertNotEqual(fingerprint(b),fingerprint(c))
        self.assertEqual(a['strata'][0]['next_episode'],a['strata'][1]['first_episode'])
        self.assertEqual(a['next_episode'],len(a['games']))
    def test_selection_receives_complete_games_before_quota_and_arm_is_frozen(self):
        # Each chunk overshoots quota; GAE/selection must see the full last chunk, never a row prefix.
        seen=[];token=object()
        class Fake:
            def collect(self,p,jobs):
                self.p=p;self.jobs=jobs
                return [([{}]*200,{'id':j['episode']}) for j in jobs],{}
        def select(collection,seed,index,quota):
            self.assertEqual(sum(len(r) for r,g in collection),1600)
            seen.append((quota,len(collection)))
            return {'actions':np.zeros(quota,np.int32)},dict(selected_rows=quota,collected_rows=sum(len(r) for r,g in collection))
        with patch('pokemon_nonlinear_policy.campaign.check_collection'),patch('pokemon_nonlinear_policy.campaign.select',select):
            for arm in ['control','nonlinear']:
                pool=Fake();b,l,_=collect_update(pool,token,read(PLAN),1,0,0,arm)
                self.assertIs(pool.p,token);self.assertEqual(len(b['actions']),budget(arm)['selected_rows'])
        self.assertEqual(seen,[(512,8),(512,8),(512,8),(512,8)])
if __name__=='__main__':unittest.main()
