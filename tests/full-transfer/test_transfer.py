import copy
import unittest
from unittest.mock import patch
import numpy as np
from pokemon_transfer.common import select,collect_update,retention,specs
from pokemon_full_train.learning import prepare_batch
from pokemon_throughput.collector import merge_rows


class Transfer(unittest.TestCase):
    def test_gae_precedes_selection_and_includes_unselected_successors(self):
        rows=[]
        for i in range(1030):
            rows.append({'features':np.full(928,i/1030,np.float32),'mask':np.ones(14,bool),'action':0,'old_logp':-1.,'value':.3,
                'reward':1. if i==1029 else 0.,'next_index':i+1 if i<1029 else -1,'bootstrap':0.,'terminal':i==1029,'trajectory':0})
        collection=[(rows,{'decisions':1030})];batch,ledger=select(collection,22,0)
        complete=prepare_batch(merge_rows(collection));ids=ledger['selected_indices']
        np.testing.assert_array_equal(batch['returns'],complete['returns'][ids])
        self.assertEqual(len(set(ids)),1024);self.assertEqual(ledger['excluded_rows'],6)
        repeat,again=select(collection,22,0);self.assertEqual(ledger,again)
        np.testing.assert_array_equal(batch['features'],repeat['features'])
        with self.assertRaises(ValueError):select([(rows[:1],{})],22,0)

    def test_multiple_games_keep_separate_gae_chains(self):
        collection=[];expected=[]
        for reward in [1.,0.]:
            rows=[]
            for i in range(520):
                rows.append({'features':np.full(928,i/520,np.float32),'mask':np.ones(14,bool),'action':0,'old_logp':-1.,'value':.3,
                    'reward':reward if i==519 else 0.,'next_index':i+1 if i<519 else -1,'bootstrap':0.,'terminal':i==519,'trajectory':0})
            collection.append((rows,{'decisions':520}));expected.append(prepare_batch(rows)['returns'])
        selected,ledger=select(collection,2,0)
        np.testing.assert_array_equal(selected['returns'],np.concatenate(expected)[ledger['selected_indices']])
        merged=merge_rows(collection)
        self.assertNotEqual(merged[0]['trajectory'],merged[520]['trajectory'])
        self.assertEqual(merged[520]['next_index'],521);self.assertEqual(merged[519]['next_index'],-1)

    def test_training_namespaces_pair_arms_but_change_seed(self):
        plan={'training_seed_namespace':'f9-full-transfer-training','max_turns':120,'training_chunk_games':8}
        a=specs(plan,7103,8);self.assertEqual([s['episode'] for s in a],list(range(8,16)))
        self.assertTrue(all(s['split']=='train' and s['opponent'] is None and not s['swap'] for s in a))
        self.assertNotEqual(a,specs(plan,8209,8))

    def test_cap_fails_instead_of_resampling(self):
        class Pool:
            def __init__(self):self.n=0
            def collect(self,p,jobs):
                self.n+=1
                return [([{}],{'decisions':1,'unavailable_choices':0}) for _ in jobs],{}
        pool=Pool();plan={'training_seed_namespace':'cap-test','max_turns':120,'training_chunk_games':8,'maximum_games_per_update':64,'selected_rows_per_update':1024}
        with patch('pokemon_transfer.common.check_games'):
            with self.assertRaisesRegex(ValueError,'cap exhausted'):collect_update(pool,None,plan,1,0,0)
        self.assertEqual(pool.n,8)

    def test_retention_requires_each_context(self):
        cells={f'{p}/{c}':{'n':16,'target_mass':.9,'greedy_accuracy':1.} for p in ['train','variant'] for c in range(16)}
        initial={'cells':cells};final=copy.deepcopy(initial);settings={'minimum_greedy_accuracy':.95,'minimum_context_mass':.6,'maximum_context_mass_drop':.05}
        self.assertTrue(retention(initial,final,settings)['passed'])
        final['cells']['variant/7']['target_mass']=.84
        self.assertFalse(retention(initial,final,settings)['passed'])
        final=copy.deepcopy(initial);final['cells'].pop('train/0')
        with self.assertRaises(ValueError):retention(initial,final,settings)

    def test_greedy_failure_and_mass_floor(self):
        initial={'cells':{f'{p}/{c}':{'n':16,'target_mass':.6,'greedy_accuracy':1.} for p in ['train','variant'] for c in range(16)}}
        settings={'minimum_greedy_accuracy':.95,'minimum_context_mass':.6,'maximum_context_mass_drop':.05}
        for field,val in [('greedy_accuracy',.9),('target_mass',.59),('n',15)]:
            final=copy.deepcopy(initial);final['cells']['train/0'][field]=val
            if field=='n':
                with self.assertRaises(ValueError):retention(initial,final,settings)
            else:self.assertFalse(retention(initial,final,settings)['passed'])


class Summaries(unittest.TestCase):
    def fixture(self):
        from pokemon_transfer.common import read
        from pokemon_full_train.storage import ROOT
        plan=read(ROOT/'contracts/full-transfer-v1/plan.json')
        def games(score,cutoff=False):
            result=[]
            for episode in range(32):
                for swap in [False,True]:
                    side='p2' if swap else 'p1';rewards={side:score,('p1' if swap else 'p2'):1-score}
                    result.append({'episode':episode,'swap':swap,'learner_side':side,'opponent':'random','provenance':{'pair':episode},
                        'outcome':{'kind':'administrative_cutoff' if cutoff else 'terminal','rewards':None if cutoff else rewards},
                        'decisions':10,'terminal_value_mse':None if cutoff else .25,'value_mean':.5})
            return result
        def panel(score,cutoff=False):
            result={}
            for o in plan['opponents']:
                records=games(score,cutoff)
                for g in records:g['opponent']=o
                result[o]=records
            return result
        probe={'cells':{f'{p}/{c}':{'n':16,'target_mass':.9,'greedy_accuracy':1.} for p in ['train','variant'] for c in range(16)}}
        weak=copy.deepcopy(probe)
        for cell in weak['cells'].values():cell['target_mass']=.7
        return plan,panel,probe,weak
    def test_separate_benefit_and_learning_gates(self):
        from pokemon_transfer.common import summarize
        plan,panel,probe,weak=self.fixture()
        result=summarize(panel(0),{'plain':panel(0),'rehearsal':panel(1)},probe,{'plain':weak,'rehearsal':probe},plan)
        self.assertTrue(result['rehearsal_benefit_passed']);self.assertTrue(result['full_battle_learning']['rehearsal']['passed'])
        self.assertFalse(result['full_battle_learning']['plain']['passed']);self.assertFalse(result['retention']['plain']['passed'])
        self.assertEqual(len(result['comparisons']),3)
    def test_cutoffs_cannot_become_wins(self):
        from pokemon_transfer.common import summarize
        plan,panel,probe,weak=self.fixture()
        result=summarize(panel(0),{'plain':panel(0),'rehearsal':panel(1,True)},probe,{'plain':weak,'rehearsal':probe},plan)
        cell=result['comparisons']['rehearsal_minus_initial']['base_power']
        self.assertEqual(cell['cutoff_aware_point_interval'],[0.,1.]);self.assertEqual(cell['final']['learner_wins'],0)
        self.assertIsNone(cell['final']['terminal_game_mean_value_mse']);self.assertFalse(result['full_battle_learning']['rehearsal']['passed'])

if __name__=='__main__':unittest.main()
