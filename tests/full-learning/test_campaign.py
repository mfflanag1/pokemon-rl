import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
from pokemon_full.policy import initialize
from pokemon_full.learner import parameter_hash
from pokemon_full_train.storage import load_checkpoint,digest
from pokemon_throughput.collector import Pool as PreviousPool
from pokemon_throughput.benchmark import compare_results
from pokemon_full_learning.collector import Pool
from pokemon_full_learning import campaign
from pokemon_full_learning.statistics import compare,paired_bounds,learning_gate
from pokemon_full_learning.verification import audit

ROOT=Path(__file__).resolve().parents[2]


def small_plan():
    plan=json.loads((ROOT/'contracts/full-learning-v1/plan.json').read_text())
    plan.update(profile='test',seeds=[9871],train_games_per_seed=16,
                evaluation_pairs_per_opponent=2,evaluation_chunk_pairs=2,max_turns=2)
    plan['statistics'].update(family_cells=3,resamples=100)
    return plan


def game(episode,swap,reward,terminal=True):
    return {'episode':episode,'swap':swap,'learner_side':'p2' if swap else 'p1','opponent':'base_power',
        'provenance':{'same_team_pair':episode,'seat':swap},'decisions':1,'value_mean':.5,
        'terminal_value_mse':(.5-reward)**2 if terminal else None,
        'outcome':{'kind':'terminal' if terminal else 'administrative_cutoff','rewards':{'p1':reward,'p2':reward}}}


class Statistics(unittest.TestCase):
    def test_cutoffs_remain_intervals_and_pairing_is_strict(self):
        initial=[game(i,s,0) for i in range(4) for s in [False,True]]
        final=[game(i,s,0,False) for i in range(4) for s in [False,True]]
        result=compare(initial,final,small_plan()['statistics'])
        self.assertEqual(result['cutoff_aware_point_interval'],[0,1])
        self.assertEqual(result['gate'],'inconclusive')
        final[0]['provenance']={}
        with self.assertRaises(ValueError): paired_bounds(initial,final)

    def test_cluster_averaging_and_approximate_gate(self):
        initial=[game(i,s,0) for i in range(20) for s in [False,True]]
        final=[game(i,s,float(not s)) for i in range(20) for s in [False,True]]
        settings=small_plan()['statistics']
        result=compare(initial,final,settings)
        self.assertEqual(result['approximate_familywise_bootstrap_interval'],[.5,.5])
        self.assertLess(result['bounded_hoeffding_interval'][0],.5)
        cells={o:copy.deepcopy(result) for o in ['random','base_power','tactical']}
        self.assertTrue(learning_gate([{'seed':1,'evaluation':cells}],settings)['passed'])
        cells['tactical']['gate']='regression'
        self.assertFalse(learning_gate([{'seed':1,'evaluation':cells}],settings)['passed'])

    def test_bootstrap_matches_independently_resampled_pair_means(self):
        initial=[game(i,s,float(i%3==0)) for i in range(16) for s in [False,True]]
        final=[game(i,s,float((i+int(s))%3!=0)) for i in range(16) for s in [False,True]]
        settings=small_plan()['statistics'];settings['resamples']=100
        result=compare(initial,final,settings)
        paired=np.asarray([np.mean([b['outcome']['rewards']['p1']-a['outcome']['rewards']['p1']
            for a,b in zip(initial[2*i:2*i+2],final[2*i:2*i+2])]) for i in range(16)])
        rng=np.random.default_rng(settings['seed'])
        means=np.asarray([paired[rng.integers(0,16,16)].mean() for _ in range(100)])
        tail=.05/(2*settings['family_cells'])
        np.testing.assert_array_equal(result['approximate_familywise_bootstrap_interval'],np.quantile(means,[tail,1-tail]))

    def test_production_budget_cannot_be_changed_in_place(self):
        plan=json.loads((ROOT/'contracts/full-learning-v1/plan.json').read_text())
        campaign.validate_plan(plan)
        plan['train_games_per_seed']=4088
        with self.assertRaisesRegex(ValueError,'registered'): campaign.validate_plan(plan)


class Integration(unittest.TestCase):
    def test_optimized_python_cannot_bypass_audit_checks(self):
        r=subprocess.run([sys.executable,'-O','-c',
            'from pokemon_full_learning.verification import audit; audit("missing", "missing")'],
            capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('requires enabled assertions',r.stderr)

    def test_new_worker_preserves_train_random_and_base_power(self):
        p=initialize(73)
        jobs=[dict(seed=73,episode=i,max_turns=5) for i in range(2)]
        jobs += [dict(seed=8412026,episode=0,max_turns=5,split='development',opponent=o)
                 for o in ['random','base_power']]
        with tempfile.TemporaryDirectory() as d:
            with PreviousPool(Path(d)/'old',2) as old: a,_=old.collect(p,jobs)
            with Pool(Path(d)/'new',2) as new: b,_=new.collect(p,jobs)
            compare_results(a,b)

    def test_tactical_evaluation_has_no_learner_rows_or_actor_access_to_advice(self):
        from pokemon_full_learning import collector
        original=collector.batched_actor
        def isolated(parameters,players,*args):
            self.assertTrue(all('tactical_actions' not in p and 'provenance' not in p for p in players))
            return original(parameters,players,*args)
        jobs=campaign.evaluation_specs(small_plan(),9871,'tactical',0,2)
        with tempfile.TemporaryDirectory() as d,Pool(d,2) as pool,patch.object(collector,'batched_actor',isolated):
            results,_=pool.collect(initialize(73),jobs)
        self.assertTrue(all(not rows for rows,_ in results))
        self.assertTrue(all(r['opponent']=='tactical' and r['decisions']>0 for _,r in results))

    def test_resume_after_committed_cohort_reproduces_original_next_update(self):
        plan=small_plan()
        with tempfile.TemporaryDirectory() as d:
            full=Path(d)/'full';resumed=Path(d)/'resumed'
            expected=campaign.run_campaign(full,plan)
            original=campaign.save_checkpoint
            def interrupt(directory,p,opt,meta):
                result=original(directory,p,opt,meta)
                if meta['cohorts']==1: raise RuntimeError('injected after durable commit')
                return result
            with patch.object(campaign,'save_checkpoint',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'injected'):
                campaign.run_campaign(resumed,plan)
            actual=campaign.run_campaign(resumed,plan,resume=True)
            a,ao,am=load_checkpoint(full/'seed-9871/checkpoints',digest(full/'inputs/manifest.json'))
            b,bo,bm=load_checkpoint(resumed/'seed-9871/checkpoints',digest(resumed/'inputs/manifest.json'))
            self.assertEqual(parameter_hash(a),parameter_hash(b))
            for x,y in zip(ao[:2],bo[:2]): self.assertEqual(parameter_hash(x),parameter_hash(y))
            self.assertEqual(int(ao[2]),int(bo[2]))
            self.assertEqual((bm['next_episode'],bm['cohorts']),(16,2))
            self.assertEqual(actual['seeds'][0]['evaluation'],expected['seeds'][0]['evaluation'])
            self.assertEqual(actual['seeds'][0]['complete_cycle_records'],1)
            # An uncommitted orphan must not confuse exact checkpoint selection.
            directory=full/'seed-9871/checkpoints'
            prior=next(directory.glob('cohort-00001-*.npz'))
            (directory/'cohort-00001-orphan.npz').write_bytes(prior.read_bytes())
            proof=audit(full,Path(d)/'audit')
            self.assertEqual(proof['totals']['training_games'],16)
            self.assertEqual(proof['totals']['evaluation_games'],24)
            self.assertTrue(proof['replays'][0]['original_next_update_exact'])
            evidence=full/'seed-9871/evaluation/final-tactical-0000.json'
            corrupted=json.loads(evidence.read_text());corrupted['games'][0]['trace'][0][2]=99
            evidence.write_text(json.dumps(corrupted))
            with self.assertRaises(AssertionError): audit(full,Path(d)/'negative-audit')
            with self.assertRaisesRegex(ValueError,'Completed'): campaign.run_campaign(resumed,plan,True)


if __name__=='__main__': unittest.main()
