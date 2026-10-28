import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import initialize, objective as original_objective
from pokemon_full.learner import parameter_hash, build_batch
from pokemon_full_train.learning import prepare_batch, objective, optimize, baseline, collect
from pokemon_full_train.storage import save_checkpoint, load_checkpoint, canonical
from pokemon_full_train.campaign import compare, score_interval, validate_plan
from pokemon_full_train.transport import Worker

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('f1_tests', ROOT/'tests/full-showdown/test_learner.py')
f1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f1)


def synthetic():
    rows = [f1.row(.2,0), f1.row(.7,1), f1.row(.3,0), f1.row(.4,1)]
    rows[0]['next_index']=2; rows[1]['next_index']=3
    rows[2].update(terminal=True,reward=1.)
    rows[3].update(terminal=True,reward=0.)
    for i,r in enumerate(rows):
        r['features'][i]=1.
    return rows


def optimizer(parameters):
    zeros = {k:jnp.zeros_like(v) for k,v in parameters.items()}
    return zeros, dict(zeros), jnp.int32(0)


class MathTests(unittest.TestCase):
    def test_numpy_gae_matches_sealed_links_and_bootstraps(self):
        rows=synthetic()
        np.testing.assert_allclose(prepare_batch(rows)['advantages'],build_batch(rows)['advantages'],atol=2e-7)
        rows[2].update(terminal=False,reward=0.,bootstrap=.8)
        rows[3].update(terminal=False,reward=0.,bootstrap=.1)
        np.testing.assert_allclose(prepare_batch(rows)['returns'],build_batch(rows)['returns'],atol=2e-7)
        rows[0]['next_index']=1
        with self.assertRaisesRegex(ValueError,'trajectory'):
            prepare_batch(rows)

    def test_padding_loss_and_gradients_match_compact_f1(self):
        p=initialize(15)
        data=prepare_batch(synthetic())
        compact={k:jnp.asarray(v) for k,v in data.items()}
        padded={k:jnp.asarray(np.pad(v,[(0,252)]+[(0,0)]*(v.ndim-1))) for k,v in data.items()}
        padded['valid']=jnp.arange(256)<4
        # Invalid transport payload must have no effect on objective/gradient.
        padded['features']=padded['features'].at[4:].set(jnp.nan)
        padded['advantages']=padded['advantages'].at[4:].set(jnp.nan)
        a,ga=jax.value_and_grad(lambda x:original_objective(x,compact)[0])(p)
        b,gb=jax.value_and_grad(lambda x:objective(x,padded)[0])(p)
        np.testing.assert_allclose(a,b,atol=2e-6)
        for k in ga:
            np.testing.assert_allclose(ga[k],gb[k],rtol=2e-5,atol=2e-7)

    def test_original_next_update_survives_checkpoint(self):
        p=initialize(17); opt=optimizer(p)
        p,opt,_=optimize(p,opt,synthetic(),21)
        with tempfile.TemporaryDirectory() as d:
            meta={'seed':17,'cohorts':1,'next_episode':8,'input_hash':'abc','cohort_records':[]}
            save_checkpoint(d,p,opt,meta)
            restored,ro,rm=load_checkpoint(d,'abc')
            a,ao,am=optimize(p,opt,synthetic(),22)
            b,bo,bm=optimize(restored,ro,synthetic(),22)
            self.assertEqual(parameter_hash(a),parameter_hash(b))
            for aa,bb in zip(ao[:2],bo[:2]):
                self.assertEqual(parameter_hash(aa),parameter_hash(bb))
            self.assertEqual(int(ao[2]),int(bo[2])); self.assertEqual(am,bm)
            self.assertEqual(rm['next_episode'],8)
            with self.assertRaisesRegex(ValueError,'binding'):
                load_checkpoint(d,'changed')
            pointer=json.loads((Path(d)/'latest.json').read_text())
            with open(Path(d)/pointer['file'],'ab') as out:
                out.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'digest'):
                load_checkpoint(d,'abc')


class EvaluationTests(unittest.TestCase):
    def test_baselines_use_only_legal_projection_actions(self):
        p=f1.player(); p['mask']=[False]*14; p['mask'][0]=p['mask'][4]=True
        rng=np.random.default_rng(12)
        self.assertEqual(baseline(p,'base_power',rng),0)
        self.assertTrue(all(baseline(p,'random',rng) in [0,4] for _ in range(30)))
        p['mask']=[False]*14; p['mask'][12]=True
        self.assertEqual(baseline(p,'base_power',rng),12)
        p['hidden_state']='forbidden'
        with self.assertRaisesRegex(ValueError,'projection'):
            baseline(p,'random',rng)

    def test_cutoffs_are_unknown_not_draws_and_pairing_is_enforced(self):
        def record(i,terminal,reward):
            return {'episode':i//2,'swap':bool(i%2),'learner_side':'p2' if i%2 else 'p1',
                'opponent':'random','provenance':{'same':i},'decisions':10,'value_mean':.5,
                'terminal_value_mse':.25 if terminal else None,
                'outcome':{'kind':'terminal' if terminal else 'administrative_cutoff',
                           'rewards':{'p1':reward,'p2':reward}}}
        initial=[record(i,True,0.) for i in range(128)]
        final=[record(i,False,0.) for i in range(128)]
        result=compare(initial,final,6)
        self.assertEqual(score_interval(final[0]),(0.,1.))
        self.assertEqual(result['paired_improvement_interval'],[0.,1.])
        self.assertEqual(result['gate'],'inconclusive')
        final=[record(i,True,1.) for i in range(128)]
        self.assertEqual(compare(initial,final,6)['gate'],'improvement_established')
        final[0]['provenance']={'other':1}
        with self.assertRaisesRegex(ValueError,'identity'):
            compare(initial,final,6)

    def test_preregistered_plan_is_admitted_but_not_final_or_new_ppo(self):
        plan=json.loads((ROOT/'contracts/full-training-v1/plan.json').read_text())
        validate_plan(plan)
        plan['ppo']['learning_rate']=.5
        with self.assertRaises(ValueError): validate_plan(plan)


class Integration(unittest.TestCase):
    def test_train_links_and_development_never_produce_update_rows(self):
        with tempfile.TemporaryDirectory() as d, Worker(d) as worker:
            rows,report=collect(worker,initialize(73),seed=73,episode=0,max_turns=2)
            self.assertGreaterEqual(len(rows),2)
            self.assertEqual(report['outcome']['kind'],'administrative_cutoff')
            self.assertTrue(all(not r['terminal'] and r['reward']==0 for r in rows))
            np.testing.assert_allclose(prepare_batch(rows)['returns'],.5,atol=1e-6)
            rows,report=collect(worker,initialize(73),seed=8412026,episode=0,max_turns=2,
                split='development',opponent='random')
            self.assertEqual(rows,[])
            self.assertEqual({t['partition'] for t in report['provenance']['teams']},{'development'})

    def test_worker_failure_retains_command_and_last_response(self):
        with tempfile.TemporaryDirectory() as d:
            with Worker(d) as worker:
                with self.assertRaises(RuntimeError):
                    worker.request('start',run_seed=73,episode_index=0,split='final',swap=False)
            evidence=list(Path(d).glob('failure-*.json'))
            self.assertEqual(len(evidence),1)
            body=json.loads(evidence[0].read_text())
            self.assertEqual(body['command']['split'],'final')
            self.assertEqual(body['previous_response']['schema_version'],'full-training-worker-v1')


if __name__=='__main__': unittest.main()

class UnavailableChoice(unittest.TestCase):
    def test_actual_reference_rejection_is_a_retained_on_policy_transition(self):
        import subprocess
        from pokemon_full.features import encode
        result=subprocess.run(['node','tests/full-showdown/trapped-fixture.mjs'],cwd=ROOT,
            capture_output=True,text=True,check=True,timeout=15)
        tape=json.loads(result.stdout)
        commands=[]
        outer=self
        class Tape:
            def request(self,op,**args):
                commands.append(op)
                if op=='start': return copy.deepcopy(tape['initial'])
                if op=='step':
                    step=commands.count('step')-1
                    outer.assertEqual(args,tape['commands'][step])
                    return copy.deepcopy(tape[['rejected','continued'][step]])
                outer.assertEqual(op,'cutoff')
                return copy.deepcopy(tape['cutoff'])
        choices=iter([9,0,0,-1,0,0])
        def scripted(parameters,player,key):
            features,mask=encode(player)
            action=next(choices)
            self.assertTrue(mask[action] if action>=0 else not mask.any())
            return features,mask,action,-float(np.log(max(mask.sum(),1))),.5
        with patch('pokemon_full_train.learning.actor',side_effect=scripted):
            rows,report=collect(Tape(),{},seed=73,episode=0,max_turns=2)
        self.assertEqual(commands,['start','step','step','cutoff'])
        self.assertEqual(report['unavailable_choices'],1)
        self.assertEqual([r['action'] for r in rows],[9,0,0])
        self.assertEqual([r['next_index'] for r in rows],[2,-1,-1])
        self.assertTrue(rows[0]['mask'][9]); self.assertFalse(rows[2]['mask'][9])
        self.assertTrue(all(not r['terminal'] and r['reward']==0 for r in rows))
        np.testing.assert_allclose(prepare_batch(rows)['returns'],.5,atol=1e-7)
