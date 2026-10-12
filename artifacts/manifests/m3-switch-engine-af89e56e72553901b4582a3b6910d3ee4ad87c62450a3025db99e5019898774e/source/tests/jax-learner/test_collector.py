import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_jax.engine import initial,view_kernel,Chance,step
from pokemon_learner.core import linked_gae as reference_gae
from pokemon_jax_learner.model import ActorInput,initialize,encode,act,predict
from pokemon_jax_learner.collector import collect,policy_hash,validate_collection
from pokemon_jax_learner.trainer import Trainer
from pokemon_jax_learner.checkpoint import state,save,load,digest

PLAN=json.loads((Path(__file__).resolve().parents[2]/'contracts/jax-learner-v1/plan.json').read_text())
SMALL={**PLAN,'games_per_collection':8,'epochs':1,'minibatch_size':32}


class Collector(unittest.TestCase):
    def test_actor_only_accepts_observable_carrier_and_no_cross_row_attention(self):
        params=initialize(4);obs=encode(view_kernel(initial(2),0));keys=jax.random.split(jax.random.key(9,impl='threefry2x32'),2)
        with self.assertRaises(ValueError):act(params,initial(2),keys)
        with self.assertRaises(ValueError):act(params,{'features':obs.features,'mask':obs.mask,'winner':0},keys)
        changed=obs._replace(features=obs.features.at[1,:].set(17))
        a,b=predict(params,obs),predict(params,changed)
        for x,y in zip(a,b):np.testing.assert_array_equal(x[0],y[0])

    def test_masks_and_terminal_rows_are_not_actions(self):
        params=initialize(4);obs=encode(view_kernel(initial(3),0))
        obs=obs._replace(mask=jnp.array([[True,False],[False,True],[False,False]]))
        actions,logp,values=act(params,obs,jax.random.split(jax.random.key(9,impl='threefry2x32'),3))
        self.assertEqual(actions.tolist(),[0,1,-1]);self.assertEqual(logp.tolist(),[0,0,0]);self.assertEqual(float(values[2]),0)
        np.testing.assert_array_equal(predict(params,obs)[0],[[1,0],[0,1],[0,0]])

    def test_whole_episode_linkage_scores_and_targets_match_numpy(self):
        b=collect(initialize(5),0,np.arange(32),environment_seed=1,action_seed=2)
        n=validate_collection(b);a,t=b.actor,b.targets
        self.assertEqual(set(f.name for f in fields(a)),{'features','mask','actions','old_logp','old_values','game_id','side','turn'})
        self.assertTrue(np.all(a.features[:,1:3]==0))  # no terminal winner targets in actor rows
        self.assertEqual(int(t.terminal.sum()),64);self.assertEqual(float(t.rewards.sum()),32)
        self.assertEqual(n%2,0)
        for game in range(32):
            rows=np.where(a.game_id==game)[0];self.assertEqual(float(t.rewards[rows].sum()),1)
            for side in (0,1):
                ix=rows[a.side[rows]==side]
                np.testing.assert_array_equal(a.turn[ix],np.arange(1,len(ix)+1))
                np.testing.assert_array_equal(t.next_index[ix],list(ix[1:])+[-1])
        adv,returns=reference_gae(t.rewards,a.old_values,t.next_index,t.bootstrap,t.terminal,t.trajectory,gamma=1,lam=.95)
        np.testing.assert_allclose(t.advantages,adv,atol=3e-6,rtol=3e-5);np.testing.assert_allclose(t.returns,returns,atol=3e-6,rtol=3e-5)
        for record in (a,t):
            self.assertTrue(all(not getattr(record,f.name).flags.writeable for f in fields(record)))

    def test_frozen_policy_and_environment_secret_separation(self):
        parameters=initialize(12);before=policy_hash(parameters);seen=[]
        def spy(p,obs,keys):
            self.assertIs(type(obs),ActorInput)
            seen.append((np.asarray(obs.features).copy(),np.asarray(predict(p,obs)[0]).copy()))
            return act(p,obs,keys)
        first=collect(parameters,0,np.arange(8),environment_seed=10,action_seed=11,actor_fn=spy)
        initial_seen=seen[0];seen.clear()
        second=collect(parameters,0,np.arange(8),environment_seed=20,action_seed=11,actor_fn=spy)
        for a,b in zip(initial_seen,seen[0]):np.testing.assert_array_equal(a,b)
        self.assertEqual(before,policy_hash(parameters));self.assertEqual(first.policy_hash,second.policy_hash)
        np.testing.assert_array_equal(first.actor.actions[:16],second.actor.actions[:16])

    def test_latent_chance_collision_actual_actor_outputs(self):
        params=initialize(6);c=Chance(jnp.zeros((1,2),bool),jnp.full((1,2),85),jnp.array([0]))
        a,_=step(initial(1),[[0,0]],c);b,_=step(initial(1),[[0,0]],c._replace(roll=jnp.full((1,2),86)))
        for seat in (0,1):
            x,y=encode(view_kernel(a,seat)),encode(view_kernel(b,seat))
            for u,v in zip(predict(params,x),predict(params,y)):np.testing.assert_array_equal(u,v)

    def test_stale_reused_drifted_and_tampered_buffers_reject(self):
        t=Trainer(7,SMALL);b=t.collect();t.update(b)
        with self.assertRaises(ValueError):t.update(b)
        b=t.collect();forged=replace(b,version=0)
        with self.assertRaises(ValueError):t.update(forged)
        t.parameters['policy_b']=t.parameters['policy_b']+1
        with self.assertRaisesRegex(ValueError,'drift'):t.update(b)
        t=Trainer(8,SMALL);b=t.collect();b.targets.rewards.flags.writeable=True;b.targets.rewards[0]+=1
        with self.assertRaisesRegex(ValueError,'content changed'):t.update(b)


class Resume(unittest.TestCase):
    def test_exact_next_collection_and_updates(self):
        t=Trainer(11,SMALL);t.update(t.collect())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.json';save(path,t);other=load(path,expected_plan=SMALL)
            self.assertEqual(state(t),state(other))
            for _ in range(2):
                a,b=t.collect(),other.collect();self.assertEqual(a.content_hash,b.content_hash)
                self.assertEqual(t.update(a),other.update(b));self.assertEqual(state(t),state(other))

    def test_crash_rollback_and_atomic_save(self):
        t=Trainer(29,SMALL);t.update(t.collect())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.json';save(path,t);before=path.read_bytes()
            expected=load(path,expected_plan=SMALL);b=expected.collect();expected.update(b)
            for where in ('collection','update'):
                bad=load(path,expected_plan=SMALL)
                with self.assertRaises(RuntimeError):
                    if where=='collection':bad.collect(crash_after_collection=True)
                    else:bad.update(bad.collect(),crash_after_minibatches=1)
                with self.assertRaisesRegex(ValueError,'completed-update'):save(Path(d)/'invalid.json',bad)
                replay=load(path,expected_plan=SMALL);rb=replay.collect();self.assertEqual(rb.content_hash,b.content_hash);replay.update(rb)
                self.assertEqual(state(replay),state(expected))
            with self.assertRaises(RuntimeError):save(path,expected,interrupt_before_replace=True)
            self.assertEqual(path.read_bytes(),before);self.assertTrue(Path(str(path)+'.partial').exists())
            self.assertEqual(load(path,expected_plan=SMALL).updates,1)

    def test_checkpoint_checksum_drift_and_counter_corruption(self):
        t=Trainer(47,SMALL)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.json';save(path,t);original=json.loads(path.read_bytes())
            for mutation in ('checksum','runtime','counter','second_moment'):
                doc=copy.deepcopy(original)
                if mutation=='checksum':doc['sha256']='0'*64
                elif mutation=='runtime':doc['body']['provenance']['runtime']['prng']='wrong'
                elif mutation=='counter':doc['body']['counters']['games']=1
                else:doc['body']['optimizer']['v']['value_b']['hex']=np.array(-1,np.float32).tobytes().hex()
                if mutation!='checksum':doc['sha256']=digest(doc['body'])
                path.write_text(json.dumps(doc))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):load(path,expected_plan=SMALL)
