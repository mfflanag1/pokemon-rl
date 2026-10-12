import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import FunctionType
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_showdown.campaign import verify_case,word_audit
from pokemon_showdown.chance import sample,from_words,words_kernel
from pokemon_showdown.loop import Executables,start,check_report
from pokemon_showdown.checkpoint import Runner,save,load,state
from pokemon_jax.engine import initial,step_kernel,view_kernel,PlayerView
from pokemon_binary.loop import encode
from pokemon_binary.policy import action_kernel
from pokemon_jax.chance import event_key
from pokemon_learner.checkpoint import digest

ROOT=Path(__file__).resolve().parents[2]

def exact(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

class ChanceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.exe=Executables();cls.exe.diagnostics()

    def test_word_address_invariance_and_bounds(self):
        ids=np.array([0,1,32,2**32-1],np.uint32);turns=np.array([1,2,3,4],np.uint32)
        actual=sample(2**32-1,ids,turns)
        words=words_kernel(jax.random.key(2**32-1),ids,turns);exact(actual,from_words(words))
        order=np.array([3,0,2,1]);other=sample(2**32-1,ids[order],turns[order])
        for a,b in zip(actual,other):np.testing.assert_array_equal(np.asarray(a)[order],b)
        for i in range(4):
            part=sample(2**32-1,ids[i:i+1],turns[i:i+1])
            for a,b in zip(actual,part):np.testing.assert_array_equal(a[i:i+1],b)
        for seed,ids_,ts in [(-1,ids,turns),(True,ids,turns),(0,[0,0],[1,1]),(0,[2**32],[1]),(0,[0],[0]),(0,[0],[5])]:
            with self.assertRaises(ValueError):sample(seed,ids_,ts)
        self.assertTrue(all(x.device.platform=='cpu' for x in jax.tree.leaves(actual)))

    def test_captured_words_and_corruption_rejection(self):
        case=json.loads((ROOT/'tests/showdown-chance/fixtures/player-case.json').read_text())
        derived,count=verify_case(case,0);self.assertEqual(derived,case)
        self.assertGreater(count['tie'],0)
        changes=[lambda c:c.update(seed='old'),lambda c:c['turns'][0]['chance'].pop(),
            lambda c:c['turns'][0]['chance'][0].update(word=2**32),
            lambda c:c['turns'][0]['chance'][0].update(result=1-c['turns'][0]['chance'][0]['result']),
            lambda c:c['turns'][0]['pending'][0].update(turn=4),
            lambda c:c['turns'][0].update(tie_first=1-c['turns'][0]['tie_first']),
            lambda c:c['turns'][0]['events'][0].update(roll=84)]
        for change in changes:
            altered=copy.deepcopy(case);change(altered)
            with self.assertRaises((ValueError,KeyError)):verify_case(altered,0)

    def test_complete_collector_uses_new_chance(self):
        # Execute the sealed predecessor's code with only its chance dependency replaced.
        # A private globals copy leaves the imported predecessor module unchanged.
        import pokemon_binary.loop as predecessor
        from pokemon_showdown.chance import sample_kernel
        original=FunctionType(predecessor.collect.__code__,
            dict(predecessor.collect.__globals__,sample_kernel=sample_kernel))
        reference=jax.jit(lambda pars,game,seed:original(pars,game,seed,self.exe.plan))
        for seed in (11,29,47):
            old=start(seed)
            args=(old.parameters,jnp.uint32(128),jnp.uint32(seed))
            exact(reference(*args),self.exe.compiled['collect'](*args))
        seed=71;l=start(seed);buffer,actual=self.exe.compiled['collect'](l.parameters,l.next_game,jnp.uint32(seed))
        s=initial(64);ids=jnp.arange(64,dtype=jnp.uint32);master=jax.random.key(seed)
        env=jax.random.fold_in(master,0x454e56);actor=jax.random.fold_in(master,0x414354)
        for t in range(4):
            views=[view_kernel(s,i) for i in (0,1)]
            view=PlayerView(*[jnp.stack([v[k] for v in views],axis=1).reshape((-1,*views[0][k].shape[1:])) for k in (0,1)])
            keys=jax.vmap(lambda game,turn:jnp.stack([event_key(actor,game,turn,side) for side in (0,1)]))(ids,s.turn.astype(jnp.uint32)).reshape(-1)
            actions,logs,values=action_kernel(l.parameters,encode(view),keys)
            valid=np.asarray(buffer.valid[t*128:(t+1)*128])
            np.testing.assert_array_equal(actions,buffer.actor.actions[t*128:(t+1)*128])
            np.testing.assert_allclose(logs,buffer.actor.old_logp[t*128:(t+1)*128],atol=3e-6,rtol=3e-5)
            chances=from_words(words_kernel(env,ids,s.turn.astype(jnp.uint32)))
            s,_=step_kernel(s,jnp.maximum(actions,0).reshape(64,2),chances)
        exact(s,actual)
        separate,_=self.exe.compiled['update'](l,buffer,jnp.uint32(seed));fused,r=self.exe(l,seed);check_report(r)
        for a,b in zip(jax.tree.leaves(separate),jax.tree.leaves(fused)):np.testing.assert_allclose(a,b,atol=3e-6,rtol=3e-5)

    def test_new_checkpoint_resume_and_old_format_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.json';r=Runner(11,self.exe)
            for _ in range(3):r.run()
            save(p,r);other=load(p,self.exe);exact(r.run(),other.run());self.assertEqual(state(r),state(other))
            baseline=p.read_bytes()
            for format in ('binary-policy-checkpoint-1','fixed-loop-checkpoint-1'):
                doc=json.loads(baseline);doc['body']['format']=format;doc['sha256']=digest(doc['body']);p.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):load(p,self.exe)
            p.write_bytes(baseline)
            for fault in ('after_collect','after_update'):
                broken=load(p,self.exe)
                with self.assertRaises(RuntimeError):broken.run(fault=fault)
                with self.assertRaises(ValueError):save(Path(tmp)/'broken.json',broken)

if __name__=='__main__':unittest.main()
