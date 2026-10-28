import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import engine,carry,chance,parity,scalar
from pokemon_switch_engine.checkpoint import fingerprint
from pokemon_switch_engine.chance import sample as original_chance
from pokemon_learner.checkpoint import digest,wire

ROOT=Path(__file__).resolve().parents[2]


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script="import {battleCase,checkUnpatched} from './packages/continuation/reference.mjs';const rows=[];for(const id of [0,2,3,4]){const r=await battleCase(id);await checkUnpatched(r);rows.push(r);}console.log(JSON.stringify(rows));"
        cls.cases=json.loads(subprocess.check_output(['node','--input-type=module','-e',script],cwd=ROOT))
        cls.ids=np.array([17,29],np.uint32);cls.admitted=carry.admission()
        s=engine.initial(np.array([[0,23],[23,0]],np.int32))
        for _ in range(12):
            ch=chance.sample(jax.random.key(317),jnp.asarray(cls.ids),s.group.astype(jnp.uint32))
            s,_=carry.step(s,3-np.asarray(s.active),ch,13)
        cls.state=s

    def test_reference_scalar_jax_and_serialized_windows(self):
        for c in self.cases:parity.check_identity(c)
        result=parity.check_batch(self.cases);self.assertEqual(result['games'],4)
        result=parity.check_windows(self.cases);self.assertGreater(result['original_next_steps'],4)

    def test_frozen_windows_and_resume_exact_next_step(self):
        s=self.state;ch=chance.sample(jax.random.key(317),jnp.asarray(self.ids),s.group.astype(jnp.uint32))
        raw=carry.encode(s,self.ids,317,13,self.admitted);r=carry.decode(raw,self.ids,317,13,self.admitted)
        noop,events=carry.step(r,np.full((2,2),-1,np.int32),ch,13)
        self.assertEqual(fingerprint(noop),fingerprint(s));self.assertTrue(np.all(np.asarray(events)==-1))
        a=carry.step(s,3-np.asarray(s.active),ch,25);b=carry.step(r,3-np.asarray(r.active),ch,25)
        self.assertEqual(fingerprint(a),fingerprint(b));self.assertTrue(np.all(np.asarray(a[0].turn)==14))

    def test_mixed_terminal_and_live_lanes(self):
        s=self.state
        # Terminal copy is a constructed host fixture; the reference tests cover natural outcomes.
        s=s._replace(phase=s.phase.at[0].set(2),winner=s.winner.at[0].set(0),hp=s.hp.at[0,1].set(0))
        ch=chance.sample(jax.random.key(317),jnp.asarray(self.ids),s.group.astype(jnp.uint32))
        actions=3-np.asarray(s.active);actions[0]=-1
        after,_=carry.step(s,actions,ch,25)
        for x,y in zip(s,after):np.testing.assert_array_equal(x[0],y[0])
        self.assertEqual(int(after.turn[1]),14)

    def test_private_variants_and_pp_do_not_leak_after_boundary(self):
        s=self.state
        for observer in (0,1):
            other=1-observer
            changed=s._replace(swift=s.swift.at[:,other,:].set(1-s.swift[:,other,:]),
                               pp=s.pp.at[:,other,:,:].add(-1))
            self.assertEqual(fingerprint(engine.view_kernel(s,observer)),fingerprint(engine.view_kernel(changed,observer)))
            self.assertNotEqual(fingerprint(engine.view_kernel(s,other)),fingerprint(engine.view_kernel(changed,other)))

    def test_event_addresses_survive_serialization_and_batch_permutation(self):
        s=self.state;master=jax.random.key(317);ids=jnp.asarray(self.ids);g=s.group.astype(jnp.uint32)
        c=chance.sample(master,ids,g);old=original_chance(master,ids,g)
        for x,y in zip(c,old):np.testing.assert_array_equal(x,y)
        for x,y in zip(c,chance.sample(master,ids[::-1],g[::-1])):np.testing.assert_array_equal(x,y[::-1])
        p=jax.tree.map(lambda x:x[::-1],s);a=carry.step(s,3-np.asarray(s.active),c,25)
        b=carry.step(p,3-np.asarray(p.active),chance.sample(master,ids[::-1],g[::-1]),25)
        self.assertEqual(fingerprint(a),fingerprint(jax.tree.map(lambda x:x[::-1],b)))

    def test_checkpoint_recomputed_corruptions_rejected(self):
        raw=carry.encode(self.state,self.ids,317,13,self.admitted);original=json.loads(raw)
        for kind in ('format','seed','ids','boundary','shape','padding','group','move_index','event_type','turn_order','foreign_admission'):
            doc=copy.deepcopy(original);b=doc['body']
            if kind=='format':b['format']='bounded-value-checkpoint-1'
            elif kind=='seed':b['master_seed']=319
            elif kind=='ids':b['game_ids'].reverse()
            elif kind=='boundary':b['boundary']=25
            elif kind=='shape':b['state']['history']['shape']=[2,32,9]
            elif kind=='foreign_admission':b['admission_hash']='0'*64
            else:
                name='group' if kind=='group' else 'history';item=b['state'][name]
                a=np.frombuffer(bytes.fromhex(item['hex']),dtype='<i4').copy().reshape(item['shape'])
                if kind=='padding':a[0,-1,0]=0
                elif kind=='move_index':a[0,0,5]=7
                elif kind=='event_type':a[0,0,8]=1
                elif kind=='turn_order':a[0,0,0]=12
                else:a[0]=0
                item['hex']=a.tobytes().hex()
            doc['sha256']=digest(b)
            with self.assertRaises(ValueError,msg=kind):carry.decode(wire(doc),self.ids,317,13,self.admitted)
        with self.assertRaises(ValueError):carry.decode(raw[:-9],self.ids,317,13,self.admitted)

    def test_exclusive_save_and_invalid_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'carry.json';carry.save(p,self.state,self.ids,317,13,self.admitted)
            with self.assertRaises(FileExistsError):carry.save(p,self.state,self.ids,317,13,self.admitted)
        with self.assertRaises(ValueError):carry.encode(self.state,self.ids,317,25,self.admitted)
        with self.assertRaises(ValueError):carry.mask(self.state,12)
        with self.assertRaises(ValueError):engine.validate_state(self.state._replace(count=jnp.array([123,24])))

    def test_tape_corruption_rejected(self):
        row=copy.deepcopy(self.cases[0]);row['steps'][0]['chance']['roll'][0]^=1
        with self.assertRaises((AssertionError,ValueError)):parity.check_batch([row])
        row=copy.deepcopy(self.cases[0]);row['spec']['seed']='sodium,'+'0'*64
        with self.assertRaises(AssertionError):parity.check_identity(row)


if __name__=='__main__':unittest.main()
