import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_switch_engine import scalar,engine,chance,parity

ROOT=Path(__file__).resolve().parents[2]


class EngineTests(unittest.TestCase):
    def test_complete_hp_display(self):
        rows=json.loads(subprocess.check_output(['node','packages/switch-engine/health.mjs'],cwd=ROOT))
        self.assertEqual(len(rows),357)
        hp,maximum,expected=np.array(rows,np.int32).T
        np.testing.assert_array_equal(engine._percent(hp,maximum),expected)
        self.assertEqual([scalar.percent(int(h),int(m)) for h,m in zip(hp,maximum)],expected.tolist())
        self.assertEqual(scalar.percent(66,120),55)

    def test_jax_damage_grid(self):
        inputs=[];expected=[]
        for row in json.loads((Path(__file__).parent/'fixtures/damage.json').read_text()):
            for i,hp in enumerate(row['hp']):
                for move in range(2):
                    for crit in range(2):
                        for roll in range(16):
                            inputs.append([scalar.SPECIES.index(row['attacker']),scalar.SPECIES.index(row['defender']),hp,move,crit,roll])
                            expected.append(row['values'][i][move][crit][roll])
        np.testing.assert_array_equal(engine.damage(*map(jnp.array,np.array(inputs).T)),expected)

    def test_fresh_and_sealed_parity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/runs') as folder:
            out=Path(folder)/'capture'
            subprocess.run(['node','packages/switch-engine/reference.mjs',str(out),'--smoke'],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
            oracle=json.loads(subprocess.check_output(['node','packages/switch-engine/audit.mjs',str(out)],cwd=ROOT))
            self.assertEqual(oracle['unpatched_games'],24)
            report=parity.audit(out);self.assertEqual(report['legacy']['games'],72);self.assertEqual(report['fresh']['games'],24)
            self.assertEqual(report['information']['hidden_initial_equal']+report['information']['hidden_after_equal'],1872)
            case=parity.packed(out/'cases-0000.json.gz')[0];case['spec']['seed']='sodium,'+'0'*64
            with self.assertRaises(AssertionError):parity.check_identity(case,parity.read(out/'plan.json'))
            case=parity.packed(out/'cases-0000.json.gz')[0];case['steps'][0]['chance']['roll'][0]^=1
            with self.assertRaises((AssertionError,ValueError)):parity.check_batch([case])
            plan=parity.read(out/'plan.json');plan['reference']['bad']='drift';(out/'plan.json').write_text(json.dumps(plan))
            failed=subprocess.run(['node','packages/switch-engine/audit.mjs',str(out)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.assertNotEqual(failed.returncode,0)

    def test_host_rejects_malformed_and_illegal(self):
        s=engine.initial(np.array([[0,0]],np.int32));c=engine.Chance(jnp.zeros((1,2),bool),jnp.zeros((1,2),jnp.int32),jnp.zeros(1,jnp.int32))
        for changed in [s._replace(hp=s.hp.at[0,0,0].set(-1)),s._replace(count=jnp.array([27])),s._replace(active=jnp.array([[2,0]])),s._replace(swift=s.swift.astype(jnp.float32))]:
            with self.assertRaises(ValueError):engine.step(changed,np.array([[0,0]]),c)
        for actions in [[[2,0]],[[-1,0]],[[4,0]]]:
            with self.assertRaises(ValueError):engine.step(s,np.array(actions),c)
        with self.assertRaises(ValueError):engine.step(s,np.array([[0,0]]),c._replace(roll=jnp.array([[16,0]])))

    def test_jax_counterfactual_private_reserve_and_speed(self):
        for observer in range(2):
            for own in range(24):
                ids=np.array([[own,t] if observer==0 else [t,own] for t in range(24)],np.int32)
                s=engine.initial(ids);v=engine.view_kernel(s,observer)
                for species in range(3):
                    group=[t for t in range(24) if scalar.ROSTER[t][0][0]==species]
                    for field in v:
                        for i in group[1:]:np.testing.assert_array_equal(field[group[0]],field[i])
                ownv=engine.view_kernel(s,1-observer)
                self.assertFalse(np.array_equal(ownv.own[0],ownv.own[2]))
        # Public HP buckets cannot encode exact opponent HP, nor opponent PP.
        s=engine.initial(np.array([[0,0]],np.int32));a=s._replace(hp=s.hp.at[0,1,0].set(118))
        b=s._replace(hp=s.hp.at[0,1,0].set(119),pp=s.pp.at[0,1,0,0].set(39))
        for x,y in zip(engine.view_kernel(a,0),engine.view_kernel(b,0)):np.testing.assert_array_equal(x,y)

    def test_chance_mapping_and_batch_permutation(self):
        ids=jnp.arange(1024,dtype=jnp.uint32);groups=ids%14;master=jax.random.key(531)
        c=chance.sample(master,ids,groups)
        for i in range(32):
            key=jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(master,chance.DOMAIN),ids[i]),groups[i])
            words=[int(jax.random.bits(jax.random.fold_in(key,r),dtype=jnp.uint32)) for r in range(5)]
            self.assertEqual(np.asarray(c.crit[i]).tolist(),[w*24//2**32==0 for w in words[:2]])
            self.assertEqual(np.asarray(c.roll[i]).tolist(),[w//2**28 for w in words[2:4]])
            self.assertEqual(int(c.tie[i]),words[4]//2**31)
        for a,b in zip(c,chance.sample(master,ids[::-1],groups[::-1])):np.testing.assert_array_equal(a,b[::-1])
        self.assertEqual(c.crit.dtype,jnp.bool_);self.assertEqual(c.roll.dtype,jnp.int32)
        self.assertEqual(set(np.asarray(c.roll).ravel()),set(range(16)))
        self.assertEqual(set(np.asarray(c.tie)),{0,1})


if __name__=='__main__':unittest.main()
