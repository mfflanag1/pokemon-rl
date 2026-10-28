"""Engineering namespaces only; never consumes the registered F13 bank."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pokemon_credit.bank import material,eligible
from pokemon_credit.common import specs,initialize,batch,diagnostics,read
from pokemon_credit.campaign import PLAN,diagnostic_specs,same,analysis
from pokemon_credit.statistics import compare
from pokemon_credit.collector import Pool
from pokemon_credit.learning import optimize,gradient_probe
from pokemon_curriculum.collector import Pool as OldPool
from pokemon_throughput.collector import merge_rows
from pokemon_full_train.storage import ROOT,save_checkpoint,load_checkpoint
from pokemon_full.learner import parameter_hash
from pokemon_fit.data import fingerprint
from pokemon_battle_audit.core import validate_scenario,reset_scenario

class CreditTests(unittest.TestCase):
    def test_public_material_and_eligibility(self):
        live={'condition':'10/100'};dead={'condition':'0 fnt'}
        p={'own':[live]*3+[dead]*3,'opponent':[live,dead,dead,dead,None,None],'phase':'move','mask':[True],'turn':8}
        self.assertEqual(material(p),(3,3))
        f={'players':{'p1':p,'p2':p},'outcome':None}
        self.assertTrue(eligible(f,'p1','midgame'));self.assertFalse(eligible(f,'p1','endgame'))
        f['players']['p1']={**p,'turn':7};self.assertFalse(eligible(f,'p1','midgame'))
        f['players']['p1']=p;f['players']['p2']={**p,'phase':'force_switch'};self.assertFalse(eligible(f,'p1','midgame'))
        self.assertEqual(material({'own':[live]*6,'opponent':[None]*6}),(6,6))

    def test_fit_probe_requests_disjoint(self):
        plan=read(PLAN);bank=[]
        for role in ['fit','probe']:
            for f in ['midgame','endgame']:
                for i in range(12):bank.append({'origin':{'role':role,'teacher_episode':i},'families':[f],'content_hash':f'{role}-{f}-{i}','learner_side':'p1'})
        fit=specs(bank,plan,1,0,32,'fit');probe=specs(bank,plan,1,0,96,'probe')
        self.assertTrue(all(s['scenario_hash'].startswith('fit') for s in fit))
        self.assertTrue(all(s['scenario_hash'].startswith('probe') for s in probe))
        self.assertEqual(len(set(s['scenario_hash'] for s in probe)),24)
        with self.assertRaises(ValueError):specs(bank,plan,1,0,1,'final')

    def test_cluster_cutoff_and_pairing(self):
        settings=read(PLAN)['statistics']
        def games(reward,cutoff=False):
            return [dict(episode=i,learner_side='p1',prefix={'scenario_hash':str(i//4)},provenance={},decisions=1,learner_decisions=1,
                outcome={'kind':'administrative_cutoff' if cutoff else 'terminal','rewards':{'p1':reward}}) for i in range(48)]
        a=games(0);b=games(1);r=compare(a,b,settings)
        self.assertEqual(r['point_interval'],[1,1]);self.assertEqual(r['gate'],'improvement')
        self.assertEqual(compare(b,a,settings)['gate'],'regression')
        self.assertEqual(compare(games(0,True),games(0,True),settings)['point_interval'],[-1,1])
        b[0]['episode']=99
        with self.assertRaises(ValueError):compare(a,b,settings)
        a[1]=copy.deepcopy(a[0])
        with self.assertRaises(ValueError):compare(a,copy.deepcopy(a),settings)

    def test_next_update_and_gradient_probe(self):
        p,opt=initialize(7103)
        with np.load(ROOT/'artifacts/runs/f10-long-transfer-v1/inputs/supervised.npz') as data:sup={k:data[k] for k in data.files}
        n=256;features=np.tile(sup['features'],(2,1));mask=np.tile(sup['mask'],(2,1));actions=np.tile(sup['targets'],2)
        from pokemon_full.policy import forward,distribution
        probs,logs=distribution(forward(p,features)[0],mask)
        b={'features':features,'mask':mask,'actions':actions,'old_logp':np.asarray(logs)[np.arange(n),actions],
            'advantages':np.full(n,-.2,np.float32),'returns':np.zeros(n,np.float32)}
        before=parameter_hash(p);probes=gradient_probe(p,b,sup,{'features':features[:8],'mask':mask[:8]})
        self.assertEqual(set(probes),{'actor','value','entropy','rehearsal','joint'})
        self.assertTrue(all(np.isfinite(list(x.values())).all() for x in probes.values()));self.assertEqual(parameter_hash(p),before)
        q,other,metrics=optimize(p,opt,b,sup);self.assertEqual(int(other[2]),2);self.assertEqual(len(metrics),2)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(Path(d),p,opt,{'input_hash':'engineering','seed':7103,'cohorts':0})
            rp,ro,_=load_checkpoint(Path(d),'engineering');q2,o2,m2=optimize(rp,ro,b,sup)
            self.assertEqual(parameter_hash(q),parameter_hash(q2));same(metrics,m2,'metrics')
            for a,c in [(other[0],o2[0]),(other[1],o2[1])]:
                for k in a:np.testing.assert_array_equal(a[k],c[k])
        bad={**b,'hidden':np.ones(256)}
        with self.assertRaises(ValueError):optimize(p,opt,bad,sup)

    def test_annotated_collector_preserves_sealed_behavior(self):
        p,_=initialize(7103);plan=read(PLAN);requests=diagnostic_specs(plan,998217)[:8]
        for s in requests:s['max_turns']=4
        with tempfile.TemporaryDirectory() as d:
            with OldPool(Path(d)/'old',8,{}) as pool:old,_=pool.collect(p,requests)
            with Pool(Path(d)/'new',8,{}) as pool:new,_=pool.collect(p,requests)
        same([g for _,g in old],[g for _,g in new],'game parity')
        a=merge_rows(old);b=merge_rows(new)
        self.assertEqual(len(a),len(b));self.assertGreater(len(b),0)
        for x,y in zip(a,b):
            for k in x:np.testing.assert_array_equal(x[k],y[k])
            self.assertEqual(y['voluntary_switch'],y['phase']=='move' and y['action']>=8)
        gae,err=batch(b,.95);mc,_=batch(b,1.)
        self.assertLess(err,2e-6);self.assertEqual(set(gae),set(mc));self.assertGreater(diagnostics(b)['groups']['cutoff']['rows'],0)

    def test_long_chain_and_f12_target_parity(self):
        rng=np.random.default_rng(913711);errors=[]
        def check(values,reward,terminal,bootstrap=0.):
            rows=[dict(value=float(v),reward=0.,next_index=i+1,bootstrap=0.,terminal=False,trajectory=0,
                       action=0,old_logp=0.,features=np.zeros(928,np.float32),mask=np.ones(14,bool),voluntary_switch=False) for i,v in enumerate(values)]
            rows[-1].update(reward=reward if terminal else 0.,terminal=terminal,bootstrap=0. if terminal else bootstrap,next_index=-1)
            _,err=batch(rows,.95);errors.append(err)
        for length in [120,240,1000]:
            for mode in ['random','alternating','high','low']:
                for reward in [0.,.5,1.]:
                    values=(rng.random(length).astype(np.float32) if mode=='random' else
                            np.resize(np.array([0,1],np.float32),length) if mode=='alternating' else
                            np.full(length,np.nextafter(np.float32(1),np.float32(0)) if mode=='high' else np.finfo(np.float32).tiny,np.float32))
                    check(values,reward,True);check(values,0.,False,.37)
        factual=0
        for seed in [7103,8209,9311]:
            for arm in ['official','curriculum']:
                record=read(ROOT/f'artifacts/runs/f12-curriculum-v1/{seed}/{arm}/update-047.json')
                for group in record['ledger']['games'].values():
                    for g in group[:4]:
                        if g['outcome']['kind']!='terminal':continue
                        values=[entry[3] for entry in g['trace'] if entry[0]==g['learner_side']]
                        check(values,g['outcome']['rewards'][g['learner_side']],True);factual+=1
        self.assertGreaterEqual(factual,24);self.assertLess(max(errors),2e-6)
        print('Long-chain parity:',len(errors),'chains;',factual,'F12 factual chains; max error',max(errors),flush=True)

    def test_live_prefix_excludes_teacher_actions(self):
        # Reuse a sealed train prefix only as engineering transport fixture.
        plan=read(ROOT/'contracts/curriculum-learning-v1/plan.json')
        from pokemon_curriculum.common import admit
        bank,_,_=admit(plan);scenario=next(s for s in bank.values() if s['prefix'])
        validate_scenario(scenario);p,_=initialize(7103)
        req=dict(seed=998215,episode=0,split='train',swap=False,max_turns=120,opponent='tactical',learner_side=scenario['learner_side'],actor_seed=998216,scenario_hash=scenario['content_hash'])
        with tempfile.TemporaryDirectory() as d:
            with Pool(Path(d),8,{scenario['content_hash']:scenario}) as pool:result,_=pool.collect(p,[req])
        rows,g=result[0];self.assertGreater(g['prefix']['decisions'],0)
        self.assertEqual(len(rows),g['learner_decisions']);self.assertEqual(rows[0]['turn']>=1,True)
        self.assertEqual(sum(r['terminal'] for r in rows),int(g['outcome']['kind']=='terminal'))
        batch(merge_rows(result),1.)

if __name__=='__main__':unittest.main()
