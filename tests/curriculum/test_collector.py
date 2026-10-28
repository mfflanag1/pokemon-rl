import copy
import hashlib
import tempfile
import unittest
import numpy as np
from pokemon_full_train.storage import ROOT
from pokemon_full_train.learning import prepare_batch
from pokemon_mixture.collector import Pool as OriginalPool
from pokemon_battle_audit.core import fingerprint,validate_scenario
from pokemon_curriculum.collector import Worker,Episode,Pool
from pokemon_curriculum.common import read,admit,initialize,training_specs,collect_update,value_diagnostics

PLAN=read(ROOT/'contracts/curriculum-learning-v1/plan.json')


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank,cls.weights,cls.supervised=admit(PLAN)
        cls.p,cls.opt=initialize(7103,PLAN)

    def test_real_train_bank_and_reset_no_prefix_rows(self):
        chosen=max(self.bank.values(),key=lambda s:len(s['prefix']))
        spec=dict(seed=42,episode=0,max_turns=120,split='train',swap=False,opponent='tactical',
                  learner_side=chosen['learner_side'],actor_seed=99,scenario_hash=chosen['content_hash'])
        ep=Episode(**spec)
        with tempfile.TemporaryDirectory() as tmp,Worker(tmp,self.bank) as worker:
            frame=worker.request('start',**ep.start);ep.receive('start',frame)
            self.assertTrue(chosen['prefix']);self.assertEqual(ep.rows,[]);self.assertEqual(ep.trace,[])
            self.assertEqual(ep.counts,chosen['counters']);self.assertEqual(frame['prefix']['target_frame_hash'],chosen['target_frame_hash'])
            self.assertEqual(sum(ep.counts),frame['prefix']['decisions'])
            self.assertEqual(frame['prefix']['simulator_start'],chosen['start'])
            self.assertLess(frame['players'][chosen['learner_side']]['turn'],120)
            ep2=Episode(**{**spec,'actor_seed':100});self.assertNotEqual(ep.seeds,ep2.seeds)

    def test_prefix_rollout_has_only_fresh_learner_rows(self):
        specs=training_specs(PLAN,7103,'prefix',0,self.bank,self.weights)
        with tempfile.TemporaryDirectory() as tmp,Pool(tmp,8,self.bank) as pool:
            results,_=pool.collect(self.p,specs)
        for spec,(rows,g) in zip(specs,results):
            scenario=self.bank[spec['scenario_hash']]
            self.assertEqual(hashlib.sha256(rows[0]['features'].tobytes()).hexdigest(),scenario['target_feature_hash'])
            self.assertEqual(len(rows),g['learner_decisions']);self.assertEqual(len(rows),g['diagnostics']['learner_decisions'])
            self.assertEqual(g['prefix']['decisions'],sum(scenario['counters']))
            self.assertEqual(len(g['trace']),g['decisions'])
            for i,row in enumerate(rows):
                self.assertEqual(row['trajectory'],0 if g['learner_side']=='p1' else 1)
                self.assertEqual(row['next_index'],i+1 if i+1<len(rows) else -1)
            batch=prepare_batch(rows);self.assertTrue(np.isfinite(batch['returns']).all())
            self.assertTrue((batch['returns']>=-1e-5).all());self.assertTrue((batch['returns']<=1+1e-5).all())

    def test_official_collector_matches_sealed_mixture(self):
        specs=training_specs({**PLAN,'max_turns':6},7103,'official',0,self.bank,self.weights)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,Pool(a,8,self.bank) as current,OriginalPool(b,8) as old:
            actual,_=current.collect(self.p,specs);expected,_=old.collect(self.p,specs)
        for (rows,g),(old_rows,old_g) in zip(actual,expected):
            self.assertEqual({k:g[k] for k in old_g},old_g)
            self.assertEqual(g['prefix']['decisions'],0)
            self.assertEqual(len(rows),len(old_rows))
            for x,y in zip(rows,old_rows):
                for key in x:np.testing.assert_array_equal(x[key],y[key])

    def test_development_prefix_and_tainted_bank_rejected(self):
        chosen=next(iter(self.bank.values()))
        with self.assertRaises(ValueError):Episode(seed=42,episode=0,max_turns=120,split='development',opponent='tactical',scenario_hash=chosen['content_hash'])
        bad=copy.deepcopy(chosen);bad['start']['split']='development'
        bad['content_hash']=fingerprint({k:v for k,v in bad.items() if k!='content_hash'})
        with self.assertRaises(ValueError):validate_scenario(bad)
        with tempfile.TemporaryDirectory() as tmp,Worker(tmp,self.bank) as w:
            with self.assertRaises(ValueError):w.request('start',run_seed=42,episode_index=0,split='final',swap=False,scenario_hash=chosen['content_hash'])

    def test_complete_gae_before_exact_stratified_selection(self):
        with tempfile.TemporaryDirectory() as tmp,Pool(tmp,8,self.bank) as pool:
            batch,ledger,_=collect_update(pool,self.p,PLAN,7103,'curriculum',0,{'official':0,'prefix':0},self.bank,self.weights)
        self.assertEqual(ledger['quotas'],{'official':922,'prefix':102})
        self.assertEqual(len(batch['actions']),1024)
        self.assertEqual(sum(s=='prefix' for s,_ in ledger['selected_indices']),102)
        self.assertEqual(len(set(map(tuple,ledger['selected_indices']))),1024)
        self.assertEqual(sum(ledger['collected_rows'].values())-1024,sum(ledger['excluded_rows'].values()))
        self.assertTrue(all(g['opponent']=='tactical' for group in ledger['games'].values() for g in group))
        self.assertTrue(all(g['prefix']['scenario_hash'] for g in ledger['games']['prefix']))
        from pokemon_transfer.learning import optimize_batch
        from pokemon_full_train.storage import save_checkpoint,load_checkpoint
        from pokemon_full_train.learning import derived_seed
        from pokemon_full.learner import parameter_hash
        from pokemon_curriculum.verify import state_equal,same
        key=derived_seed('f12-update',7103,0)
        updated,opt,metrics=optimize_batch(self.p,self.opt,batch,self.supervised,key,.1)
        self.assertNotEqual(parameter_hash(updated),parameter_hash(self.p));self.assertEqual(int(opt[2]),8)
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path
            path=Path(tmp)
            save_checkpoint(path,self.p,self.opt,{'input_hash':'preflight','seed':7103,'cohorts':0,'cursors':{'official':0,'prefix':0}})
            restored,ropt,_=load_checkpoint(path,'preflight')
            again,other,repeat=optimize_batch(restored,ropt,batch,self.supervised,key,.1)
            same(metrics,repeat,'preflight original next update metrics')
            self.assertEqual(parameter_hash(updated),parameter_hash(again))
            for i in [0,1]:
                for k in opt[i]:np.testing.assert_array_equal(opt[i][k],other[i][k])
            state_equal(self.p,self.opt,path,'preflight',7103,0,{'official':0,'prefix':0})
            with self.assertRaises(ValueError):state_equal(self.p,self.opt,path,'preflight',7103,1,{'official':0,'prefix':0})

    def test_terminal_value_fitted_constant_excludes_cutoffs(self):
        def game(reward,kind='terminal'):
            return {'learner_side':'p1','outcome':{'kind':kind,'rewards':{'p1':reward,'p2':1-reward}},
                    'trace':[['p1',1,0,.7,'hash']]}
        out=value_diagnostics([game(1),game(0),game(1,'administrative_cutoff')])
        self.assertEqual(out['terminal_games'],2);self.assertEqual(out['cutoff_games'],1)
        self.assertEqual(out['same_cell_fitted_constant']['game_prediction'],.5)
        self.assertEqual(out['same_cell_fitted_constant']['game_mse'],.25)

    def test_aggregate_report_corruption_rejected(self):
        from pokemon_curriculum.verify import verify_totals
        from pokemon_full_train.campaign import value_summary
        game={'learner_side':'p1','outcome':{'kind':'terminal','rewards':{'p1':0,'p2':1}},
              'decisions':2,'learner_rows':1,'unavailable_choices':0,'prefix':{'decisions':3},
              'terminal_value_mse':.49,'value_mean':.7}
        analysis={'7103':{'curriculum_benefit':{'passed':False}}}
        report={'training':value_summary([game]),'evaluation':value_summary([game]),
                'training_unavailable_choices':0,'evaluation_unavailable_choices':0,
                'learner_rows_collected':1,'prefix_replay_decisions':3,
                'all_seed_curriculum_benefit':False,'final_reserved_data_used':False,
                'public_server_used':False,'prime_used':False}
        verify_totals(report,[game],[game],analysis)
        for field in ['learner_rows_collected','prefix_replay_decisions','all_seed_curriculum_benefit','final_reserved_data_used']:
            bad=copy.deepcopy(report);bad[field]=True if type(bad[field]) is bool else bad[field]+1
            with self.assertRaises(ValueError):verify_totals(bad,[game],[game],analysis)
        bad=copy.deepcopy(report);bad['training']['learner_wins']=1
        with self.assertRaises(ValueError):verify_totals(bad,[game],[game],analysis)

    def test_sealer_rejects_incomplete_or_unbound_proofs(self):
        from unittest.mock import patch
        from pathlib import Path
        from pokemon_curriculum.seal import check_records
        from pokemon_curriculum.checks import COMMANDS
        manifest={'plan':PLAN,'sources':{'module.py':'sourcehash'},'runtime':{'python':'test-runtime'}}
        report={'status':'passed','plan':PLAN,'input_hash':'hash','counts':PLAN['expected_counts'],
                'all_seed_curriculum_benefit':False,'interpretation':PLAN['interpretation']}
        verification={'status':'passed','input_hash':'hash','report_sha256':'hash','all_seed_curriculum_benefit':False,
            'checks':{**PLAN['expected_counts'],'updates':288,'stage_checkpoints':18,'original_next_update_recoveries':6}}
        tests={'status':'passed','sources':manifest['sources'],'runtime':manifest['runtime'],'tests_passed':sum(c for _,_,c in COMMANDS),
               'runs':[{'command':command,'exit_code':0,'tests':count,'count_verified':True,'log':name,'sha256':'hash'}
                       for name,command,count in COMMANDS]}
        run=ROOT/'fake-run';review=ROOT/'fake-review'
        def exit_record(path):
            production=path.name=='production-exit.json'
            artifact=run/'report.json' if production else review/'verification/verification.json'
            module='pokemon_curriculum.campaign' if production else 'pokemon_curriculum.verify'
            args=['--run',str(run)]+([] if production else ['--evidence',str(review/'verification')])
            return {'exit_code':0,'artifact':str(artifact.relative_to(ROOT)),'sha256':'hash',
                    'command':[str(ROOT/'.venv/bin/python'),'-m',module,*args]}
        logs={name:f'Ran {count} tests in 1.0s\n\nOK\n' for name,_,count in COMMANDS}
        with patch('pokemon_curriculum.seal.digest',return_value='hash'),patch('pokemon_curriculum.seal.read',side_effect=exit_record),patch.object(Path,'read_text',lambda path:logs[path.name]):
            def check(r=report,v=verification,t=tests):check_records(run,review,manifest,r,v,t)
            check()
            for field,value in [('status','running'),('input_hash','wrong'),('counts',{}),('plan',{}),('interpretation','false claim')]:
                bad=copy.deepcopy(report);bad[field]=value
                with self.assertRaises(ValueError):check(r=bad)
            for field,value in [('report_sha256','wrong'),('all_seed_curriculum_benefit',True)]:
                bad=copy.deepcopy(verification);bad[field]=value
                with self.assertRaises(ValueError):check(v=bad)
            for field in ['evaluation_games','stage_checkpoints','original_next_update_recoveries','updates']:
                bad=copy.deepcopy(verification);bad['checks'][field]-=1
                with self.assertRaises(ValueError):check(v=bad)
            bad=copy.deepcopy(tests);bad['sources']={}
            with self.assertRaises(ValueError):check(t=bad)
            bad=copy.deepcopy(tests);bad['runs'][0]['sha256']='wrong'
            with self.assertRaises(ValueError):check(t=bad)
            for field,value in [('exit_code',1),('count_verified',False),('command',['wrong'])]:
                bad=copy.deepcopy(tests);bad['runs'][0][field]=value
                with self.assertRaises(ValueError):check(t=bad)
            for field,value in [('tests_passed',0),('runtime',{})]:
                bad=copy.deepcopy(tests);bad[field]=value
                with self.assertRaises(ValueError):check(t=bad)
            with patch.object(Path,'read_text',return_value='Ran 7 tests in 1.0s\nFAILED\n'):
                with self.assertRaises(ValueError):check()
            for field,value in [('sha256','wrong'),('command',['wrong']),('artifact','wrong')]:
                def bad_exit(path):
                    record=exit_record(path);record[field]=value;return record
                with patch('pokemon_curriculum.seal.read',side_effect=bad_exit):
                    with self.assertRaises(ValueError):check()
            with patch('pokemon_curriculum.seal.read',return_value={'exit_code':1}):
                with self.assertRaises(ValueError):check()


if __name__=='__main__':unittest.main()
