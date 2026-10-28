import copy
import json
from pathlib import Path
import tempfile
import unittest
from pokemon_battle_audit.core import (Selector, fingerprint, feature_hash, traced_command,
    start_spec, actor_seeds, policy_command, record_step, reset_scenario, continue_game)
from pokemon_battle_audit.campaign import games_for, parameters_for
from pokemon_battle_audit.transport import Worker
from pokemon_full_train.storage import ROOT
from pokemon_full_train.learning import derived_seed

PLAN=json.loads((ROOT/'contracts/battle-audit-v1/plan.json').read_text())


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters=parameters_for(PLAN,7103,'rehearsal')

    def test_selection_does_not_take_outcome_and_is_bounded(self):
        s=Selector(['hazards','hidden_information'],2)
        d={'disagreement':True,'tags':['hazards','hidden_information']}
        self.assertEqual(s.select(0,d),d['tags'])
        self.assertEqual(s.select(0,d),[])
        self.assertEqual(s.select(1,{**d,'disagreement':False}),[])
        self.assertEqual(s.select(1,d),d['tags'])
        self.assertEqual(s.select(2,d),[])

    def test_original_actor_keys_and_factual_projection(self):
        game=games_for(PLAN,7103,'rehearsal')[0]
        with tempfile.TemporaryDirectory() as tmp, Worker(tmp) as w:
            frame=w.request('start',**start_spec(game,PLAN['evaluation_seed']))
            seeds=actor_seeds(derived_seed('f4-evaluation',7103,'tactical'),game['episode'],game['swap'])
            counts=[0,0]; cursor=0
            for _ in range(6):
                command,end=traced_command(frame,game['trace'],cursor,game['learner_side'])
                sampled,values=policy_command(self.parameters,frame,seeds,counts,game['learner_side'])
                self.assertEqual(sampled,command)
                for row in game['trace'][cursor:end]:
                    if row[0]==game['learner_side']: self.assertEqual(values[row[0]],row[3])
                poisoned=copy.deepcopy(game['trace']); poisoned[cursor][4]='0'*64
                with self.assertRaisesRegex(ValueError,'feature/request'): traced_command(frame,poisoned,cursor,game['learner_side'])
                frame=w.request('step',**command)
                for i,s in enumerate(['p1','p2']): counts[i]+=int(s in command['actions'])
                cursor=end

    def test_train_reset_hands_off_before_target_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp, Worker(tmp) as w:
            start=dict(run_seed=derived_seed('f11-test',7103),episode_index=0,split='train',swap=False)
            frame=w.request('start',**start); provenance=frame['provenance']
            seeds=actor_seeds(start['run_seed'],0,False)
            command,values=policy_command(self.parameters,frame,seeds,[0,0],'p1')
            target=w.request('step',**command)
            step=record_step(frame,command,values,target)
            scenario={'schema_version':'f11-train-scenario-v1','split':'train','start':start,
                'provenance':provenance,'learner_side':'p1','actor_seeds':seeds,'counters':[1,1],
                'prefix':[{k:step[k] for k in ['before','command','submitted','after']}],
                'target_frame_hash':fingerprint(target['players']),'target_feature_hash':feature_hash(target['players']['p1'])}
            scenario['content_hash']=fingerprint(scenario)
            restored,counts=reset_scenario(w,scenario)
            self.assertEqual(restored['players'],target['players'])
            self.assertEqual(restored['submissions'],2) # prefix only; target not consumed
            suffix,end=continue_game(w,self.parameters,restored,seeds,counts,'p1',{**PLAN,'max_turns':4})
            self.assertTrue(suffix)
            self.assertEqual(suffix,json.loads(json.dumps(suffix,allow_nan=False)))
            self.assertEqual(suffix[0]['before'],scenario['target_frame_hash'])
            self.assertIn(end['outcome']['kind'],['terminal','administrative_cutoff'])
            restored,counts=reset_scenario(w,scenario)
            again,end2=continue_game(w,self.parameters,restored,seeds,counts,'p1',{**PLAN,'max_turns':4})
            self.assertEqual((suffix,end['outcome']),(again,end2['outcome']))
            for key,value in [('split','development'),('counters',[0,0]),('target_feature_hash','0'*64)]:
                bad=copy.deepcopy(scenario); bad[key]=value
                bad['content_hash']=fingerprint({k:v for k,v in bad.items() if k!='content_hash'})
                with self.assertRaises(ValueError): reset_scenario(w,bad)

    def test_final_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp, Worker(tmp) as w:
            with self.assertRaisesRegex(RuntimeError,'final'):
                w.request('start',run_seed=42,episode_index=0,split='final',swap=False)


if __name__=='__main__': unittest.main()
