from pathlib import Path
import tempfile
import unittest
import numpy as np
from pokemon_imitation.teacher import generate
from pokemon_imitation.common import PLAN,read,initialize,team_keys
from pokemon_imitation.learning import bc_step
from pokemon_imitation.collection import collect_update
from pokemon_mixture.collector import Pool
from pokemon_transfer_v2.collector import Pool as EvalPool
from pokemon_transfer_v2.common import evaluation
from pokemon_full_train.storage import ROOT
from pokemon_fit.data import fingerprint

class LiveTests(unittest.TestCase):
    def test_teacher_replay_split_and_accepted_rows(self):
        plan=read(PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,g,inv=generate(999151,4,'train',plan,root/'a');again,other,inv2=generate(999151,4,'train',plan,root/'b',root/'a')
            probe,pg,_=generate(999152,4,'development',plan,root/'p')
            self.assertEqual(inv,inv2);self.assertEqual(g,other);self.assertEqual(fingerprint(data),fingerprint(again))
            self.assertFalse(team_keys(g)&team_keys(pg));self.assertEqual(set(data['game_index']),set(range(4)))
            self.assertEqual(len(data['targets']),sum(x['accepted_rows'] for x in g));self.assertEqual(sum(x['decisions'] for x in g),len(data['targets'])+sum(x['rejected_choices'] for x in g))
            for game in g:
                targets=[s['action'] for step in game['steps'] for s in step['submitted'] if s['accepted']]
                np.testing.assert_array_equal(data['targets'][data['game_index']==game['episode']],targets)
            p,opt=initialize(7103)
            with np.load(ROOT/'artifacts/runs/f10-long-transfer-v1/inputs/supervised.npz') as d:old={k:d[k] for k in d.files}
            p,opt,item=bc_step(p,opt,data,old,999153,0)
            self.assertEqual(len(item['teacher_indices']),64)
    def test_full_rl_cohort_and_development_isolation(self):
        plan=read(PLAN);p,_=initialize(7103)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with Pool(root/'train',8) as pool:selected,ledger=collect_update(pool,p,plan,999154,0,0)
            self.assertEqual(len(selected['actions']),1024);self.assertEqual(len(set(ledger['selected_indices'])),1024)
            self.assertTrue(all(g['split']=='train' and g['opponent']=='tactical' for g in ledger['games']))
            test={**plan,'evaluation_seed':999155,'evaluation_pairs_per_opponent':4,'opponents':['tactical']}
            with EvalPool(root/'eval',8) as pool:panel,_=evaluation(pool,p,test,999156)
            self.assertEqual(len(panel['tactical']),8);self.assertFalse(team_keys(ledger['games'])&team_keys(panel['tactical']))
            self.assertTrue(all(g['split']=='development' and not g.get('learner_rows',0) for g in panel['tactical']))
if __name__=='__main__':unittest.main()
