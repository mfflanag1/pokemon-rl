import unittest,tempfile
from pathlib import Path
import numpy as np
from pokemon_balanced.common import PLAN,read,initialize,team_keys
from pokemon_balanced.sampling import classify
from pokemon_imitation.teacher import generate
from pokemon_transfer_v2.collector import Pool
from pokemon_transfer_v2.common import evaluation
class LiveTests(unittest.TestCase):
    def test_fresh_teacher_replay_and_phase_action_linkage(self):
        plan=read(PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,g,inv=generate(999161,4,'development',plan,root/'a');data2,g2,inv2=generate(999161,4,'development',plan,root/'b',root/'a')
            self.assertEqual(g,g2);self.assertEqual(inv,inv2);labels=classify(data)
            rows=[i for game in g for i in game['row_identity']]
            self.assertEqual(len(rows),len(labels))
            for i,row in enumerate(rows):
                expected=3 if row['phase'] in ['replace','revival'] else 0 if row['action']<4 else 1 if row['action']<8 else 2
                self.assertEqual(int(labels[i]),expected)
            for k in data:np.testing.assert_array_equal(data[k],data2[k])
    def test_live_development_collection_generates_no_training_rows(self):
        plan={**read(PLAN),'evaluation_seed':999162,'evaluation_pairs_per_opponent':4,'opponents':['tactical']};p,_=initialize(7103)
        with tempfile.TemporaryDirectory() as tmp:
            with Pool(Path(tmp),8) as pool:panel,_=evaluation(pool,p,plan,999163)
            games=panel['tactical'];self.assertEqual(len(games),8)
            self.assertTrue(all(g['split']=='development' for g in games));self.assertTrue(all(t['partition']=='development' for g in games for t in g['provenance']['teams']))
            self.assertEqual(len(team_keys(games)),8)
if __name__=='__main__':unittest.main()
