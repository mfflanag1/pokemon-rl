import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from pokemon_switch_audit.transport import Worker
from pokemon_switch_audit.replay import inspect_game
from pokemon_full_train.storage import ROOT
class ReplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.g=json.loads((ROOT/'artifacts/runs/f15-full-imitation-v1/teacher/game-0000.json').read_text())
  with np.load(ROOT/'artifacts/runs/f16-balanced-imitation-v1/inputs/teacher.npz',allow_pickle=False) as z:cls.d={k:z[k] for k in z.files}
 def test_all_original_rows_and_projection(self):
  with tempfile.TemporaryDirectory() as tmp:
   with Worker(tmp) as w:r=inspect_game(w,self.g,self.d,0)
  self.assertEqual(len(r),self.g['accepted_rows']);self.assertTrue(all(set(x['advice'])=={'projection','entry','types','boosts','entry_types_boosts'} and all(type(a)is int and x['player']['mask'][a] for a in x['advice'].values()) for x in r))
 def test_original_corruption_fails(self):
  g=copy.deepcopy(self.g);g['steps'][0]['before']='0'*64
  with tempfile.TemporaryDirectory() as tmp:
   with Worker(tmp) as w:
    with self.assertRaisesRegex(ValueError,'Original player view differs'):inspect_game(w,g,self.d,0)
if __name__=='__main__':unittest.main()
