import tempfile,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from pokemon_battle_audit.transport import Worker
from pokemon_imitation.teacher import collect_game
from pokemon_diversity.collection import ObservedWorker,validate_metadata

class Live(unittest.TestCase):
    def compare(self,seed,split):
        with tempfile.TemporaryDirectory() as td:
            with Worker(Path(td)/'original') as w:a,ga=collect_game(w,seed,0,split,120)
            with Worker(Path(td)/'observed') as w:
                observed=ObservedWorker(w);b,gb=collect_game(observed,seed,0,split,120)
            self.assertEqual(ga,gb);self.assertEqual(observed.identities,ga['row_identity'])
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            validate_metadata(b,observed.metadata)
            self.assertGreater(len(observed.metadata),0)
            self.assertTrue(all(t['partition']==split for t in ga['provenance']['teams']))
    def test_train_teacher_unchanged(self):self.compare(1025261899,'train')
    def test_development_teacher_unchanged(self):self.compare(1025261898,'development')
if __name__=='__main__':unittest.main()
