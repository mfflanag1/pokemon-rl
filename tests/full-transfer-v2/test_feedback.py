import copy
import json
import tempfile
import unittest
from pathlib import Path
from pokemon_full_train.storage import ROOT
from pokemon_full_train.learning import derived_seed
from pokemon_full_learning.campaign import evaluation_specs
from pokemon_transfer_v2.common import check_collection,initialize,read
from pokemon_transfer_v2.collector import Pool


class Feedback(unittest.TestCase):
    def test_rejected_advertised_decisions_are_allowed_without_filtering(self):
        rows=[{'action':10},{'action':0}];game={'decisions':2,'unavailable_choices':1}
        collection=[(rows,game)];before=copy.deepcopy(collection)
        check_collection(collection,True);self.assertEqual(collection,before)
        check_collection([([],game)],False)
        with self.assertRaises(ValueError):check_collection(collection,False)
        with self.assertRaises(ValueError):check_collection([([],game)],True)
        for invalid in [-1,3,True,1.5]:
            with self.assertRaises(ValueError):check_collection([([],{**game,'unavailable_choices':invalid})],False)

    def test_original_trapped_pair_reproduces_with_journal_and_no_learner_rows(self):
        plan=read(ROOT/'contracts/full-transfer-v2/plan.json');p,opt=initialize(7103,plan)
        # Exactly the failure coordinates, two repeated eight-slot collections.
        jobs=evaluation_specs(plan,7103,'random',4,4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            with Pool(path,8) as pool:
                first,_=pool.collect(p,jobs);second,_=pool.collect(p,jobs)
            self.assertEqual(first,second);check_collection(first,False)
            self.assertEqual(sum(g['unavailable_choices'] for rows,g in first),2)
            records=[json.loads(line) for file in path.glob('feedback-*.jsonl') for line in file.read_text().splitlines()]
            self.assertEqual(len(records),4)
            for r in records:
                self.assertEqual(r['start']['episode_index'],7)
                self.assertEqual(r['submitted']['action'],10)
                self.assertIn('trapped',r['submitted']['feedback'][0])
                side=r['submitted']['side'];self.assertTrue(r['before']['players'][side]['mask'][10])
                self.assertFalse(r['after']['players'][side]['mask'][10])
            self.assertEqual(int(opt[2]),0)


if __name__=='__main__':unittest.main()
