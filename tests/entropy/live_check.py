"""Engineering prefix and evaluation-role checks; no production game keys."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from pokemon_full_train.storage import ROOT
from pokemon_credit.common import read,initialize
from pokemon_credit.collector import Pool
from pokemon_credit.campaign import collect_update
from pokemon_entropy.campaign import evaluation,behavior
from pokemon_entropy.common import PLAN
from pokemon_entropy.learning import optimize
from pokemon_fit.data import fingerprint

class LiveTests(unittest.TestCase):
    def test_fixed_observation_probe_and_repeat(self):
        plan=read(PLAN);bank=read(ROOT/plan['bank']);p,_=initialize(7103)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with Pool(root/'a',8,{s['content_hash']:s for s in bank}) as pool:games,probe=evaluation(pool,p,plan,bank,999141)
            with Pool(root/'b',8,{s['content_hash']:s for s in bank}) as pool:other,other_probe=evaluation(pool,p,plan,bank,999141)
            self.assertEqual(games,other);self.assertEqual(fingerprint(probe),fingerprint(other_probe))
            starts={s['content_hash']:s['start'] for s in bank}
            self.assertTrue(all(g['prefix']['simulator_start']==starts[g['prefix']['scenario_hash']] for group in games.values() for g in group))
            a=behavior(p,probe,root/'a.npz');b=behavior(p,probe,root/'b.npz',root/'a.npz');self.assertEqual(a,b)
        self.assertEqual(sum(map(len,games.values())),96);self.assertGreater(a['rows'],0)
        self.assertEqual(a['rows'],a['switch_opportunity_rows']);self.assertEqual(set(probe),{'features','mask'})
        self.assertTrue(np.all(probe['mask'][:,:8].any(1)&probe['mask'][:,8:].any(1)))
        with np.load(ROOT/'artifacts/runs/f10-long-transfer-v1/inputs/supervised.npz') as d:sup={k:d[k] for k in d.files}
        with self.assertRaises(ValueError):optimize(p,initialize(7103)[1],probe,sup,0.)

    def test_training_excludes_all_probe_starts(self):
        plan=read(PLAN);bank=read(ROOT/plan['bank']);p,_=initialize(7103)
        with tempfile.TemporaryDirectory() as tmp:
            with Pool(Path(tmp),8,{s['content_hash']:s for s in bank}) as pool:
                selected,ledger=collect_update(pool,p,plan,bank,999142,0,0,.95)
        fit={s['content_hash'] for s in bank if s['origin']['role']=='fit'};probe={s['content_hash'] for s in bank if s['origin']['role']=='probe'}
        self.assertTrue(all(g['prefix']['scenario_hash'] in fit and g['prefix']['scenario_hash'] not in probe for g in ledger['games']))
        self.assertEqual(len(selected['actions']),256);self.assertEqual(len(set(ledger['selected_indices'])),256)
        self.assertGreaterEqual(ledger['learner_rows'],256);self.assertLess(ledger['target_parity_error'],2e-6)

if __name__=='__main__':unittest.main()
