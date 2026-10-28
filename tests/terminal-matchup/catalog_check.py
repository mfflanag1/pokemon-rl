import json,subprocess,unittest
import numpy as np
from pokemon_terminal_matchup.common import ROOT,read,PLAN,initial,tactical
from pokemon_terminal_matchup.features import validate_catalog

class Catalog(unittest.TestCase):
    def test_entire_pinned_dex_catalog_reconstruction(self):
        catalog=read(ROOT/read(PLAN)['catalog'])
        result=subprocess.run(['node','packages/switch-audit/catalog.mjs'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),catalog)
        validate_catalog(catalog)
        self.assertEqual(len(catalog['species']),3476);self.assertEqual(len(catalog['moves']),939)

    def test_all_512_real_oracle_players_both_arms_all_seeds(self):
        catalog=read(ROOT/read(PLAN)['catalog']);oracle=read(ROOT/'artifacts/runs/f20-shared-policy-v1/inputs/tactical-oracle.json')
        for seed in [7103,8209,9311]:
            p,_,_=initial(None,seed,'zero');a=tactical(p,oracle,'zero',catalog);b=tactical(p,oracle,'matchup',catalog)
            self.assertEqual(a,b)
            original=read(ROOT/f'artifacts/runs/f22-shared-ppo-v1/{seed}/flat/128/tactical.json')
            np.testing.assert_allclose([x['probabilities'] for x in a['predictions']],
                [x['probabilities'] for x in original['predictions']],atol=3e-7,rtol=3e-6)
            self.assertEqual(len(a['cells']),32)

if __name__=='__main__':unittest.main()
