import copy
import unittest
from pokemon_full_train.storage import ROOT
from pokemon_long_transfer.common import read,validate_plan
from pokemon_full_learning.campaign import evaluation_specs


class Plan(unittest.TestCase):
    def test_fixed_budget_and_fresh_panel(self):
        plan=read(ROOT/'contracts/long-transfer-v1/plan.json');old=read(ROOT/'contracts/full-transfer-v2/plan.json');validate_plan(plan)
        self.assertEqual(plan['updates_per_arm'],old['updates_per_arm']*8)
        for field in ['seeds','arms','coefficients','selected_rows_per_update','max_turns','tactical_thresholds','predecessor_checkpoint']:
            self.assertEqual(plan[field],old[field])
        self.assertNotEqual(plan['evaluation_seed'],old['evaluation_seed'])
        self.assertNotEqual(plan['training_seed_namespace'],old['training_seed_namespace'])
        jobs=evaluation_specs(plan,7103,'tactical',252,4)
        self.assertEqual([(s['episode'],s['swap']) for s in jobs],[(i,swap) for i in range(252,256) for swap in [False,True]])
        self.assertTrue(all(j['split']=='development' and j['seed']==plan['evaluation_seed'] for j in jobs))

    def test_budget_seed_and_public_scope_drift_rejected(self):
        plan=read(ROOT/'contracts/long-transfer-v1/plan.json')
        for key,value in [('updates_per_arm',129),('seeds',[7103]),('public_server_used',True),('final_reserved_data_used',True),('evaluation_pairs_per_opponent',255),('coefficients',{'plain':0.,'rehearsal':.2})]:
            candidate=copy.deepcopy(plan);candidate[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_plan(candidate)


if __name__=='__main__':unittest.main()
