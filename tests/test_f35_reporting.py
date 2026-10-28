import copy
import unittest

from pokemon_teacher_assisted.common import read, validate_plan
from pokemon_teacher_assisted.reporting import analyze, gate
from pokemon_teacher_assisted.schedules import check_records
from pokemon_full_learning.statistics import compare


def games(score, opponent='baseline-v1', cutoff=False):
    return [{'episode': i, 'swap': swap, 'learner_side': 'p2' if swap else 'p1',
        'opponent': opponent, 'provenance': {'pair': i, 'swap': swap}, 'decisions': 3,
        'value_mean': .5, 'terminal_value_mse': .25 if not cutoff else None,
        'outcome': {'kind': 'administrative_cutoff' if cutoff else 'terminal',
            'rewards': {'p1': score if not swap else 1 - score, 'p2': 1 - score if not swap else score}}}
        for i in range(16) for swap in (False, True)]


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'seed': 1, 'resamples': 128, 'alpha': .05, 'family_cells': 6,
            'diagnostic_family_cells': 39, 'practical_improvement': .05, 'max_final_cutoff_rate': .05}

    def test_same_schedule_treatment_contrast_and_gate(self):
        result = compare(games(0), games(1), self.settings)
        self.assertEqual(result['cutoff_aware_point_interval'], [1., 1.])
        self.assertTrue(gate(result, self.settings)['passed'])
        reverse = compare(games(1), games(0), self.settings)
        self.assertEqual(reverse['gate'], 'regression')

    def test_cutoffs_remain_unresolved_and_never_become_losses(self):
        result = compare(games(0, cutoff=True), games(0, cutoff=True), self.settings)
        self.assertEqual(result['cutoff_aware_point_interval'], [-1., 1.])
        self.assertFalse(gate(result, self.settings)['passed'])
        self.assertEqual(result['initial']['terminals'], 0)

    def test_mismatched_pair_and_missing_arm_fail(self):
        final = games(1)
        final[2]['provenance']['pair'] = 99
        with self.assertRaises(ValueError):
            compare(games(0), final, self.settings)
        with self.assertRaises(ValueError):
            analyze({'initial': {'baseline-v1': games(0)}, 'assisted': {'baseline-v1': games(1)}}, self.settings)

    def test_all_fifteen_seed_contrasts_and_two_families_retained(self):
        opponents = ('baseline-v1', 'foul_play', 'tactical', 'base_power', 'random')
        panels = {endpoint: {o: games(score, o) for o in opponents}
                  for endpoint, score in [('initial', 0), ('control', 0), ('assisted', 1)]}
        result = analyze(panels, self.settings)
        self.assertEqual(sum(len(v) for v in result['contrasts'].values()), 15)
        families = [v['multiplicity_family'] for cells in result['contrasts'].values() for v in cells.values()]
        self.assertEqual(families.count('six primary/teacher contrasts'), 2)
        self.assertEqual(families.count('39 diagnostic contrasts'), 13)

    def test_schedule_intersection_blocks_instead_of_resampling(self):
        jobs = [{'group': role, 'split': 'development', 'seed': i, 'episode': 0, 'swap': False}
                for i, role in enumerate(('teacher_probe', 'evaluation'))]
        records = [{**job, 'teams': [{'team_key': 'a' * 64, 'partition': 'development'},
            {'team_key': str(i) * 64, 'partition': 'development'}]} for i, job in enumerate(jobs)]
        with self.assertRaisesRegex(ValueError, 'intersection'):
            check_records(jobs, records, set())

    def test_plan_budget_change_rejected(self):
        plan = read('contracts/teacher-assisted-v1/plan.json')
        validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed['cohorts_per_seed_arm'] += 1
        with self.assertRaises(ValueError):
            validate_plan(changed)


if __name__ == '__main__':
    unittest.main()
