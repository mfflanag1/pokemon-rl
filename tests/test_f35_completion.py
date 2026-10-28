"""Small synthetic completion records; no games, JAX or scientific workflow."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('f35_completion', ROOT / 'scripts/check-teacher-assisted.py')
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def value(games):
    return {'games': games, 'terminals': games, 'cutoffs': 0, 'decisions': games * 2,
        'learner_wins': games // 2, 'learner_losses': games // 2, 'true_ties': games % 2,
        'terminal_game_mean_value_mse': .25, 'game_mean_predicted_value': .5}


def fixture():
    plan = json.loads((ROOT / 'contracts/teacher-assisted-v1/plan.json').read_text())
    stages = [{'stage': s, 'pid': i + 100, 'exit_code': 0, 'deadline_timeout': False,
               'cleanup_status': 'passed', 'leaked_after_success': False, 'wait_error': None}
              for i, s in enumerate(checker.STAGES)]
    cleanups = {r['stage']: {'status': 'passed', 'root_pid': r['pid'], 'returncode': 0,
        'stage_was_running': False, 'remaining_pids': [], 'errors': [], 'identity_changed_skipped': [],
        'live_descendants_at_cleanup': [], 'initial_live_pids': [], 'signals': []} for r in stages}
    receipt = {'status': 'completed', 'input_hash': 'a' * 64, 'stages': stages}
    manifest = {'plan': plan}
    counts = {k: v for k, v in checker.COUNTS.items() if k not in
        ('external_evaluation_games', 'all_evaluation_games', 'teacher_train_games', 'teacher_probe_games')}
    counts['training_games'] = 144
    summary = {'status': 'passed', 'input_hash': receipt['input_hash'], 'counts': counts,
        'external_counts': {'games': 1152, 'terminals': 1152, 'cutoffs': 0, 'decisions': 2304},
        'extra_demonstration_presentations': 1572864, 'verification': dict(checker.VERIFICATION),
        'teacher_roles': {role: {'games': n, 'accepted_labels': n * 2, 'terminal_games': n, 'cutoff_games': 0}
            for role, n in [('train', 384), ('probe', 96)]},
        'contrasts': 45, 'all_seed_primary_passed': False,
        'all_seed_teacher_adaptation_passed': False, 'seeds': {}}
    for seed in checker.SEEDS:
        endpoints = {e: {o: value(n) for o, n in checker.GAMES.items()} for e in checker.ENDPOINTS}
        accounting = {a: {'counts': dict(checker.ARM_COUNTS), 'training': value(24),
            'streams': {s: {'selected_rows': n, 'collected_rows': n + 8, 'excluded_rows': 8, 'games': 8}
                for s, n in checker.STREAM_ROWS.items()}} for a in checker.ARMS}
        contrasts = {}
        for opponent, games in checker.GAMES.items():
            contrasts[opponent] = {}
            for name, (before, after) in checker.CONTRASTS.items():
                family = 'six primary/teacher contrasts' if name == 'assisted_vs_control' and opponent in ('baseline-v1', 'foul_play') else '39 diagnostic contrasts'
                contrasts[opponent][name] = {'before_endpoint': before, 'after_endpoint': after,
                    'initial': copy.deepcopy(endpoints[before][opponent]),
                    'final': copy.deepcopy(endpoints[after][opponent]), 'pairs': games // 2,
                    'multiplicity_family': family, 'cutoff_aware_point_interval': [0., 0.],
                    'approximate_familywise_bootstrap_interval': [0., 0.]}
        gate = {'checks': {'practical_gain': False, 'adjusted_lower_positive': False,
                          'both_final_cutoffs_admitted': True}, 'passed': False}
        summary['seeds'][seed] = {'endpoints': endpoints, 'accounting': accounting,
            'analysis': {'contrasts': contrasts, 'primary_gate': copy.deepcopy(gate),
                         'teacher_adaptation_gate': copy.deepcopy(gate)}}
    completion = {'status': 'passed', 'input_hash': receipt['input_hash']}
    workflow_exit = {**completion, 'exit_code': 0}
    return receipt, manifest, summary, completion, workflow_exit, {r['stage']: copy.deepcopy(r) for r in stages}, cleanups


class CompletionTests(unittest.TestCase):
    def test_complete_comparison_with_failed_learning_gates_is_valid(self):
        data = fixture()
        checker.validate_completion(*data)
        self.assertEqual(data[1]['plan']['expected_counts'], checker.COUNTS)
        self.assertEqual(sum(checker.GAMES.values()) * 9, 8064)

    def test_every_fixed_training_counter_is_enforced(self):
        for key in ('cohorts', 'optimizer_steps', 'selected_rows', 'row_presentations',
                    'demonstration_presentations', 'evaluation_games', 'checkpoint_comparisons',
                    'original_next_update_recoveries', 'training_games'):
            with self.subTest(key=key):
                data = fixture()
                data[2]['counts'][key] -= 1
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_missing_seed_arm_endpoint_opponent_and_contrast_are_rejected(self):
        paths = [('seeds', '7103'), ('seeds', '7103', 'accounting', 'control'),
            ('seeds', '7103', 'endpoints', 'assisted'),
            ('seeds', '7103', 'endpoints', 'initial', 'random'),
            ('seeds', '7103', 'analysis', 'contrasts', 'foul_play', 'assisted_vs_control')]
        for path in paths:
            with self.subTest(path=path):
                data = fixture()
                current = data[2]
                for key in path[:-1]:
                    current = current[key]
                del current[path[-1]]
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_all_required_stage_exits_must_succeed_in_order(self):
        for mutation in ('missing', 'duplicate', 'reorder', 'timeout', 'nonzero', 'unlinked', 'missing_exit'):
            with self.subTest(mutation=mutation):
                data = fixture()
                stage = data[0]['stages'][0]
                if mutation == 'missing':
                    data[0]['stages'].pop()
                elif mutation == 'duplicate':
                    data[0]['stages'][1] = copy.deepcopy(stage)
                elif mutation == 'reorder':
                    data[0]['stages'].reverse()
                elif mutation == 'timeout':
                    stage['deadline_timeout'] = data[5][stage['stage']]['deadline_timeout'] = True
                elif mutation == 'nonzero':
                    stage['exit_code'] = data[5][stage['stage']]['exit_code'] = 1
                elif mutation == 'unlinked':
                    data[5][stage['stage']]['extra'] = True
                else:
                    del data[5][stage['stage']]
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_teacher_external_and_every_verifier_count_are_enforced(self):
        for role in ('train', 'probe'):
            data = fixture()
            data[2]['teacher_roles'][role]['games'] -= 1
            with self.assertRaises(ValueError):
                checker.validate_completion(*data)
        for key in checker.VERIFICATION:
            with self.subTest(key=key):
                data = fixture()
                data[2]['verification'][key] -= 1
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)
        data = fixture()
        data[2]['external_counts']['games'] = 1150
        with self.assertRaises(ValueError):
            checker.validate_completion(*data)

    def test_cleanup_cannot_claim_pass_with_leaks_errors_skips_or_wrong_stage(self):
        mutations = {'status': 'failed', 'root_pid': 9999, 'returncode': 1, 'stage_was_running': True,
            'remaining_pids': [42], 'errors': ['failed'], 'identity_changed_skipped': [42],
            'live_descendants_at_cleanup': [42], 'initial_live_pids': [42], 'signals': [{'pid': 42}]}
        for key, bad in mutations.items():
            with self.subTest(key=key):
                data = fixture()
                data[6]['training'][key] = bad
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)
        data = fixture()
        del data[6]['training']
        with self.assertRaises(ValueError):
            checker.validate_completion(*data)

    def test_stage_failure_flags_cannot_be_hidden_by_passed_cleanup(self):
        for key, bad in [('cleanup_status', 'failed'), ('leaked_after_success', True), ('wait_error', 'interrupted')]:
            with self.subTest(key=key):
                data = fixture()
                data[0]['stages'][0][key] = bad
                data[5]['fixed-schedules'][key] = bad
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_pair_counts_directions_and_multiplicity_families_are_enforced(self):
        for key, bad in [('pairs', 255), ('before_endpoint', 'initial'),
                         ('multiplicity_family', '39 diagnostic contrasts')]:
            with self.subTest(key=key):
                data = fixture()
                data[2]['seeds']['7103']['analysis']['contrasts']['baseline-v1']['assisted_vs_control'][key] = bad
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_stream_and_endpoint_accounting_cannot_hide_a_missing_cell(self):
        data = fixture()
        data[2]['seeds']['7103']['accounting']['assisted']['streams']['tactical']['selected_rows'] -= 1
        with self.assertRaises(ValueError):
            checker.validate_completion(*data)
        data = fixture()
        data[2]['seeds']['7103']['endpoints']['initial']['foul_play']['games'] -= 1
        with self.assertRaises(ValueError):
            checker.validate_completion(*data)

    def test_terminal_and_cutoff_accounting_and_value_scope_are_enforced(self):
        for field, bad in [('terminals', 511), ('learner_wins', 0),
                           ('terminal_game_mean_value_mse', None), ('game_mean_predicted_value', float('nan'))]:
            with self.subTest(field=field):
                data = fixture()
                data[2]['seeds']['7103']['endpoints']['initial']['baseline-v1'][field] = bad
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)

    def test_completion_linkage_and_claimed_gate_cannot_be_rewritten(self):
        for mutation in ('input', 'exit', 'gate', 'all_seed', 'budget'):
            with self.subTest(mutation=mutation):
                data = fixture()
                if mutation == 'input':
                    data[3]['input_hash'] = 'wrong'
                elif mutation == 'exit':
                    data[4]['exit_code'] = 1
                elif mutation == 'gate':
                    data[2]['seeds']['7103']['analysis']['primary_gate']['passed'] = True
                elif mutation == 'all_seed':
                    data[2]['all_seed_primary_passed'] = True
                else:
                    data[1]['plan']['expected_counts']['cohorts'] -= 1
                with self.assertRaises(ValueError):
                    checker.validate_completion(*data)


if __name__ == '__main__':
    unittest.main()
