"""Reuse F4 paired cutoff intervals with F5's complete 27-contrast family."""
from pokemon_full_learning.statistics import compare


def contrasts(initial, finals, settings):
    return {'versus_initial': {arm: {op: compare(initial[op], games, settings)
        for op, games in panel.items()} for arm, panel in finals.items()},
        'mixture_versus_self_play': {op: compare(finals['self_play'][op], games, settings)
        for op, games in finals['mixture'].items()}}


def mixture_gate(results, settings):
    seeds = []
    for result in results:
        c = result['contrasts']; tactical = c['mixture_versus_self_play']['tactical']
        checks = {
            'tactical_vs_control_confidence_positive': tactical['approximate_familywise_bootstrap_interval'][0] > 0,
            'tactical_vs_control_practical_gain': tactical['cutoff_aware_point_interval'][0] >= settings['practical_improvement'],
            'tactical_vs_initial_confidence_positive': c['versus_initial']['mixture']['tactical']['approximate_familywise_bootstrap_interval'][0] > 0,
            'no_detected_opponent_regression': all(cell['gate'] != 'regression' for cell in c['mixture_versus_self_play'].values()),
            'all_final_cutoff_rates_admitted': all(cell['final']['cutoffs']/cell['final']['games'] <= settings['max_final_cutoff_rate']
                for arm in c['versus_initial'].values() for cell in arm.values())}
        seeds.append({'seed': result['seed'], 'checks': checks, 'passed': all(checks.values())})
    return {'passed': all(s['passed'] for s in seeds), 'per_seed': seeds,
            'scope': 'Exploratory approximate-bootstrap gate; matched optimization exposure, not matched collection cost or a public-strength certificate.'}
