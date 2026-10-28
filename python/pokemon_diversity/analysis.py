"""Preregistered descriptive gates and whole-game paired accuracy intervals."""
import math
import numpy as np
from pokemon_balanced.sampling import classify
from pokemon_balanced.learning import NAMES
from pokemon_full_train.learning import derived_seed
from pokemon_imitation.data import summarize


def paired_accuracy(before, after, data, stats, seed):
    summarize(before,data['mask'],data['targets'],data['game_index'])
    summarize(after,data['mask'],data['targets'],data['game_index'])
    classes = classify(data); result = {}
    delta = (after.argmax(1)==data['targets']).astype(np.float64)-(before.argmax(1)==data['targets'])
    for code,name in enumerate(NAMES):
        ids = np.flatnonzero(classes==code)
        games = np.unique(data['game_index'][ids])
        if not len(games):
            raise ValueError('Missing contrast family')
        values = np.asarray([delta[ids[data['game_index'][ids]==g]].mean() for g in games])
        rng = np.random.default_rng(derived_seed('f18-paired',stats['seed'],seed,name))
        draws = values[rng.integers(0,len(values),(stats['resamples'],len(values)))].mean(1)
        tail = stats['alpha']/(2*stats['contrasts'])
        point = float(values.mean()); radius = math.sqrt(2*math.log(2*stats['contrasts']/stats['alpha'])/len(values))
        result[name] = {'games':len(games),'rows':len(ids),'paired_game_accuracy_difference':point,
            'approximate_adjusted_bootstrap_interval':np.quantile(draws,[tail,1-tail]).tolist(),
            'hoeffding_interval':[max(-1.,point-radius),min(1.,point+radius)],
            'game_ids':games.tolist(),'per_game_differences':values.tolist()}
    return result


def gate(arms, plan):
    original = arms['original']['probe']['families']; expanded = arms['expanded']['probe']['families']
    thresholds = plan['coverage_thresholds']
    checks = {'voluntary_accuracy':expanded['voluntary_switch']['game_accuracy']>=thresholds['voluntary_min_accuracy'],
        'voluntary_gain':expanded['voluntary_switch']['game_accuracy']-original['voluntary_switch']['game_accuracy']>=thresholds['voluntary_min_gain'],
        **{name+'_preserved':original[name]['game_accuracy']-expanded[name]['game_accuracy']<=thresholds['maximum_other_accuracy_drop'] for name in ['normal_move','tera_move','forced_switch']},
        'known_contexts_retained':arms['expanded']['retention']['passed']}
    return {'checks':checks,'passed':all(checks.values())}


def team_admission(original_keys, added_keys, probe_keys, past_development):
    if original_keys & added_keys or probe_keys & (original_keys | added_keys | past_development) or added_keys & past_development:
        raise ValueError('Exact team overlap between disallowed roles')
    return {'original_train_teams':len(original_keys),'added_train_teams':len(added_keys),
        'fresh_probe_teams':len(probe_keys),'prior_development_teams_excluded':len(past_development),'disallowed_overlap':0}
