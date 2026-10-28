"""F22 fixed budgets, role boundaries, strict checkpoints and battle gates."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.storage import ROOT, digest, save_checkpoint
from pokemon_full_train.learning import PPO, derived_seed
from pokemon_full_learning.campaign import evaluation_specs, check_games
from pokemon_full_learning.statistics import compare
from pokemon_diversity.common import read, receipt, team_keys
from pokemon_shared_battle.common import admit as f21_admit, load_policy
from pokemon_shared.learning import check_state, shapes
from pokemon_shared.common import same_state
from pokemon_long_transfer.value import summarize_value
PLAN = ROOT/'contracts/shared-ppo-v1/plan.json'


def validate_plan(p):
    fixed = {'seeds': [7103, 8209, 9311], 'arms': ['flat', 'shared'],
        'opponents': ['random', 'base_power', 'tactical'], 'workers': 8,
        'max_turns': 120, 'cohorts': 128, 'endpoints': [0, 64, 128],
        'selected_rows_per_update': 1024, 'training_chunk_games': 8,
        'maximum_games_per_update': 64, 'evaluation_pairs_per_opponent': 128,
        'evaluation_chunk_pairs': 4, 'selfplay_pairs': 16,
        'evaluation_seed': 1025262201, 'selfplay_evaluation_seed': 1025262202,
        'checkpoint_cohorts': [0,16,32,48,63,64,80,96,112,127,128],
        'recovery_indices': [63,127], 'ppo': PPO}
    if p != read(PLAN) or any(p.get(k) != v for k,v in fixed.items()):
        raise ValueError('F22 fixed schedule/budget mismatch')
    if p['statistics'] != {'method':'paired-cluster-percentile-bootstrap',
            'family_cells':45, 'alpha':.05, 'resamples':20000, 'seed':1025262203,
            'practical_improvement':.05, 'max_final_cutoff_rate':.05}:
        raise ValueError('Changed statistical family')
    if p['expected_counts'] != {'cohorts':768, 'optimizer_steps':6144,
            'selected_ppo_rows':786432, 'ppo_row_presentations':1572864,
            'evaluation_games':16128, 'selfplay_diagnostic_games':576,
            'checkpoint_comparisons':66, 'original_next_update_recoveries':12}:
        raise ValueError('Changed registered counts')
    if any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):
        raise ValueError('Forbidden scope')


def admit(plan):
    r = receipt(ROOT/plan['f21_receipt'], plan['f21_sha256'], 'f21-shared-battle')
    if r['status'] != 'passed':
        raise ValueError('F21 incomplete')
    previous = read(ROOT/'contracts/shared-battle-v1/plan.json')
    oracle, train, past = f21_admit(previous)
    base = ROOT/r['run']
    expected = {base/str(seed)/arm/'games'/f'{opponent}-{first:04d}.json'
        for seed in previous['seeds'] for arm in previous['arms']
        for opponent in previous['opponents'] for first in range(0,128,4)}
    if set(base.glob('*/*/games/*.json')) != expected or len(expected) != 864:
        raise ValueError('Incomplete F21 historical evaluation inventory')
    historical_games = [g for path in sorted(expected) for g in read(path)['games']]
    if len(historical_games) != 6912 or len(team_keys(historical_games)) != 256:
        raise ValueError('F21 historical game/team count mismatch')
    past.update(team_keys(historical_games))
    if train & past:
        raise ValueError('Historical role overlap')
    return previous, oracle, train, past


def initial(previous, seed, arm):
    p, _, identity = load_policy(previous, seed, arm)
    z = jax.tree.map(jnp.zeros_like, p)
    opt = (z, z, jnp.int32(0))
    check_state(p, opt, arm)
    return p, opt, identity


def metadata(h, seed, arm, cohort, cursor):
    return {'input_hash': h, 'seed': seed, 'architecture': arm,
        'cohorts': cohort, 'next_episode': cursor, 'optimizer_step': 8*cohort}


def save(folder, p, opt, h, seed, arm, cohort, cursor):
    check_state(p, opt, arm)
    if int(opt[2]) != 8*cohort:
        raise ValueError('Save counter mismatch')
    return save_checkpoint(folder, p, opt, metadata(h,seed,arm,cohort,cursor))


def load(folder, h, seed, arm, cohort, cursor):
    pointer = read(folder/'latest.json')
    if set(pointer) != {'file','sha256'} or Path(pointer['file']).name != pointer['file']:
        raise ValueError('Checkpoint pointer schema')
    path = folder/pointer['file']
    if digest(path) != pointer['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        expected = {'metadata'} | {pre+'__'+k for pre in ['p','m','v'] for k in shapes(arm)}
        if set(z.files) != expected or z['metadata'].dtype != np.uint8 or z['metadata'].ndim != 1:
            raise ValueError('Checkpoint inventory')
        m = json.loads(z['metadata'].tobytes())
        if m != metadata(h,seed,arm,cohort,cursor) or any(type(m[k]) is not int for k in ['seed','cohorts','next_episode','optimizer_step']):
            raise ValueError('Checkpoint identity')
        groups = [{k:z[pre+'__'+k] for k in shapes(arm)} for pre in ['p','m','v']]
        check_state(groups[0], (groups[1],groups[2],np.int32(8*cohort)), arm)
        groups = [{k:jnp.asarray(a) for k,a in g.items()} for g in groups]
    return groups[0], (groups[1],groups[2],jnp.int32(8*cohort))


def diagnostic_specs(plan, seed, first, count):
    modified = {**plan, 'evaluation_seed': plan['selfplay_evaluation_seed']}
    return evaluation_specs(modified,seed,'self_play',first,count)


def check_collection(collection, first, count, split, opponent=None):
    games = [g for _,g in collection]
    check_games(games,split=split,first=first,count=count,opponent=opponent)
    for rows,g in collection:
        if (split == 'train' and len(rows) != g['decisions']) or (split == 'development' and rows):
            raise ValueError('Training/development learner-row boundary')
        controlled = [r for r in g['trace'] if opponent is None or r[0] == g['learner_side']]
        if len(controlled) != len(g['actor_trace']):
            raise ValueError('Actor record count')
        for row,a in zip(controlled,g['actor_trace']):
            if row != [a['side'],a['request_id'],a['action'],a['value'],a['feature_sha256']]:
                raise ValueError('Actor record linkage')
            mask = np.asarray(a['mask'])
            if mask.shape != (14,) or mask.dtype != np.bool_ or not 0 <= a['action'] < 14 or not mask[a['action']]:
                raise ValueError('Invalid legal action')
            if not np.isfinite(a['log_probability']) or a['log_probability'] > 1e-6 or not 0 <= a['value'] <= 1:
                raise ValueError('Invalid likelihood/value')
        if type(g['unavailable_choices']) is not int or not 0 <= g['unavailable_choices'] <= g['decisions']:
            raise ValueError('Unavailable-choice count')
    return games


def selfplay_value(games):
    # Each terminal contributes one trace per seat. Errors stay descriptive;
    # the two seats are not independent games.
    views = []
    for g in games:
        for side in ['p1','p2']:
            trace = [[s,r,a,v if s==side else None,h] for s,r,a,v,h in g['trace']]
            views.append({**g,'learner_side':side,'trace':trace})
    return {'physical_games':len(games), 'seat_views':summarize_value(views)}


def analysis(panels, reference, plan):
    pairs = [('flat_learning',panels['flat']['0'],panels['flat']['128']),
        ('shared_learning',panels['shared']['0'],panels['shared']['128']),
        ('shared_vs_flat',panels['flat']['128'],panels['shared']['128']),
        ('flat_vs_reference',reference,panels['flat']['128']),
        ('shared_vs_reference',reference,panels['shared']['128'])]
    comparisons = {name:{o:compare(a[o],b[o],plan['statistics']) for o in plan['opponents']} for name,a,b in pairs}
    def gate(cells, primary, others):
        p = cells[primary]; s = plan['statistics']
        checks = {'point_gain':p['cutoff_aware_point_interval'][0]>=s['practical_improvement'],
            'adjusted_lower_positive':p['approximate_familywise_bootstrap_interval'][0]>0,
            'no_detected_regression':all(cells[o]['gate']!='regression' for o in others),
            'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cells.values())}
        return {'checks':checks,'passed':all(checks.values())}
    return {'comparisons':comparisons,
        'learning':{a:gate(comparisons[a+'_learning'],'base_power',plan['opponents']) for a in plan['arms']},
        'architecture_benefit':gate(comparisons['shared_vs_flat'],'tactical',['random','base_power'])}
