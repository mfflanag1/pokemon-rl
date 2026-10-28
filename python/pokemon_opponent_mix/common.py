"""F26 fixed opponent exposure, budgets, role boundaries, strict checkpoints and battle gates."""
import json
import hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.storage import ROOT, digest, save_checkpoint
from pokemon_full_train.learning import PPO, derived_seed
from pokemon_full_learning.campaign import evaluation_specs, check_games
from pokemon_full_learning.statistics import compare
from pokemon_diversity.common import read, receipt, team_keys
from pokemon_shared_battle.common import load_policy
from pokemon_shared_ppo.common import admit as f22_admit, load as f22_load
from pokemon_full.learner import parameter_hash
from pokemon_matchup.network import extend,SHAPES,check_state as model_state,forward
from pokemon_terminal_matchup.features import encode

from pokemon_shared.common import same_state
from pokemon_long_transfer.value import summarize_value
PLAN = ROOT/'contracts/opponent-mix-v1/plan.json'

def shapes(arm):
    if arm not in ['self','mixed']:raise ValueError('Unknown arm')
    return SHAPES

def check_state(p,opt,arm):
    shapes(arm);model_state(p,opt)


def validate_plan(p):
    if p != read(PLAN) or p['seeds'] != [7103,8209,9311] or p['arms'] != ['self','mixed']:
        raise ValueError('Fixed plan/seeds/arms')
    fixed={'cohorts':128,'endpoints':[0,64,128],'workers':8,'max_turns':120,
        'selected_rows_per_update':1024,'training_chunk_games':8,'maximum_games_per_update':64,
        'evaluation_pairs_per_opponent':128,'evaluation_chunk_pairs':4,'selfplay_pairs':16,
        'evaluation_seed':1025262601,'selfplay_evaluation_seed':1025262602,
        'checkpoint_cohorts':[0,63,64,127,128],'recovery_indices':[63,127],
        'ppo':PPO,'features':1278,'extra_features':350}
    if any(p.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed budget/schedule')
    if p['statistics']!={'method':'paired-cluster-percentile-bootstrap','family_cells':27,
        'alpha':.05,'resamples':20000,'seed':1025262603,'practical_improvement':.05,'max_final_cutoff_rate':.05}:
        raise ValueError('Statistical family')
    if p['expected_counts']!={'cohorts':768,'optimizer_steps':6144,'selected_ppo_rows':786432,
        'ppo_row_presentations':1572864,'evaluation_games':13824,'selfplay_diagnostic_games':576,
        'checkpoint_comparisons':30,'original_next_update_recoveries':12}:raise ValueError('Expected counts')
    if any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):
        raise ValueError('Forbidden scope')


def admit(plan):
    from pokemon_terminal_matchup.common import admit as f25_admit
    previous,oracle,train,past=f25_admit(read(ROOT/'contracts/terminal-matchup-v1/plan.json'))
    prior=receipt(ROOT/plan['f25_receipt'],plan['f25_sha256'],'f25-terminal-matchup')
    if prior['status']!='passed' or prior['counts']['cohorts']!=3072:
        raise ValueError('F25 predecessor completion')
    run=ROOT/prior['run']; n=0
    for path in sorted(run.glob('*/*/update-*.json')):
        train.update(team_keys(read(path)['ledger']['games']));n+=1
    if n!=3072:raise ValueError('F25 training history incomplete')
    for pattern in ['*/*/games/*.json','*/*/*/games/*.json','*/*/*/selfplay/*.json']:
        for path in sorted(run.glob(pattern)):
            past.update(team_keys(read(path)['games']))
    if train & past:raise ValueError('Historical role collision')
    return previous,oracle,train,past


def initial(previous,seed,arm):
    from pokemon_terminal_matchup.common import load as f25_load
    shapes(arm);base=ROOT/'artifacts/runs/f25-terminal-matchup-v1'
    cursor=read(base/'report.json')['seeds'][str(seed)]['arms']['matchup']['final_next_episode']
    folder=base/str(seed)/'matchup/checkpoints/512';h=digest(base/'inputs/manifest.json')
    p,oldopt=f25_load(folder,h,seed,'matchup',512,cursor)
    zeros={k:jnp.zeros_like(v) for k,v in p.items()}
    opt=(zeros,{k:jnp.zeros_like(v) for k,v in p.items()},jnp.int32(0))
    check_state(p,opt,arm)
    return p,opt,{'seed':seed,'source':str(folder.relative_to(ROOT)),
        'pointer_sha256':digest(folder/'latest.json'),'parameter_hash':parameter_hash(p),
        'source_optimizer_step':int(oldopt[2]),'optimizer_reset':True}


def tactical(p,oracle,arm,catalog):
    if len(oracle['records'])!=512:raise ValueError('Familiar probe inventory')
    from pokemon_full.policy import distribution
    groups={};predictions=[]
    for first in range(0,512,8):
        records=oracle['records'][first:first+8];enc=[encode(r['player'],catalog,arm) for r in records]
        logits,values=forward(p,jnp.asarray(np.stack([x for x,_ in enc])))
        probs,_=distribution(logits,jnp.asarray(np.stack([m for _,m in enc])))
        for r,pr,value in zip(records,np.asarray(probs),np.asarray(values)):
            item={'target_mass':float(pr[r['target']]),'greedy_correct':int(pr.argmax())==r['target'],
                'expected_local_reward':sum(float(pr[int(a)])*v for a,v in r['scores'].items()),
                'probabilities':pr.tolist(),'full_battle_value':float(value)}
            predictions.append(item);groups.setdefault(f"{r['spec']['panel']}/{r['spec']['context']}",[]).append(item)
    return {'predictions':predictions,'cells':{k:{'n':len(v),
        'target_mass':float(np.mean([r['target_mass'] for r in v])),
        'greedy_accuracy':float(np.mean([r['greedy_correct'] for r in v])),
        'expected_local_reward':float(np.mean([r['expected_local_reward'] for r in v]))} for k,v in groups.items()}}


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
        if (split == 'train' and len(rows) != len(g['actor_trace'])) or (split == 'development' and rows):
            raise ValueError('Training/development learner-row boundary')
        controlled = [r for r in g['trace'] if g['opponent'] is None or r[0] == g['learner_side']]
        if split == 'train':
            expected_seat='p2' if g['episode']%2 else 'p1'
            if g['learner_side']!=expected_seat:raise ValueError('Unbalanced training seat schedule')
            for row,a in zip(rows,g['actor_trace']):
                if row['trajectory']!=['p1','p2'].index(a['side']) or row['action']!=a['action'] or row['old_logp']!=a['log_probability'] or row['value']!=a['value'] or hashlib.sha256(row['features'].tobytes()).hexdigest()!=a['feature_sha256'] or row['mask'].tolist()!=a['mask']:
                    raise ValueError('Non-policy row or likelihood in PPO')
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


def analysis(panels, plan):
    pairs = [('self_learning',panels['self']['0'],panels['self']['128']),
        ('mixed_learning',panels['mixed']['0'],panels['mixed']['128']),
        ('mixed_vs_self',panels['self']['128'],panels['mixed']['128'])]
    comparisons = {name:{o:compare(a[o],b[o],plan['statistics']) for o in plan['opponents']} for name,a,b in pairs}
    def gate(cells, primary='base_power'):
        p = cells[primary]; s = plan['statistics']
        checks = {'point_gain':p['cutoff_aware_point_interval'][0]>=s['practical_improvement'],
            'adjusted_lower_positive':p['approximate_familywise_bootstrap_interval'][0]>0,
            'no_detected_regression':all(cells[o]['gate']!='regression' for o in ['random','base_power']),
            'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cells.values())}
        return {'checks':checks,'passed':all(checks.values())}
    return {'comparisons':comparisons,
        'learning':{a:gate(comparisons[a+'_learning']) for a in plan['arms']},
        'mixture_benefit':gate(comparisons['mixed_vs_self']),
        'tactical_adaptation':{a:gate(comparisons[a+'_learning'],'tactical') for a in plan['arms']},
        'tactical_adaptation_benefit':gate(comparisons['mixed_vs_self'],'tactical')}
