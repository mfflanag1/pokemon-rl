"""F29 fixed batch budgets, inherited game admission and arm-bound checkpoints."""
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.storage import ROOT,digest,save_checkpoint
from pokemon_full_train.learning import PPO
from pokemon_full_learning.statistics import compare
from pokemon_full.learner import parameter_hash
from .network import SHAPES,check_state as model_state,extend,forward
from pokemon_full_train.learning import derived_seed
from pokemon_terminal_matchup.features import encode
from pokemon_diversity.common import read,receipt,team_keys
from pokemon_opponent_mix.common import (load_policy,same_state,evaluation_specs,
    diagnostic_specs,check_collection,selfplay_value)
PLAN=ROOT/'contracts/nonlinear-policy-v1/plan.json'
BUDGETS={'control': {'cohorts': 256, 'selected_rows': 1024, 'optimizer_steps_per_cohort': 8, 'maximum_games_per_stream': 64, 'endpoints': [0, 128, 256], 'checkpoint_cohorts': [0, 127, 128, 255, 256], 'recovery_indices': [127, 255]}, 'nonlinear': {'cohorts': 256, 'selected_rows': 1024, 'optimizer_steps_per_cohort': 8, 'maximum_games_per_stream': 64, 'endpoints': [0, 128, 256], 'checkpoint_cohorts': [0, 127, 128, 255, 256], 'recovery_indices': [127, 255]}}
def budget(arm):
    if arm not in BUDGETS:raise ValueError('Unknown batch arm')
    return BUDGETS[arm]
def shapes(arm):
    budget(arm);return SHAPES
def check_state(p,opt,arm):
    budget(arm);model_state(p,opt)
def validate_plan(p):
    if p!=read(PLAN) or p['seeds']!=[7103,8209,9311] or p['arms']!=['control','nonlinear']:
        raise ValueError('Fixed plan/seeds/arms')
    fixed=dict(arm_budgets=BUDGETS,workers=8,max_turns=120,training_chunk_games=8,
        evaluation_pairs_per_opponent=128,evaluation_chunk_pairs=4,selfplay_pairs=16,
        evaluation_seed=1025262901,selfplay_evaluation_seed=1025262902,ppo=PPO,features=1278,extra_features=350,residual_width=32,parameter_count=63718,added_policy_parameters=2912,primary_opponent='tactical',initialization_seed_namespace='f29-residual-initialization')
    if any(p.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed budget/schedule')
    if p['statistics']!=dict(method='paired-cluster-percentile-bootstrap',family_cells=27,
        alpha=.05,resamples=20000,seed=1025262903,practical_improvement=.05,max_final_cutoff_rate=.05):
        raise ValueError('Statistical family')
    if p['expected_counts']!=dict(cohorts=1536,optimizer_steps=12288,selected_ppo_rows=1572864,
        ppo_row_presentations=3145728,evaluation_games=13824,selfplay_diagnostic_games=576,
        checkpoint_comparisons=30,original_next_update_recoveries=12):raise ValueError('Expected counts')
    if any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):
        raise ValueError('Forbidden scope')
def admit(plan):
    from pokemon_batch_collection.common import admit as f28_admit
    previous,oracle,train,past=f28_admit(read(ROOT/'contracts/batch-collection-v1/plan.json'))
    prior=receipt(ROOT/plan['f28_receipt'],plan['f28_sha256'],'f28-batch-collection')
    if prior['status']!='passed' or prior['counts']['cohorts']!=432:raise ValueError('F28 completion')
    run=ROOT/prior['run'];n=0
    for path in sorted(run.glob('*/*/update-*.json')):
        train.update(team_keys(read(path)['ledger']['games']));n+=1
    if n!=432:raise ValueError('F28 training history incomplete')
    for pattern in ['*/*/*/games/*.json','*/*/*/selfplay/*.json']:
        for path in sorted(run.glob(pattern)):past.update(team_keys(read(path)['games']))
    preflight=ROOT/'artifacts/reviews/f29-nonlinear-policy/preflight'
    if (preflight/'report.json').exists():
        evidence=sorted(preflight.glob('*/*/result.json'))
        if len(evidence)!=6 or read(preflight/'report.json')['status']!='passed':raise ValueError('Preflight history incomplete')
        for path in evidence:train.update(team_keys(read(path)['ledger']['games']))
    if train & past:raise ValueError('Historical role collision')
    return previous,oracle,train,past

def initial(previous,seed,arm):
    from pokemon_opponent_mix.common import initial as f26_initial
    budget(arm)
    p,_,identity=f26_initial(previous,seed,'mixed')
    init_seed=derived_seed('f29-residual-initialization',seed)
    q,opt=extend(p,init_seed)
    if sum(a.size for a in q.values())!=63718:raise ValueError('Augmented parameter count')
    return q,opt,{**identity,'predecessor_parameter_hash':identity['parameter_hash'],
        'parameter_hash':parameter_hash(q),'residual_initialization_seed':init_seed,
        'parameters':sum(a.size for a in q.values()),'zero_output_initialization':True}


def metadata(h, seed, arm, cohort, cursor):
    return {'input_hash': h, 'seed': seed, 'architecture': arm,
        'cohorts': cohort, 'next_episode': cursor, 'selected_rows':budget(arm)['selected_rows']*cohort, 'optimizer_step': budget(arm)['optimizer_steps_per_cohort']*cohort}


def save(folder, p, opt, h, seed, arm, cohort, cursor):
    check_state(p, opt, arm)
    if int(opt[2]) != budget(arm)['optimizer_steps_per_cohort']*cohort:
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
        if m != metadata(h,seed,arm,cohort,cursor) or any(type(m[k]) is not int for k in ['seed','cohorts','next_episode','optimizer_step','selected_rows']):
            raise ValueError('Checkpoint identity')
        groups = [{k:z[pre+'__'+k] for k in shapes(arm)} for pre in ['p','m','v']]
        check_state(groups[0], (groups[1],groups[2],np.int32(budget(arm)['optimizer_steps_per_cohort']*cohort)), arm)
        groups = [{k:jnp.asarray(a) for k,a in g.items()} for g in groups]
    return groups[0], (groups[1],groups[2],jnp.int32(budget(arm)['optimizer_steps_per_cohort']*cohort))


def tactical(p,oracle,arm,catalog):
    if len(oracle['records'])!=512:raise ValueError('Familiar probe inventory')
    from pokemon_full.policy import distribution
    groups={};predictions=[]
    for first in range(0,512,8):
        records=oracle['records'][first:first+8];enc=[encode(r['player'],catalog,'matchup') for r in records]
        logits,values=forward(p,jnp.asarray(np.stack([x for x,_ in enc])),arm)
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


def analysis(panels, plan):
    pairs = [('control_learning',panels['control']['0'],panels['control']['262144']),
        ('nonlinear_learning',panels['nonlinear']['0'],panels['nonlinear']['262144']),
        ('nonlinear_vs_control',panels['control']['262144'],panels['nonlinear']['262144'])]
    comparisons = {name:{o:compare(a[o],b[o],plan['statistics']) for o in plan['opponents']} for name,a,b in pairs}
    def gate(cells, primary='tactical'):
        p = cells[primary]; s = plan['statistics']
        checks = {'point_gain':p['cutoff_aware_point_interval'][0]>=s['practical_improvement'],
            'adjusted_lower_positive':p['approximate_familywise_bootstrap_interval'][0]>0,
            'no_detected_regression':all(cells[o]['gate']!='regression' for o in ['random','base_power']),
            'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cells.values())}
        return {'checks':checks,'passed':all(checks.values())}
    return {'comparisons':comparisons,
        'learning':{a:gate(comparisons[a+'_learning']) for a in plan['arms']},
        'feature_benefit':gate(comparisons['nonlinear_vs_control']),
        'base_power_learning':{a:gate(comparisons[a+'_learning'],'base_power') for a in plan['arms']},
        'base_power_benefit':gate(comparisons['nonlinear_vs_control'],'base_power')}
